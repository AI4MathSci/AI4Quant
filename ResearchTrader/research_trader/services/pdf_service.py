"""
PDF processing service for extracting text from research papers
"""

import asyncio
import logging
import os
import tempfile
from datetime import datetime

import aiohttp

# Import PDF processing libraries
try:
    import PyPDF2
    from pdfminer.high_level import extract_text as pdfminer_extract_text

    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

from research_trader.models.summary import PaperText
from research_trader.services.cache import CacheService
from research_trader.utils.errors import ServiceError

# Configure logging
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Service for processing PDF files and extracting text"""

    def __init__(self):
        """Initialize the PDF processor"""
        if not PDFMINER_AVAILABLE:
            logger.warning("PDFMiner not available. PDF processing functionality will be limited.")

    async def download_pdf(self, pdf_url: str) -> bytes:
        """
        Download a PDF file from a URL

        Args:
            pdf_url: URL of the PDF to download

        Returns:
            Binary content of the PDF
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(pdf_url) as response:
                    if response.status != 200:
                        raise ServiceError(
                            detail=f"Failed to download PDF: HTTP status {response.status}",
                            service="pdf_processor",
                            status_code=503,
                        )
                    return await response.read()
        except aiohttp.ClientError as e:
            raise ServiceError(
                detail=f"Error downloading PDF: {str(e)}", service="pdf_processor", status_code=503
            )

    async def extract_text_from_bytes(self, pdf_bytes: bytes) -> str:
        """
        Extract text from PDF bytes

        Args:
            pdf_bytes: Binary content of the PDF

        Returns:
            Extracted text from the PDF
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp_path = tmp.name
            try:
                # Write PDF bytes to temp file
                tmp.write(pdf_bytes)
                tmp.flush()

                # Extract text from the temp file
                return await self.extract_text_from_file(tmp_path)

            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    async def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text from a PDF file

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text from the PDF
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        # Try pdfminer.six first (better text extraction)
        if PDFMINER_AVAILABLE:
            try:
                # Run extraction in a thread pool to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(None, pdfminer_extract_text, file_path)

                if text.strip():
                    return text
                logger.warning("PDFMiner extracted empty text, falling back to PyPDF2")
            except Exception as e:
                logger.warning(f"PDFMiner extraction failed: {str(e)}, falling back to PyPDF2")

        # Fall back to PyPDF2
        try:
            text = ""

            # Run in thread pool to avoid blocking
            def extract_with_pypdf():
                nonlocal text
                with open(file_path, "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(reader.pages)):
                        page = reader.pages[page_num]
                        text += page.extract_text() + "\n\n"
                return text

            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, extract_with_pypdf)

            if not text.strip():
                raise ValueError("Extracted text is empty")

            return text

        except Exception as e:
            raise ServiceError(
                detail=f"Failed to extract text from PDF: {str(e)}",
                service="pdf_processor",
                status_code=500,
            )

    async def get_paper_text(self, paper_id: str, pdf_url: str | None) -> str:
        """Get full text for a paper, downloading and processing if necessary"""
        # Check if we already have the paper text cached
        cached_text = await CacheService.get_cached_paper_text(paper_id)
        if cached_text:
            return cached_text.full_text

        # If no PDF URL provided, can't fetch the text
        if not pdf_url:
            raise ServiceError(
                detail="No PDF URL provided for paper", service="pdf_processor", status_code=400
            )

        # Make sure pdf_url is a string
        if not isinstance(pdf_url, str):
            logger.warning(f"PDF URL is not a string: {pdf_url}, converting to string")
            pdf_url = str(pdf_url)

        # Download and process the PDF
        try:
            pdf_bytes = await self.download_pdf(pdf_url)
            text = await self.extract_text_from_bytes(pdf_bytes)

            # If text extraction returned empty or very short text, it likely failed
            if len(text.strip()) < 100:
                raise ServiceError(
                    detail="Failed to extract meaningful text from PDF",
                    service="pdf_processor",
                    status_code=500,
                )

            # Create and cache the paper text object
            paper_text_obj = PaperText(
                paper_id=paper_id,
                title="Unknown",  # Will be updated when we have paper metadata
                abstract="",  # Will be updated when we have paper metadata
                full_text=text,
                extraction_date=datetime.now().isoformat(),
            )

            await CacheService.cache_paper_text(paper_id, paper_text_obj)

            # Return the extracted text
            return text
        except Exception as e:
            logger.exception(f"Error processing PDF: {str(e)}")
            raise ServiceError(
                detail=f"Could not process PDF: {str(e)}", service="pdf_processor", status_code=500
            )

    async def attempt_extract_sections(self, text: str) -> dict[str, str]:
        """
        Attempt to extract sections from paper text

        Args:
            text: Full text of the paper

        Returns:
            Dictionary of section names to section content
        """
        sections = {}

        # Common section headings in research papers
        common_sections = [
            "abstract",
            "introduction",
            "background",
            "related work",
            "methodology",
            "method",
            "approach",
            "experiment",
            "implementation",
            "evaluation",
            "result",
            "discussion",
            "conclusion",
            "reference",
            "appendix",
            "acknowledgment",
        ]

        # Try to identify sections based on common headings
        lines = text.split("\n")
        current_section = None
        current_content = []

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                if current_section:
                    current_content.append("")  # Preserve paragraph breaks
                continue

            # Check if this line is a section heading
            is_heading = False
            lowercase_line = line.lower()

            # Heuristics to identify section headings:
            # 1. Line is all uppercase
            # 2. Line starts with a number followed by a period or space
            # 3. Line matches a common section name
            # 4. Line is short (< 60 chars) and ends with a colon
            if (
                line.isupper()
                or (line[0].isdigit() and len(line) > 1 and (line[1] == "." or line[1] == " "))
                or any(
                    lowercase_line.startswith(section) or lowercase_line.endswith(section)
                    for section in common_sections
                )
                or (len(line) < 60 and line.endswith(":"))
            ):
                is_heading = True

                # Remove numbering and standardize the section name
                section_name = lowercase_line
                for i, char in enumerate(section_name):
                    if not char.isdigit() and char not in ".: ":
                        section_name = section_name[i:]
                        break

                section_name = section_name.strip(". :").lower()

                # Standardize section names
                for common in common_sections:
                    if common in section_name:
                        section_name = common
                        break

                # If we found a valid section name, save the previous section
                if section_name and section_name in common_sections:
                    if current_section and current_content:
                        sections[current_section] = "\n".join(current_content).strip()

                    current_section = section_name
                    current_content = []
                    continue

            # Add the line to the current section content
            if current_section:
                current_content.append(line)

        # Add the last section
        if current_section and current_content:
            sections[current_section] = "\n".join(current_content).strip()

        # If we couldn't identify any sections, create a single "text" section
        if not sections:
            sections["text"] = text

        return sections

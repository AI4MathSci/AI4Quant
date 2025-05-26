"""
Service for processing research papers: downloading, parsing, summarizing.
"""

import asyncio
import logging
import os
import tempfile
from datetime import datetime

import httpx
from PyPDF2 import PdfReader

from research_trader.config import settings
from research_trader.models.paper import Paper, PaperContent, PaperMetadata
from research_trader.services.cache_service import CacheService
from research_trader.services.openai_service import OpenAIService

logger = logging.getLogger(__name__)


class PaperProcessingService:
    """Orchestrates downloading, parsing, and summarizing papers."""

    def __init__(self, openai_service: OpenAIService, cache_service: CacheService):
        self.openai_service = openai_service
        self.cache_service = cache_service

    async def _download_pdf(self, pdf_url: str, paper_id: str) -> str | None:
        """Downloads PDF to a temporary file.
        Returns the path to the downloaded file or None on failure."""
        if not pdf_url:
            logger.warning(f"No PDF URL provided for paper {paper_id}.")
            return None

        try:
            # Use DOWNLOAD_TIMEOUT from settings
            async with httpx.AsyncClient(
                follow_redirects=True, timeout=settings.DOWNLOAD_TIMEOUT
            ) as client:
                logger.info(f"Downloading PDF for paper {paper_id} from {pdf_url}")
                response = await client.get(pdf_url)
                response.raise_for_status()

                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                    await response.aread()
                    temp_pdf.write(response.content)
                    logger.info(f"Successfully downloaded PDF for {paper_id} to {temp_pdf.name}")
                    return temp_pdf.name

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error downloading PDF for {paper_id} ({e.response.status_code}): {e.response.text}"
            )
            return None
        except httpx.RequestError as e:
            logger.error(f"Request error downloading PDF for {paper_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading PDF for {paper_id}: {e}", exc_info=True)
            return None

    async def _extract_text_from_pdf(self, pdf_path: str) -> str | None:
        """Extracts text content from a PDF file using PyPDF2."""
        logger.info(f"Extracting text from PDF: {pdf_path}")
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found for extraction: {pdf_path}")
            return None

        try:
            # Open the PDF file
            reader = PdfReader(pdf_path)
            num_pages = len(reader.pages)
            logger.debug(f"PDF '{os.path.basename(pdf_path)}' has {num_pages} pages.")

            # Extract text from each page
            all_text = []
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:  # Check if text extraction returned something
                        all_text.append(page_text)
                    else:
                        logger.warning(
                            f"No text extracted from page {i + 1} of {os.path.basename(pdf_path)}"
                        )
                except Exception as page_err:
                    logger.error(
                        f"Error extracting text from page {i + 1} of {os.path.basename(pdf_path)}: {page_err}"
                    )
                    # Optionally continue to next page or return None/partial text
                    continue

            full_text = "\n".join(all_text).strip()

            if not full_text:
                logger.warning(
                    f"Text extraction resulted in empty string for {os.path.basename(pdf_path)}"
                )
                return None  # Return None if no text could be extracted at all

            logger.info(
                f"Successfully extracted text from {os.path.basename(pdf_path)}. Length: {len(full_text)}"
            )
            return full_text
        except Exception as e:
            logger.error(f"Failed to process PDF file {pdf_path} with PyPDF2: {e}", exc_info=True)
            return None
        finally:
            # Clean up the temporary file
            try:
                if pdf_path and os.path.exists(pdf_path):
                    os.remove(pdf_path)
                    logger.debug(f"Removed temporary PDF file: {pdf_path}")
            except OSError as e:
                logger.error(f"Error removing temporary PDF file {pdf_path}: {e}")

    async def process_paper(
        self, paper_metadata: PaperMetadata, force_reprocess: bool = False
    ) -> Paper:
        """
        Processes a single paper: checks cache, downloads, parses, summarizes, and updates cache.

        Args:
            paper_metadata: The metadata of the paper to process.
            force_reprocess: If True, ignore cache and reprocess fully.

        Returns:
            The fully processed Paper object.

        Raises:
            ProcessingError: If a critical step in processing fails.
        """
        paper_id = paper_metadata.paper_id
        logger.info(f"Starting processing for paper: {paper_id} ('{paper_metadata.title[:50]}...')")

        cached_paper = None
        if not force_reprocess:
            cached_paper = await self.cache_service.get_paper(paper_id)
            if cached_paper and cached_paper.content and cached_paper.content.comprehensive_summary:
                logger.info(f"Paper {paper_id} found in cache with content. Skipping processing.")
                return cached_paper

        # Use cached paper if available but incomplete, otherwise create new
        paper = cached_paper or Paper(metadata=paper_metadata)

        # Ensure content object exists
        if paper.content is None:
            paper.content = PaperContent()

        # --- PDF Download and Text Extraction --- (Only if PDF URL exists and text is missing)
        pdf_path = None
        if paper_metadata.pdf_url and not paper.content.full_text:
            pdf_path = await self._download_pdf(str(paper_metadata.pdf_url), paper_id)
            if pdf_path:
                extracted_text = await self._extract_text_from_pdf(pdf_path)
                if extracted_text:
                    paper.content.full_text = extracted_text
                else:
                    logger.warning(
                        f"Failed to extract text for paper {paper_id}. Summarization will use abstract."
                    )
            else:
                logger.warning(
                    f"Failed to download PDF for paper {paper_id}. Summarization will use abstract."
                )

        # --- Summarization --- (Use full text if available, otherwise abstract)
        text_to_summarize = paper.content.full_text or paper_metadata.abstract
        if not text_to_summarize:
            logger.error(
                f"No text available (neither full text nor abstract) for summarization for paper {paper_id}. Cannot proceed."
            )
            raise Exception(f"No text content found for paper {paper_id}", paper_id=paper_id)

        # Generate summaries if missing or forced
        should_summarize = (
            force_reprocess
            or not paper.content.comprehensive_summary
            or not paper.content.structured_summary
        )

        if should_summarize:
            logger.info(f"Generating summaries for paper {paper_id}...")
            # Run summaries in parallel
            structured_summary_task = self.openai_service.generate_structured_summary(
                paper_text=text_to_summarize,
                title=paper_metadata.title,
                authors=paper_metadata.authors,
            )
            comprehensive_summary_task = self.openai_service.generate_comprehensive_summary(
                paper_text=text_to_summarize,
                title=paper_metadata.title,
                authors=paper_metadata.authors,
            )

            structured_summary, comprehensive_summary = await asyncio.gather(
                structured_summary_task, comprehensive_summary_task
            )

            if structured_summary:
                paper.content.structured_summary = structured_summary
                logger.info(f"Generated structured summary for {paper_id}.")
            else:
                logger.warning(f"Failed to generate structured summary for {paper_id}.")
                # Decide if this is a critical failure? Maybe not.

            if comprehensive_summary:
                paper.content.comprehensive_summary = comprehensive_summary
                logger.info(f"Generated comprehensive summary for {paper_id}.")
            else:
                logger.warning(f"Failed to generate comprehensive summary for {paper_id}.")
                # Decide if this is a critical failure? Often summaries are key.
                # raise ProcessingError(f"Failed to generate comprehensive summary for {paper_id}", paper_id=paper_id)

        # --- Update Cache ---
        # Update last_updated timestamp
        paper.last_updated = datetime.utcnow()
        await self.cache_service.set_paper(paper)
        logger.info(f"Finished processing and updated cache for paper: {paper_id}")

        return paper

    async def process_papers_batch(
        self, papers_metadata: list[PaperMetadata], force_reprocess: bool = False
    ) -> list[Paper]:
        """Processes a batch of papers concurrently."""
        tasks = [self.process_paper(meta, force_reprocess) for meta in papers_metadata]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_papers = []
        for i, result in enumerate(results):
            paper_id = papers_metadata[i].paper_id
            if isinstance(result, Exception):
                logger.error(f"Failed to process paper {paper_id}: {result}", exc_info=result)
                # Optionally return partial results or raise a summary exception
            elif isinstance(result, Paper):
                processed_papers.append(result)
            else:
                logger.error(f"Unexpected result type for paper {paper_id}: {type(result)}")

        return processed_papers

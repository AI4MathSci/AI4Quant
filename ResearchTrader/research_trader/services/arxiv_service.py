"""
Service for interacting with the ArXiv API.
"""

import asyncio
import logging
import time
from datetime import datetime

import arxiv
from arxiv import Result as ArxivResult
from pydantic import HttpUrl

from research_trader.config import settings
from research_trader.models.paper import PaperMetadata
from research_trader.utils.errors import ServiceError

logger = logging.getLogger(__name__)


class ArxivService:
    """Client for interacting with the ArXiv API using the arxiv library."""

    def __init__(self):
        """
        Initializes the ArxivService.
        Uses ARXIV_REQUEST_INTERVAL from settings.
        """
        self.last_request_time = 0
        # Use setting for interval
        self.min_request_interval: float = settings.ARXIV_REQUEST_INTERVAL

    async def _enforce_rate_limit(self):
        """Ensure we don't exceed ArXiv's rate limits."""
        now = time.time()
        time_since_last = now - self.last_request_time

        if time_since_last < self.min_request_interval:
            sleep_duration = self.min_request_interval - time_since_last
            logger.debug(f"ArXiv rate limit: sleeping for {sleep_duration:.2f} seconds.")
            await asyncio.sleep(sleep_duration)

        self.last_request_time = time.time()

    def _parse_arxiv_id(self, entry_id: str) -> str:
        """Extracts the core ArXiv ID from the entry_id URL."""
        # Example: http://arxiv.org/abs/2301.00001v1 -> 2301.00001v1
        return entry_id.split("/")[-1]

    def _convert_to_paper_metadata(self, result: ArxivResult) -> PaperMetadata:
        """Convert arxiv.Result to our PaperMetadata model."""
        paper_id = self._parse_arxiv_id(result.entry_id)
        pdf_url = getattr(result, "pdf_url", None)
        primary_category = getattr(result, "primary_category", None)
        tags = [primary_category] if primary_category else []
        tags.extend(getattr(result, "categories", []))

        # ArXiv library might return datetime or date object for published
        published_date = result.published
        if not isinstance(published_date, datetime):
            # If it's just a date, set time to midnight UTC for consistency
            published_date = datetime.combine(published_date, datetime.min.time())
            # Consider adding timezone info if necessary, e.g., .replace(tzinfo=timezone.utc)

        return PaperMetadata(
            paper_id=paper_id,
            title=result.title,
            authors=[author.name for author in result.authors],
            abstract=result.summary,
            published_date=published_date,
            pdf_url=HttpUrl(pdf_url) if pdf_url else None,
            source_url=HttpUrl(result.entry_id) if result.entry_id else None,
            tags=list(set(tags)),  # Remove duplicates
        )

    async def search_papers(
        self, query: str, max_results: int = settings.MAX_PAPERS_FETCH
    ) -> list[PaperMetadata]:
        """
        Fetch the latest papers from ArXiv matching the query.
        Focuses on quantitative finance (q-fin) category.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.

        Returns:
            List of PaperMetadata objects.

        Raises:
            ServiceError: If ArXiv API request fails.
        """
        await self._enforce_rate_limit()

        # Prepend q-fin category search unless query explicitly targets another
        # Basic check, might need refinement based on expected query patterns
        if "cat:" not in query.lower() and "category:" not in query.lower():
            search_query = f"cat:q-fin* AND ({query})"  # Wrap user query for clarity
        else:
            search_query = query

        logger.info(f"Searching ArXiv with query: '{search_query}', max_results={max_results}")

        try:
            # Run the search in a separate thread to avoid blocking the event loop
            # as the arxiv library's search is synchronous.
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )

            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, list, search.results())

            papers_metadata = [self._convert_to_paper_metadata(result) for result in results]
            logger.info(f"Found {len(papers_metadata)} papers from ArXiv.")
            return papers_metadata

        except Exception as e:
            error_msg = f"Error fetching papers from ArXiv: {e.__class__.__name__}: {e}"
            logger.error(error_msg)
            raise ServiceError(detail=error_msg, service="arxiv", status_code=503)

    async def get_paper_by_id(self, paper_id: str) -> PaperMetadata | None:
        """
        Fetch a specific paper by its ID (e.g., '2301.00001v1').

        Args:
            paper_id: ArXiv paper ID (should be the core ID without URL prefix).

        Returns:
            PaperMetadata object or None if not found.

        Raises:
            ServiceError: If ArXiv API request fails.
        """
        await self._enforce_rate_limit()

        logger.info(f"Fetching ArXiv paper by ID: {paper_id}")

        try:
            search = arxiv.Search(id_list=[paper_id])
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, list, search.results())

            if not results:
                logger.warning(f"Paper with ID {paper_id} not found on ArXiv.")
                return None

            metadata = self._convert_to_paper_metadata(results[0])
            logger.info(f"Successfully fetched metadata for paper {paper_id}.")
            return metadata

        except Exception as e:
            error_msg = f"Error fetching paper {paper_id} from ArXiv: {e.__class__.__name__}: {e}"
            logger.error(error_msg)
            raise ServiceError(detail=error_msg, service="arxiv", status_code=503)

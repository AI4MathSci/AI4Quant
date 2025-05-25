"""
API Endpoints for managing and interacting with research papers.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException, Path, Query
from pydantic import BaseModel, Field

from research_trader.config import settings
from research_trader.models.paper import Paper  # Import the consolidated Paper model
from research_trader.services import (
    ArxivService,
    CacheService,
    OpenAIService,
    PaperProcessingService,
)
from research_trader.utils.errors import ServiceError

logger = logging.getLogger(__name__)

# --- Dependency Injection Setup ---
arxiv_service = ArxivService()
cache_service = CacheService()
openai_service = OpenAIService()
paper_processing_service = PaperProcessingService(openai_service, cache_service)


# --- Router Definition ---
router = APIRouter(
    prefix="/papers",
    tags=["Papers"],
)

# --- Request/Response Models ---


class SearchRequest(BaseModel):
    query: str = Field(
        ..., description="Search query for ArXiv.", example="deep reinforcement learning trading"
    )
    max_results: int = Field(
        default=settings.MAX_PAPERS_FETCH,
        ge=1,
        le=50,
        description="Maximum number of papers to fetch and process.",
    )
    force_reprocess: bool = Field(
        default=False, description="Force reprocessing even if papers are cached."
    )


class PaperSummaryResponse(BaseModel):
    """Simplified Paper response for search results, focusing on metadata and summary."""

    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    published_date: datetime | None
    pdf_url: str | None  # Use str for response model
    source_url: str | None  # Use str for response model
    tags: list[str]
    structured_summary: dict[str, str | list[str]] | None = Field(
        None,
        description="Structured summary (Objective: str, Methods: List[str], Results: List[str], Conclusions: List[str]).",
    )
    comprehensive_summary: str | None = Field(
        None, description="Comprehensive summary of the paper."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "paper_id": "2301.00001v1",
                "title": "Example Paper Title",
                "authors": ["Author One", "Author Two"],
                "abstract": "This is the abstract...",
                "published_date": "2023-01-01T12:00:00Z",
                "pdf_url": "http://arxiv.org/pdf/2301.00001v1.pdf",
                "source_url": "http://arxiv.org/abs/2301.00001v1",
                "tags": ["cs.LG", "q-fin.CP"],
                "structured_summary": {
                    "objective": "This paper discusses...",
                    "methods": ["Method One", "Method Two"],
                    "results": ["Result One", "Result Two"],
                    "conclusions": ["Conclusion One", "Conclusion Two"],
                },
                "comprehensive_summary": "This paper discusses...",
            }
        }
    }

    @classmethod
    def from_paper(cls, paper: Paper) -> "PaperSummaryResponse":
        """Factory method to create response from Paper model."""
        return cls(
            paper_id=paper.metadata.paper_id,
            title=paper.metadata.title,
            authors=paper.metadata.authors,
            abstract=paper.metadata.abstract,
            published_date=paper.metadata.published_date,
            pdf_url=str(paper.metadata.pdf_url) if paper.metadata.pdf_url else None,
            source_url=str(paper.metadata.source_url) if paper.metadata.source_url else None,
            tags=paper.metadata.tags,
            summary=paper.content.comprehensive_summary if paper.content else None,
        )


# --- Endpoints ---


@router.post("/", response_model=list[PaperSummaryResponse], status_code=200)
async def search_and_process_papers(
    search_request: SearchRequest, background_tasks: BackgroundTasks
) -> list[PaperSummaryResponse]:
    """
    Search ArXiv for papers, process them (download, parse, summarize), cache, and return summaries.

    Processing happens in the background for potentially long-running tasks.
    The initial response returns metadata found, background tasks populate cache with full content.
    """
    try:
        logger.info(
            f"Received search request: query='{search_request.query}', max_results={search_request.max_results}"
        )
        # 1. Search ArXiv for metadata
        papers_metadata = await arxiv_service.search_papers(
            query=search_request.query, max_results=search_request.max_results
        )

        if not papers_metadata:
            logger.info(f"No papers found for query: '{search_request.query}'")
            return []

        logger.info(f"Found {len(papers_metadata)} paper(s). Triggering background processing.")

        # 2. Trigger background processing for all found papers
        # Note: This approach returns quickly but doesn't guarantee processing is finished
        # when the response is sent. The client might need to poll /papers/{id} later
        # or we could implement websockets/SSE for progress updates.
        background_tasks.add_task(
            paper_processing_service.process_papers_batch,
            papers_metadata,
            force_reprocess=search_request.force_reprocess,
        )

        # 3. Return initial metadata/summary (if available in cache quickly)
        # We can do a quick cache check here for papers already processed
        initial_papers = []
        for meta in papers_metadata:
            cached_paper = await cache_service.get_paper(meta.paper_id)
            if cached_paper:
                initial_papers.append(PaperSummaryResponse.from_paper(cached_paper))
            else:
                # Create a temporary Paper object with only metadata for the response
                temp_paper = Paper(metadata=meta)
                initial_papers.append(PaperSummaryResponse.from_paper(temp_paper))

        logger.info(f"Returning initial response for {len(initial_papers)} papers.")
        return initial_papers

    except ServiceError as e:
        logger.error(f"Service error during paper search: {e.detail}", exc_info=True)
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        logger.error(f"Unexpected error during paper search: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An internal server error occurred during paper search."
        )


@router.get("/", response_model=list[PaperSummaryResponse])
async def get_cached_papers() -> list[PaperSummaryResponse]:
    """
    Retrieve summaries of all papers currently held in the cache.
    """
    logger.info("Request to retrieve all cached papers.")
    try:
        all_papers = await cache_service.get_all_papers()
        logger.info(f"Found {len(all_papers)} papers in cache.")
        # Sort by last updated or published date?
        # all_papers.sort(key=lambda p: p.last_updated, reverse=True)
        return [PaperSummaryResponse.from_paper(paper) for paper in all_papers]
    except Exception as e:
        logger.error(f"Unexpected error retrieving cached papers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve cached papers.")


@router.get("/{paper_id}", response_model=Paper)
async def get_paper_details(
    paper_id: str = Path(..., description="The ArXiv ID of the paper (e.g., '2301.00001v1')."),
    force_reprocess: bool = Query(
        False, description="Force reprocessing even if the paper is cached."
    ),
) -> Paper:
    """
    Retrieve the full details of a specific paper, processing it if necessary.

    If the paper is not in the cache or `force_reprocess` is True,
    it will attempt to fetch from ArXiv and process it.
    """
    logger.info(f"Request for paper details: {paper_id}, force_reprocess={force_reprocess}")
    cached_paper = None
    if not force_reprocess:
        cached_paper = await cache_service.get_paper(paper_id)
        if cached_paper:
            logger.info(f"Paper {paper_id} found in cache.")
            return cached_paper

    logger.info(f"Paper {paper_id} not in cache or reprocessing forced. Fetching from ArXiv.")
    try:
        # Fetch metadata first
        paper_metadata = await arxiv_service.get_paper_by_id(paper_id)
        if not paper_metadata:
            raise HTTPException(
                status_code=404, detail=f"Paper with ID '{paper_id}' not found on ArXiv."
            )

        # Process the paper (download, parse, summarize)
        processed_paper = await paper_processing_service.process_paper(
            paper_metadata, force_reprocess=True
        )  # Force if cache miss or explicit
        return processed_paper

    except (ProcessingError, ServiceError) as e:
        logger.error(f"Error processing paper {paper_id}: {e}", exc_info=True)
        status_code = e.status_code if hasattr(e, "status_code") else 500
        raise HTTPException(status_code=status_code, detail=str(e))
    except HTTPException as e:  # Re-raise specific HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Unexpected error retrieving paper {paper_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while retrieving paper {paper_id}.",
        )

"""
Router for paper Q&A endpoints.
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from research_trader.models.paper import Paper  # Use the consolidated Paper model
from research_trader.services import (
    ArxivService,  # Needed if we decide to fetch missing papers
    CacheService,
    OpenAIService,
    PaperProcessingService,  # Needed if we decide to process missing papers
)

# Assuming service instances are created elsewhere and injected, or created here like in papers.py
# For consistency with papers.py, let's instantiate them here for now.
arxiv_service = ArxivService()
cache_service = CacheService()
openai_service = OpenAIService()
paper_processing_service = PaperProcessingService(openai_service, cache_service)

logger = logging.getLogger(__name__)

# --- Request/Response Models ---


class QARequest(BaseModel):
    """Request model for Q&A"""

    question: str = Field(..., description="The question to ask about the papers.")
    paper_ids: list[str] = Field(
        ...,
        description="List of paper IDs (e.g., '2301.00001v1') to use as context. These papers must have been previously processed/cached.",
        min_items=1,
    )


class QAResponse(BaseModel):
    """Response model for Q&A"""

    question: str
    answer: str | None = Field(
        None, description="The generated answer. Can be None if generation failed."
    )
    context_paper_ids: list[str] = Field(description="List of paper IDs used for context.")


# --- Router Definition ---
router = APIRouter(prefix="/qa", tags=["Q&A"])


# --- Endpoint ---


@router.post("/", response_model=QAResponse)
async def ask_question_about_papers(request: QARequest):
    """
    Ask a question based on the content of previously processed/cached papers.

    Requires paper IDs to be provided. The system will fetch the cached
    content (abstract, summaries, potentially full text snippets if implemented in context formatting)
    for these papers and use it to answer the question.

    If any requested paper ID is not found in the cache, an error will be returned.
    """
    logger.info(
        f"Received Q&A request: question='{request.question[:50]}...', paper_ids={request.paper_ids}"
    )

    # 1. Fetch context papers from cache
    context_papers: list[Paper] = []
    missing_ids = []
    fetch_tasks = [cache_service.get_paper(pid) for pid in request.paper_ids]
    results = await asyncio.gather(*fetch_tasks)

    for i, paper in enumerate(results):
        paper_id = request.paper_ids[i]
        if paper and paper.content:  # Ensure paper and its content exists
            context_papers.append(paper)
        else:
            logger.warning(f"Paper ID {paper_id} not found in cache or lacks content for Q&A.")
            missing_ids.append(paper_id)

    # Handle missing papers - we require papers to be processed beforehand
    if missing_ids:
        raise HTTPException(
            status_code=404,
            detail=f"The following paper IDs were not found in the cache or lack processed content: {', '.join(missing_ids)}. Please process them first via the /papers/{{paper_id}} endpoint.",
        )

    if not context_papers:  # Should not happen if missing_ids is handled, but as a safeguard
        raise HTTPException(
            status_code=400, detail="No valid context papers found for the provided IDs."
        )

    # 2. Generate answer using OpenAI Service
    logger.info(f"Generating answer using {len(context_papers)} papers as context.")
    try:
        answer = await openai_service.answer_question(request.question, context_papers)

        if answer is None:
            # Handle cases where the service explicitly returns None (e.g., generation failure)
            logger.error(
                f"OpenAI service failed to generate an answer for question: '{request.question[:50]}...'"
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to generate answer due to an internal service error.",
            )

    except Exception as e:
        # Catch potential exceptions from the service layer
        logger.error(f"Error during OpenAI answer generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An error occurred while generating the answer."
        )

    # 3. Format and return response
    response = QAResponse(
        question=request.question,
        answer=answer,
        context_paper_ids=[p.paper_id for p in context_papers],
    )
    logger.info(f"Successfully generated answer for question: '{request.question[:50]}...'")
    return response

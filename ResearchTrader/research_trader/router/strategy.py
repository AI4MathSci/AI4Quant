"""
Router for trading strategy generation endpoints.
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from research_trader.models.paper import Paper, StrategyOutput  # Import StrategyOutput
from research_trader.services import (
    ArxivService,  # If needed for checking existence
    CacheService,
    OpenAIService,
    PaperProcessingService,  # If needed for on-the-fly processing (discouraged)
)

# Assuming service instances are created elsewhere and injected, or created here like in papers.py
# For consistency with papers.py, let's instantiate them here for now.
arxiv_service = ArxivService()
cache_service = CacheService()
openai_service = OpenAIService()
paper_processing_service = PaperProcessingService(openai_service, cache_service)

logger = logging.getLogger(__name__)

# --- Request/Response Models ---


class StrategyGenerationRequest(BaseModel):
    """Request model for generating a trading strategy outline."""

    paper_ids: list[str] = Field(
        ...,
        description="List of paper IDs (e.g., '2301.00001v1') to use as context. These papers must have been previously processed/cached.",
        min_items=1,
    )
    strategy_prompt: str = Field(
        ...,
        description="Detailed prompt describing the desired strategy, including market, timeframe, risk, specific indicators, or logic inspired by the papers.",
        example="Generate a mean-reversion strategy for AAPL stock on the hourly timeframe, using Bollinger Bands and RSI, inspired by the volatility analysis in paper 2301.00001v1.",
    )
    # Removed market, timeframe, risk_profile as separate fields - should be in the prompt.
    # additional_context: Optional[str] = Field(None, description="Any other specific requirements or constraints.")


class StrategyGenerationResponse(BaseModel):
    """Response model for strategy generation, embedding the structured output."""

    strategy: StrategyOutput | None = Field(
        None, description="The generated structured strategy output."
    )
    context_paper_ids: list[str] = Field(description="List of paper IDs used for context.")
    notes: str = Field(description="Notes about the generation process or potential issues.")


# --- Router Definition ---
router = APIRouter(prefix="/strategy", tags=["Strategy Generation"])


# --- Endpoint ---


@router.post("/", response_model=StrategyGenerationResponse)
async def generate_strategy_outline(
    request: StrategyGenerationRequest,
) -> StrategyGenerationResponse:
    """
    Generate a structured Python trading strategy outline based on the content of
    previously processed/cached papers and a user prompt.

    Requires paper IDs to be provided. The system fetches cached content
    for these papers to inform the strategy generation.

    If any requested paper ID is not found in the cache, an error is returned.
    The generated output is a *conceptual outline* and not production-ready.
    """
    logger.info(
        f"Received strategy generation request: prompt='{request.strategy_prompt[:50]}...', paper_ids={request.paper_ids}"
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
            logger.warning(
                f"Paper ID {paper_id} not found in cache or lacks content for strategy generation."
            )
            missing_ids.append(paper_id)

    # Handle missing papers
    if missing_ids:
        raise HTTPException(
            status_code=404,
            detail=f"The following paper IDs were not found in the cache or lack processed content: {', '.join(missing_ids)}. Please process them first via the /papers/{{paper_id}} endpoint.",
        )

    if not context_papers:  # Safeguard
        raise HTTPException(
            status_code=400, detail="No valid context papers found for the provided IDs."
        )

    # 2. Generate strategy using OpenAI Service
    logger.info(
        f"Generating structured strategy using {len(context_papers)} papers and prompt: '{request.strategy_prompt[:50]}...'"
    )
    notes = "Strategy generation initiated."
    generated_strategy_output: StrategyOutput | None = None
    try:
        # This now returns a StrategyOutput object or None
        generated_strategy_output = await openai_service.generate_strategy(
            papers=context_papers, strategy_prompt=request.strategy_prompt
        )

        if generated_strategy_output is None:
            notes = "Strategy generation failed or produced no output."
            logger.error(
                f"OpenAI service failed to generate strategy for prompt: '{request.strategy_prompt[:50]}...'"
            )
        else:
            notes = "Structured strategy outline generated successfully."

    except Exception as e:
        # Catch potential exceptions from the service layer
        logger.error(f"Error during OpenAI strategy generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An error occurred while generating the strategy code."
        )

    # 3. Format and return response
    response = StrategyGenerationResponse(
        strategy=generated_strategy_output,  # Assign the generated object here
        context_paper_ids=[p.paper_id for p in context_papers],
        notes=notes,
    )
    logger.info(
        f"Successfully completed strategy generation request for prompt: '{request.strategy_prompt[:50]}...'"
    )
    return response

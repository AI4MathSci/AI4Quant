"""
Paper data models for ResearchTrader
"""

from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl


class PaperMetadata(BaseModel):
    """Basic metadata retrieved from ArXiv."""

    paper_id: str = Field(..., description="Unique identifier for the paper (e.g., ArXiv ID).")
    title: str = Field(..., description="Title of the paper.")
    authors: list[str] = Field(default_factory=list, description="List of author names.")
    abstract: str = Field(..., description="Abstract of the paper.")
    published_date: datetime | None = Field(None, description="Date the paper was published.")
    pdf_url: HttpUrl | None = Field(None, description="Direct URL to the paper's PDF.")
    source_url: HttpUrl | None = Field(
        None, description="URL to the paper's page (e.g., ArXiv page)."
    )
    tags: list[str] = Field(default_factory=list, description="Relevant tags or categories.")


class PaperContent(BaseModel):
    """Extracted content and analysis of the paper."""

    full_text: str | None = Field(None, description="Full text extracted from the PDF.")
    structured_summary: dict[str, str | list[str]] | None = Field(
        None,
        description="Structured summary (Objective: str, Methods: List[str], Results: List[str], Conclusions: List[str]).",
    )
    comprehensive_summary: str | None = Field(
        None, description="LLM-generated comprehensive summary."
    )


class Paper(BaseModel):
    """Consolidated representation of a research paper."""

    metadata: PaperMetadata
    content: PaperContent | None = None  # Content might be loaded lazily
    last_updated: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp of last update in cache."
    )

    # Allow easy access to metadata fields
    @property
    def paper_id(self) -> str:
        return self.metadata.paper_id

    @property
    def title(self) -> str:
        return self.metadata.title

    # Add other properties as needed


class StrategyOutput(BaseModel):
    """Structured output for generated trading strategies."""

    python_code: str | None = Field(
        None, description="Conceptual Python code outline for the strategy."
    )
    pseudocode: str | None = Field(
        None, description="High-level pseudocode or logical steps of the strategy."
    )
    strategy_description: str | None = Field(
        None, description="Explanation of the strategy, its goals, and paper inspirations."
    )
    how_to_use: str | None = Field(
        None, description="Notes on implementation, data, parameters, and limitations."
    )


class PaperList(BaseModel):
    """Response model for list of papers"""

    papers: list[Paper]
    total_count: int
    query: str

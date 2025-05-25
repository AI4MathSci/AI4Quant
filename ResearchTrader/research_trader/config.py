"""
Configuration settings for the ResearchTrader application
Uses Pydantic BaseSettings for environment variables handling
"""

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # --- API Endpoints & Keys ---
    ARXIV_API_URL: str = "https://export.arxiv.org/api/query"
    OPENAI_API_KEY: str = Field(..., description="Your OpenAI API Key.")
    OPENAI_MODEL: str = Field(
        "gpt-3.5-turbo", description="Default OpenAI model for chat completions."
    )
    OPENAI_BASE_URL: str = Field(
        "https://api.openai.com/v1", description="Base URL for OpenAI API."
    )

    # --- Caching ---
    ENABLE_CACHE: bool = Field(True, description="Enable/disable in-memory caching.")
    CACHE_TTL: int = Field(3600, description="Default cache time-to-live in seconds (1 hour).")
    # REDIS_URL: str | None = Field(None, description="Optional Redis URL for persistent caching.")

    # --- Rate Limiting & Intervals ---
    # ARXIV_RATE_LIMIT: int = 60 # Requests per minute - Not directly used, interval is more practical
    ARXIV_REQUEST_INTERVAL: float = Field(
        3.0, ge=1.0, description="Minimum seconds between ArXiv API requests."
    )
    # OPENAI_RATE_LIMIT: int = 100 # Tokens per minute - Hard to enforce without token counting

    # --- Timeouts ---
    ARXIV_TIMEOUT: float = Field(
        20.0, ge=5.0, description="Timeout for ArXiv API calls in seconds."
    )
    OPENAI_TIMEOUT: float = Field(
        120.0, ge=10.0, description="Timeout for OpenAI API calls in seconds."
    )
    DOWNLOAD_TIMEOUT: float = Field(
        60.0, ge=10.0, description="Timeout for downloading PDF files in seconds."
    )

    # --- Application Behavior ---
    MAX_PAPERS_FETCH: int = Field(
        10, ge=5, le=100, description="Maximum number of papers to fetch from ArXiv in one search."
    )
    DEFAULT_PAPERS_FETCH: int = Field(
        10, ge=1, le=50, description="Default number of papers to fetch if not specified."
    )
    # Removed MAX_PAPERS/DEFAULT_PAPERS, renamed for clarity

    # --- LLM Context Sizes ---
    # Max length of text sent to LLM for summarization/analysis
    MAX_TEXT_LENGTH_FOR_SUMMARY: int = Field(
        50000, ge=1000, description="Max characters of paper text to use for summary generation."
    )
    # Max length of combined context from *each paper* sent for Q&A
    MAX_CONTEXT_PER_PAPER_FOR_QA: int = Field(
        30000, ge=500, description="Max characters of context *per paper* to send for Q&A."
    )
    # Max length of combined context from all papers sent for strategy generation
    MAX_CONTEXT_LENGTH_FOR_STRATEGY: int = Field(
        30000,
        ge=1000,
        description="Max characters of *total context* to send for strategy generation.",
    )

    # --- Logging ---
    LOG_LEVEL: str = Field("INFO", description="Logging level (e.g., DEBUG, INFO, WARNING).")

    # --- Server Configuration ---
    API_HOST: str = Field("0.0.0.0", description="Host for the FastAPI backend server.")
    API_PORT: int = Field(8000, description="Port for the FastAPI backend server.")
    API_RELOAD: bool = Field(
        True, description="Enable auto-reload for the FastAPI backend server (development)."
    )

    GRADIO_SERVER_NAME: str | None = Field(
        "0.0.0.0",
        description="Server name for the Gradio frontend (None for default localhost). Use 0.0.0.0 to allow network access.",
    )
    GRADIO_SERVER_PORT: int = Field(7860, description="Port for the Gradio frontend server.")

    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra fields from env file
        case_sensitive = False  # Allow lowercase env vars
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Export settings instance
settings = get_settings()

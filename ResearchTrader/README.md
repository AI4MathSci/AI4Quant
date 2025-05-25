# ResearchTrader

ResearchTrader is a web application designed to accelerate the journey from quantitative finance research to actionable insights. It helps users discover, explore, and operationalize the latest ArXiv q-fin papers through a user-friendly interface.

## Key Features

*   **Targeted Discovery:** Search ArXiv specifically within quantitative finance (q-fin) categories using free-text queries to find relevant research papers.
*   **Automated Processing:** Fetched papers are automatically processed in the background:
    *   PDFs are downloaded (when available).
    *   Full text is extracted (requires `pypdf` or similar).
    *   Structured summaries (Objective, Methods, Results, Conclusions) are generated via LLM.
    *   Comprehensive, human-readable summaries tailored for quants are generated via LLM.
*   **Efficient Caching:** Processed papers (metadata, text, summaries) are cached in memory for quick retrieval.
*   **Interactive Exploration:**
    *   View search results in a clear table format.
    *   Select individual papers to see detailed metadata and generated summaries.
*   **Contextual Q&A:** Ask natural language questions based on the content of one or more selected papers from the cache. The system uses the processed paper information (abstracts, summaries) to generate answers.
*   **Strategy Outline Generation:** Generate conceptual Python trading strategy outlines based on the insights from selected cached papers and a user-provided prompt. The output includes descriptions, pseudocode, Python code snippets, and usage notes.

## Tech Stack

*   **Backend:** Python, FastAPI, Pydantic
*   **Frontend:** Gradio
*   **LLM Integration:** OpenAI API (via `httpx`)
*   **Paper Source:** ArXiv API (via `arxiv` library)
*   **PDF Parsing:** Requires external library (e.g., `pypdf`) - *not included by default*
*   **Package Management:** `uv`
*   **Code Formatting:** `black`, `isort`
*   **Testing:** `pytest`

## How to Use (Gradio Interface)

1.  **Start the Backend & Frontend:** Follow the instructions in the [Development](#development) section below to run both the FastAPI server and the Gradio app. Access the Gradio UI (typically `http://localhost:7860`).
2.  **Search for Papers:**
    *   Navigate to the "Search & Explore Papers" tab.
    *   Enter your search keywords (e.g., "volatility forecasting", "transformer trading").
    *   Adjust the "Max Papers to Fetch" slider if needed.
    *   Click "Search Papers". The application will fetch metadata from ArXiv and trigger background processing (PDF download, text extraction, summarization). Initial results (metadata) will appear in the table.
3.  **Explore Individual Papers:**
    *   Click on any row in the "Search Results" table.
    *   The "Selected Paper Details" section below the table will populate with the paper's title, summary/abstract, and structured analysis (Objective, Methods, etc.) once processing is complete. (Processing might take some time, especially for the first fetch).
4.  **Select Papers for Context:**
    *   Locate the "Available Papers for Context" section below the search results table. This shows papers that have been processed and are available in the cache.
    *   Click "Refresh List from Cache" to ensure it's up-to-date.
    *   Use the checkboxes to select one or more papers you want to use as context for Q&A or strategy generation.
5.  **Ask Questions (Q&A Tab):**
    *   Go to the "Q&A" tab.
    *   Ensure you have selected papers in the "Available Papers for Context" list on the previous tab.
    *   Type your question about the selected papers into the "Your Question" box.
    *   Click "Ask Question". The answer generated based on the context of the selected papers will appear below.
6.  **Generate Strategy Outlines (Strategy Generation Tab):**
    *   Go to the "Strategy Generation" tab.
    *   Ensure you have selected papers in the "Available Papers for Context" list.
    *   Write a detailed prompt in the "Strategy Prompt" box, describing the desired strategy and referencing concepts from the selected papers (mentioning their IDs if helpful).
    *   Click "Generate Strategy Outline". The generated output (Description, Pseudocode, Usage Notes, Python Code) will appear below.

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) - A fast Python package installer and resolver

### Setup with uv

1. Clone the repository:
```bash
git clone https://github.com/yourusername/research_trader.git # Replace with your repo URL
cd research_trader
```

2. Create a virtual environment with uv:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies with uv:
```bash
# Base dependencies
uv pip install -e .
# Development dependencies (formatting, linting, testing)
uv pip install -e ".[dev]"
# Optional: Add a PDF parsing library if you need full-text extraction
# e.g., uv pip install pypdf
```

4. Create a `.env` file based on the provided template:
```bash
cp .env.example .env
# Edit .env to add your OpenAI API key and any other required settings
```

## Development

### Running the Backend

Start the FastAPI server (make sure your `.env` file is configured):

```bash
# Using uvicorn directly (recommended for reload)
uvicorn research_trader.main:app --reload --host 0.0.0.0 --port 8000
# Or using the main.py entry point (may respect settings from .env for host/port/reload)
# python research_trader/main.py
```

The API will be available at `http://localhost:8000` (or as configured).

API documentation (Swagger UI and ReDoc) is available at:
- `http://localhost:8000/swagger`
- `http://localhost:8000/docs`

### Running the Frontend

Ensure the backend API is running first.

Start the Gradio frontend:

```bash
python research_trader/app.py
```

The frontend will typically be available at `http://localhost:7860`.

### Testing

Run tests with pytest:

```bash
# Make sure test dependencies are installed
uv pip install -e ".[test]"
pytest
```

### Code Formatting

Format code with black and isort:

```bash
black research_trader tests
isort research_trader tests
```

## API Endpoints (v0.2.0+)

The API provides the core functionalities backing the Gradio app:

| Endpoint          | Method | Description                                                                  |
|-------------------|--------|------------------------------------------------------------------------------|
| `/papers/`        | POST   | Search ArXiv, trigger background processing, return initial paper summaries. |
| `/papers/`        | GET    | Retrieve summaries of all papers currently in the cache.                     |
| `/papers/{id}`    | GET    | Retrieve full details of a specific paper (processing if needed).            |
| `/qa/`            | POST   | Ask a question based on the content of specified (cached) paper IDs.         |
| `/strategy/`      | POST   | Generate a structured Python trading strategy outline based on specified papers. |

Refer to the API documentation (`/docs` or `/swagger`) for detailed request/response models.

## Project Structure (v0.2.0+)

```
research_trader/
├── main.py                  # FastAPI application instance & startup
├── app.py                   # Gradio frontend application
├── config.py                # Pydantic settings (loads .env)
├── models/                  # Pydantic data models
│   ├── __init__.py
│   └── paper.py             # Core Paper, PaperMetadata, PaperContent, StrategyOutput models
├── router/                  # API routers (FastAPI)
│   ├── __init__.py
│   ├── papers.py            # Endpoints for /papers/ resource
│   ├── qa.py                # Endpoint for /qa/ resource
│   └── strategy.py          # Endpoint for /strategy/ resource
├── services/                # Business logic layer
│   ├── __init__.py
│   ├── arxiv_service.py     # Interacts with ArXiv API
│   ├── cache_service.py     # In-memory caching logic
│   ├── openai_service.py    # Interacts with OpenAI API
│   └── paper_processing_service.py # Orchestrates paper download, parsing, summary
└── utils/                   # Utility functions and custom errors
    ├── __init__.py
    └── errors.py            # Custom error classes

.env.example                 # Example environment variables
.gitignore
README.md
pyproject.toml               # Project configuration and dependencies (used by uv)
# tests/                     # Unit and integration tests
```

## Configuration

The application uses environment variables for configuration, managed by Pydantic settings in `config.py`. Key variables (like `OPENAI_API_KEY`, timeouts, LLM context lengths, server ports) should be set in a `.env` file (copy from `.env.example`).

*Note*: PDF parsing requires an external library (e.g., `pypdf`). Ensure you install one if you intend to use the full-text extraction feature. If not installed or if extraction fails, the application will fall back to using the paper's abstract for summaries, Q&A, and strategy generation.

## License

MIT

## Acknowledgments

- ArXiv API for research paper access
- OpenAI for language model capabilities
- Gradio for the web interface framework
- FastAPI for the backend API framework
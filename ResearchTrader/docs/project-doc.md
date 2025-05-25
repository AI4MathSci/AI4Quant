# ResearchTrader Project Documentation

## 1. Project Overview

ResearchTrader is a Gradio-based web application designed to help quantitative finance professionals discover, explore, and operationalize the latest ArXiv quantitative finance research. The platform bridges the gap between academic research and practical trading strategies by providing free-text search, AI-powered summaries, natural language Q&A, and automatic Python strategy generation.

## 2. Objectives

- **Discover**: Find the most relevant ArXiv q-fin papers matching user queries
- **Analyze & Summarize**: Extract trading-focused insights from papers including methods, results, and practical applications
- **Explore**: Ask natural-language questions about papers and receive evidence-based answers
- **Generate**: Produce executable Python trading strategies inspired by research findings

## 3. Key Features

1. **Research Discovery**
   - Free-text search across ArXiv quantitative finance papers
   - Advanced filtering by date, author, and category
   - Latest research prioritization

2. **AI-Powered Analysis**
   - Full-text PDF extraction and processing
   - Structured breakdown of papers (objective, methods, results, conclusions)
   - Trading application identification
   - Implementation complexity assessment
   - Data requirements analysis

3. **Interactive Exploration**
   - Natural language Q&A about papers
   - Follow-up question suggestions
   - Confidence scoring for answers

4. **Strategy Generation**
   - Customizable strategy parameters (market, timeframe, risk profile)
   - Complete Python code generation
   - Implementation guidance and limitations analysis
   - Code export and sharing

5. **User Experience**
   - Intuitive Gradio interface
   - Responsive design for all devices
   - Comprehensive error handling
   - Loading indicators for long-running operations

## 4. System Architecture

```
┌─────────────┐      ┌───────────────┐      ┌─────────────┐
│   Gradio    │◀────▶│    FastAPI    │◀────▶│  ArXiv API  │
│  Frontend   │      │    Backend    │      └─────────────┘
│ (UI + Chat) │      │  (Routers,    │
└─────────────┘      │   Services,   │      ┌─────────────┐
                     │   Models)     │◀────▶│ OpenAI API  │
                     └───────────────┘      └─────────────┘
                            ▲
                            │
                            ▼
                     ┌───────────────┐      ┌─────────────┐
                     │ PDF Processor │◀────▶│ArXiv PDFs   │
                     │ (Text Extract)│      └─────────────┘
                     └───────────────┘
                            ▲
                            │
                            ▼
                     ┌───────────────┐
                     │ Cache Service │
                     │(In-memory/Redis)│
                     └───────────────┘
```

## 5. System Components

### 5.1 Frontend (Gradio)

The frontend provides an intuitive interface for interacting with the ResearchTrader API, built using Gradio, a Python library for creating customizable web interfaces.

**Key Screens:**
1. **Search Interface**: Query input and filters for discovering papers
2. **Results Gallery**: Card-based display of matching papers
3. **Paper Analysis View**: Structured breakdown of paper contents
4. **Q&A Interface**: Natural language chat with papers
5. **Strategy Generator**: Custom parameters for generating trading code

### 5.2 Backend (FastAPI)

The backend is a FastAPI application that handles all API requests, processing, and external service communication.

#### Directory Structure

```
research_trader/
├── main.py                         # FastAPI application instance & startup
├── config.py                       # Pydantic settings configuration
├── models/                         # Pydantic data models
│   ├── __init__.py
│   ├── paper.py                    # Paper schema and validation
│   ├── summary.py                  # Summary and PaperText schemas
│   └── strategy.py                 # Strategy generation schemas
├── router/                         # API endpoints
│   ├── __init__.py
│   ├── search.py                   # Paper search functionality
│   ├── summarize.py                # Paper analysis & summarization
│   ├── qa.py                       # Question answering endpoints
│   └── strategy.py                 # Strategy generation endpoints
├── services/                       # Business logic layer
│   ├── __init__.py
│   ├── arxiv_client.py             # ArXiv API integration
│   ├── openai_client.py            # OpenAI API integration
│   ├── pdf_service.py              # PDF processing functionality
│   └── cache.py                    # Caching service
└── utils/                          # Utility functions
    ├── __init__.py
    └── errors.py                   # Custom error handling
```

#### Key Components

**FastAPI Application (`main.py`)**
- Configures the API server, middleware, and routers
- Sets up error handling and logging
- Initializes the application on startup

**Configuration Management (`config.py`)**
- Environment-based configuration using Pydantic
- Cached settings to improve performance
- Configuration for API endpoints, timeouts, and rate limits

**API Routers**
- **`search.py`**: Paper discovery endpoints
- **`summarize.py`**: Paper analysis and full-text processing
- **`qa.py`**: Question answering functionality
- **`strategy.py`**: Trading strategy generation

**Service Layer**
- **`arxiv_client.py`**: Handles communication with ArXiv API
- **`openai_client.py`**: LLM integration for analysis and generation
- **`pdf_service.py`**: PDF downloading and text extraction
- **`cache.py`**: In-memory caching for performance optimization

## 6. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search/` | GET | Search for papers matching a query |
| `/search/{paper_id}` | GET | Get a specific paper by ID |
| `/summarize/{paper_id}` | GET | Generate a comprehensive paper summary |
| `/summarize/text/{paper_id}` | GET | Retrieve full text of a paper |
| `/summarize/upload` | POST | Upload and analyze a PDF paper |
| `/qa/` | POST | Ask questions about papers |
| `/strategy/` | POST | Generate a trading strategy |
| `/strategy/stream` | POST | Generate a trading strategy (streaming) |

## 7. Data Flow

1. **Search Flow**:
   - User submits search query → `/search/` endpoint
   - ArXiv API is queried for matching papers
   - Results are cached and returned to the frontend

2. **Analysis Flow**:
   - User selects a paper → `/summarize/{paper_id}` endpoint
   - If not cached, downloads PDF from ArXiv
   - Extracts and processes full text
   - Sends to OpenAI for structured analysis
   - Results are cached and returned to frontend

3. **Q&A Flow**:
   - User asks question → `/qa/` endpoint
   - System retrieves paper summaries
   - Question and paper content sent to OpenAI
   - Answer, confidence score, and suggestions returned

4. **Strategy Flow**:
   - User configures parameters → `/strategy/` endpoint
   - System retrieves paper summaries
   - Parameters and paper analysis sent to OpenAI
   - Complete trading strategy returned

## 8. PDF Processing Pipeline

The PDF processing pipeline enables full-text analysis of research papers:

1. **Download**: Asynchronous PDF retrieval from ArXiv
2. **Extraction**: Text extraction using PDFMiner with PyPDF2 fallback
3. **Section Detection**: Heuristic-based identification of paper sections
4. **Processing**: LLM-based analysis of complete paper content
5. **Caching**: Storage of processed text to avoid redundant processing

## 9. Caching System

The caching system optimizes performance and reduces API calls:

- **In-memory Cache**: Fast, thread-safe storage for frequently accessed data
- **Tiered Caching**: Different TTLs based on data type and processing cost
- **Cache Categories**:
  - Paper metadata (search results)
  - Full paper text (extracted from PDFs)
  - Generated summaries and analyses
  - (Optional) Redis integration for distributed deployments

## 10. Tech Stack

- **Frontend**: Python, Gradio
- **Backend**: Python 3.10+, FastAPI, Uvicorn
- **Package Management**: uv with pyproject.toml
- **PDF Processing**: PDFMiner.six, PyPDF2
- **API Communication**: httpx, aiohttp
- **AI Integration**: OpenAI API (GPT-4 models)
- **Data Management**: In-memory cache (optional Redis)
- **Development Tools**: Black, isort, ruff, pytest

## 11. Deployment

The application can be deployed using various methods:

- **Local Development**: uvicorn with reload for development
- **Production Deployment**: 
  - Containerization with Docker
  - Orchestration with Kubernetes (optional)
  - CI/CD through GitHub Actions

## 12. Future Enhancements

- **Backtesting Integration**: Run strategies on historical data
- **User Accounts**: Save sessions, favorite papers, custom prompts
- **Multi-Model Support**: Allow switching between different LLM providers
- **Database Integration**: Persistent storage beyond caching
- **Analytics Dashboard**: Usage metrics and popular queries tracking
- **Citation Management**: Export citations in academic formats
- **Collaborative Features**: Sharing and commenting on papers/strategies

## 13. Project Dependencies

Key dependencies are specified in `pyproject.toml` and include:

- FastAPI and Uvicorn for the API server
- Pydantic for data validation and settings
- httpx and aiohttp for API communication
- PDFMiner.six and PyPDF2 for PDF processing
- Gradio for the user interface
- arxiv package for ArXiv API integration
- OpenAI SDK for LLM integration

## 14. Getting Started

See README.md for installation instructions and development setup.
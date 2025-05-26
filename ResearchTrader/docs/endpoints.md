# ArXiv Quantitative Finance API Documentation

## Endpoints Overview

### 1. Search Endpoint (`/search/`)
**Purpose**: Finds relevant research papers on ArXiv based on a search query.

**Implementation Details**:
- Uses the arxiv Python package to search the ArXiv API
- Filters for quantitative finance papers (q-fin category)
- Returns a list of papers with metadata: title, authors, publication date, abstract, etc.
- Each paper includes a PDF URL for later full-text extraction

**Caching Mechanism**:
- Search results are cached using the query string and max_results as the cache key
- Subsequent identical searches will return cached results without hitting the ArXiv API
- Default cache TTL is 1 hour (configurable in settings)

### 2. Paper Detail Endpoint (`/search/{paper_id}`)
**Purpose**: Retrieves detailed information about a specific paper.

**Implementation Details**:
- Fetches a single paper from ArXiv by ID
- Returns complete metadata including authors, categories, links, etc.
- Does not download the full text at this stage, only metadata

**Caching Mechanism**:
- Paper metadata is cached using the paper_id as the cache key
- Other endpoints can reuse this cached data without re-fetching from ArXiv

### 3. Summary Endpoint (`/summarize/{paper_id}`)
**Purpose**: Generates a comprehensive, trading-focused summary of a paper.

**Implementation Details**:
- Full PDF Processing: This is where the full PDF is downloaded and processed
- The endpoint follows this workflow:
  1. Checks if a cached summary exists
  2. If not, checks if the full text is already cached
  3. If not, downloads the PDF using the URL from paper metadata
  4. Extracts text using PDFMiner.six (with PyPDF2 as fallback)
  5. Stores the full text in the cache
  6. Sends the text to OpenAI for analysis and summary generation
  7. Caches and returns the structured summary

**Full Text Storage**:
- Full text is stored in a PaperText object that includes:
  - paper_id: The ArXiv ID
  - title: Paper title
  - abstract: Original abstract
  - full_text: Complete extracted text
  - sections: Attempted extraction of paper sections
  - extraction_date: Timestamp of when the text was extracted

- This object is cached with a longer TTL than other objects (2x default)

**Caching Mechanism**:
- Three distinct cache entries are created:
  1. Paper metadata (from search/detail endpoints)
  2. Full paper text (stored in PaperText object)
  3. Generated summary (final result with all analysis)

- Each has its own cache key format and potentially different TTLs

### 4. Paper Text Endpoint (`/summarize/text/{paper_id}`)
**Purpose**: Directly retrieves the full text of a paper.

**Implementation Details**:
- Returns the cached full text if available
- Otherwise, downloads and processes the PDF on demand
- Returns the complete PaperText object

**Caching Mechanism**:
- Uses the same cache as the summary endpoint for paper text
- Creates a new cache entry if not already present

### 5. PDF Upload Endpoint (`/summarize/upload`)
**Purpose**: Allows users to upload their own PDF files for analysis.

**Implementation Details**:
- Accepts a PDF file upload along with metadata (title, authors)
- Extracts text from the uploaded PDF
- Generates a unique ID for the uploaded paper
- Processes the text to create a trading-focused summary
- Stores both the text and summary in cache

**Storage Mechanism**:
- Uploaded files are temporarily saved to disk during processing
- The extracted text is stored in cache, not the original PDF
- A new paper ID is generated based on timestamp

### 6. Q&A Endpoint (`/qa/`)
**Purpose**: Answers questions about papers based on their content.

**Implementation Details**:
- Accepts a question and one or more paper IDs
- Retrieves the full summaries of those papers (including sections)
- Uses OpenAI to generate an answer based on the paper content
- Returns the answer, confidence score, and suggested follow-up questions

**Caching Mechanism**:
- Reuses cached summaries and paper text
- Q&A responses themselves are not cached (as they depend on unique questions)

### 7. Strategy Generation Endpoint (`/strategy/`)
**Purpose**: Creates Python trading strategies based on research papers.

**Implementation Details**:
- Takes one or more paper IDs plus trading parameters (market, timeframe, risk)
- Retrieves the full summaries of those papers
- Uses OpenAI to generate a complete trading strategy implementation
- Returns structured data including strategy name, description, code, usage notes, and limitations

**Caching Mechanism**:
- Reuses cached summaries and paper text
- Strategy responses are not cached (as they depend on specific parameters)

### 8. Streaming Strategy Endpoint (`/strategy/stream`)
**Purpose**: Same as strategy generation but with streaming response.

**Implementation Details**:
- Uses the same input parameters as the regular strategy endpoint
- Returns chunks of the response as they're generated
- Allows for progressive display in the frontend

**Caching Mechanism**:
- No caching for streaming responses

## How Full PDF Processing Works

### Download Phase:
- When a summary is requested, the system checks if we already have the full text
- If not, it downloads the PDF using aiohttp based on the PDF URL from paper metadata
- Downloads are done asynchronously to avoid blocking the API

### Text Extraction Phase:
- The downloaded PDF is saved to a temporary file
- PDFMiner.six is used first for high-quality text extraction
- If PDFMiner fails, PyPDF2 is used as a fallback
- All extraction is done in thread pools to avoid blocking the event loop

### Section Detection Phase:
- The system attempts to identify common paper sections (Introduction, Methods, etc.)
- Uses heuristics like capitalization, numbering, and common section names
- Creates a structured dictionary of sections when possible

### Storage Phase:
- The extracted text and identified sections are stored in a PaperText object
- This object is cached with a longer TTL than other objects

### Cleanup Phase:
- Temporary files are deleted after processing
- Only the extracted text is kept, not the original PDF

## Caching System Details

The caching system uses an in-memory store with the following characteristics:

### Key Generation:
- Cache keys are MD5 hashes of the input parameters
- Different prefixes are used for different types of data (search, paper, text, summary)

### TTL (Time-To-Live):
- Default TTL is 1 hour (3600 seconds)
- Paper text has a 2x longer TTL since it's expensive to reprocess
- TTL is configurable in settings

### Cache Types:
- search_cache: Search results by query + max_results
- paper_cache: Paper metadata by paper_id
- paper_text_cache: Full text by paper_id
- summary_cache: Generated summaries by paper_id

### Memory Management:
- Expired entries are cleared on access
- No automatic purging of old entries (would require a background task)

### Thread Safety:
- Uses asyncio locks to prevent concurrent modifications
- Safe for use in the async FastAPI environment

The caching system significantly improves performance, especially for repeated access to the same papers, and reduces API calls to both ArXiv and OpenAI.
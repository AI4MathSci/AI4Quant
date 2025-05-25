1. Landing / Search Screen
Goal: Let the user issue their first query.

UI Elements

App title / logo

Single text‑input (“Enter a trading research topic…”)

“Search” button

Short description / help tooltip (“Fetches the 10 latest arXiv Q‑Finance papers and lets you explore them.”)

User Action → enters query, clicks Search

System → validate non‑empty; show spinner; call arXiv API

2. Results Overview
Goal: Show the 10 fetched papers at a glance.

UI Elements

Paginated / scrollable list of Paper Cards (10 total)

Title (clickable)

Authors + date

One‑line summary snippet

“View Details” button

Left‑hand “Filters” panel (date range, author search)

User Actions

Scroll through cards

Filter / refine if desired

Click View Details on a card

3. Paper Detail / Structured Breakdown
Goal: Present each paper’s key sections in a uniform layout.

UI Elements

Tabbed view or Accordion showing:

Objective

Strategies / Methods

Results / Conclusions

Full Summary (paragraph)

“Ask the AI about this paper” chat icon

“Generate Trading Strategy” button

“Back to Results” link/button

User Actions

Read sections

Click Ask the AI → opens embedded QA chat (see next)

Click Generate Trading Strategy → goes to strategy module

4. Embedded QA Chat
Goal: Let the user query across the fetched papers (or this one).

UI Elements

Chat window (“Ask me anything about these 10 papers…”)

History pane of prior questions/answers

User Action → type question (e.g., “Which strategies rely on momentum?”) → press Enter

System → run RAG over the structured sections + summaries → return answer

5. Strategy Generation Module
Goal: Produce executable Python code for a trading strategy inspired by the literature.

UI Elements

Text area (“Describe the kind of strategy you want…”) pre‑filled with the current paper’s title or the user’s query

Dropdown: “Select data source” (e.g. YahooFinance, CSV upload)

“Generate Code” button

User Action → tweak prompt, choose data source, click Generate Code

System → call LLM with “Write me a Python strategy…” → stream back code

6. Code Display & Download
Goal: Let the user inspect, copy, or download the generated strategy.

UI Elements

Syntax‑highlighted code block (streaming)

“Copy to Clipboard” button

“Download .py” button

“Run Backtest” (optional advanced feature)

User Actions

Copy or download the file

(Optionally) run backtest

7. Error & Empty States
No Results Found → “No papers matched your query. Try broadening your terms.”

API Error → “Something went wrong fetching papers. Retry?”

8. Footer & Help
UI Elements

Link to “About / Methodology” (explains ArXiv scraping, structuring logic)

Link to “Feedback” (email / GitHub issues)

Version number

Putting it in your project doc
Introduction – What the app does

User‑Flow Diagram – Embed the above steps as a flowchart

Screen‑by‑Screen Mockups – Sketch each screen

Technical Notes – Arxiv API, LLM structuring, Gradio components
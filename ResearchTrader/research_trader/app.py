"""
Gradio frontend for ResearchTrader
"""

import os
from datetime import datetime
from typing import Any

import gradio as gr
import httpx
import pandas as pd

from research_trader.config import settings

API_BASE_URL = os.getenv("API_BASE_URL", f"http://{settings.API_HOST}:{settings.API_PORT}")
DEMO_TITLE = "ResearchTrader"
DEMO_DESCRIPTION = """
# ResearchTrader: Quantitative Finance Research Explorer

Discover, explore, and operationalize the latest ArXiv quantitative finance research.
Search for papers, explore structured summaries, ask questions based on selected papers, and generate Python trading strategy outlines.
"""
DEMO_ARTICLE = """
## How to Use
1. **Search**: Enter keywords to find relevant q-fin papers. Processing (download, summary) happens in the background.
2. **Explore**: Click on a paper row in the search results to view its details below.
3. **Select Context**: Use the "Available Papers for Context" section (refresh if needed) to choose papers for Q&A and Strategy Generation.
4. **Ask**: Use the Q&A tab with your selected context papers.
5. **Generate**: Use the Strategy tab with your selected context papers and prompt.
"""


# --- API Client Functions ---


# Function to list papers (for the CheckboxGroup)
async def list_papers_api() -> list[dict[str, Any]]:
    """List all cached papers from the backend API (GET /papers/)."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{API_BASE_URL}/papers/")
            response.raise_for_status()
            # Returns List[PaperSummaryResponse]
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"API Error listing papers ({e.response.status_code}): {e.response.text}")
            gr.Warning(f"API Error listing papers: {e.response.status_code}")
            return []
        except httpx.RequestError as e:
            print(f"Network Error listing papers: {e}")
            gr.Warning(f"Network Error connecting to API at {API_BASE_URL}.")
            return []
        except Exception as e:
            print(f"Unexpected error listing papers: {e}")
            gr.Warning("An unexpected error occurred listing papers.")
            return []


# --- (Keep existing API client functions: search_papers_api, get_paper_details_api, ask_question_api, generate_strategy_api_structured) ---
async def search_papers_api(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Search for papers via the backend API (POST /papers/)."""
    payload = {"query": query, "max_results": max_results}
    async with httpx.AsyncClient(
        timeout=60.0
    ) as client:  # Increased timeout for search+initial fetch
        try:
            response = await client.post(f"{API_BASE_URL}/papers/", json=payload)
            response.raise_for_status()  # Raise exception for 4xx/5xx errors
            # The response now contains PaperSummaryResponse objects
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"API Error searching papers ({e.response.status_code}): {e.response.text}")
            gr.Warning(
                f"API Error searching papers: {e.response.status_code} - {e.response.text[:100]}"
            )
            return []
        except httpx.RequestError as e:
            print(f"Network Error searching papers: {e}")
            gr.Error(f"Network Error connecting to API at {API_BASE_URL}. Is the backend running?")
            return []
        except Exception as e:
            print(f"Unexpected error searching papers: {e}")
            gr.Error("An unexpected error occurred during search.")
            return []


async def get_paper_details_api(paper_id: str) -> dict[str, Any] | None:
    """Get full paper details from the backend API (GET /papers/{paper_id})."""
    async with httpx.AsyncClient(
        timeout=180.0
    ) as client:  # Longer timeout for potential processing
        try:
            response = await client.get(f"{API_BASE_URL}/papers/{paper_id}")
            response.raise_for_status()
            # Response is the full Paper object
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                print(f"Paper {paper_id} not found in backend.")
                # This might happen if background processing hasn't finished, maybe retry?
                # For now, just return None
                return None
            else:
                print(
                    f"API Error getting paper details ({e.response.status_code}): {e.response.text}"
                )
                gr.Warning(f"API Error fetching paper details: {e.response.status_code}")
                return None
        except httpx.RequestError as e:
            print(f"Network Error getting paper details: {e}")
            gr.Error(f"Network Error connecting to API at {API_BASE_URL}.")
            return None
        except Exception as e:
            print(f"Unexpected error getting paper details: {e}")
            gr.Error("An unexpected error occurred fetching paper details.")
            return None


async def ask_question_api(question: str, paper_ids: list[str]) -> str:
    """Ask a question via the backend API (POST /qa/)."""
    payload = {"question": question, "paper_ids": paper_ids}
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(f"{API_BASE_URL}/qa/", json=payload)
            response.raise_for_status()
            result = response.json()  # Returns QAResponse model
            return result.get("answer", "No answer generated.")
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            try:  # Try parsing JSON error detail
                error_json = e.response.json()
                error_detail = error_json.get("detail", error_detail)
            except:
                pass
            print(f"API Error asking question ({e.response.status_code}): {error_detail}")
            gr.Warning(
                f"API Error asking question: {e.response.status_code} - {error_detail[:200]}"
            )
            return f"Error asking question: {error_detail}"
        except httpx.RequestError as e:
            print(f"Network Error asking question: {e}")
            gr.Error(f"Network Error connecting to API at {API_BASE_URL}.")
            return "Network error asking question."
        except Exception as e:
            print(f"Unexpected error asking question: {e}")
            gr.Error("An unexpected error occurred asking the question.")
            return "Unexpected error asking question."


# Keep generate_strategy_api_structured as it returns the needed dict
async def generate_strategy_api_structured(
    paper_ids: list[str], strategy_prompt: str
) -> dict[str, Any]:
    """Generate a strategy via the backend API (POST /strategy/), returning structured data."""
    payload = {"paper_ids": paper_ids, "strategy_prompt": strategy_prompt}
    async with httpx.AsyncClient(timeout=180.0) as client:  # Longer timeout for generation
        try:
            response = await client.post(f"{API_BASE_URL}/strategy/", json=payload)
            response.raise_for_status()
            result = response.json()  # Returns StrategyGenerationResponse model

            # Check for generation notes/errors from backend
            if result.get("notes") and "failed" in result.get("notes", "").lower():
                gr.Warning(f"Strategy generation issue: {result.get('notes')}")
                return {"error": result.get("notes", "Generation failed.")}

            # Extract the nested strategy object
            strategy_output = result.get("strategy")
            if strategy_output:
                return strategy_output  # Return the dict directly
            else:
                gr.Warning("Strategy generation returned empty output.")
                return {"error": "Generation returned empty output."}

        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            try:
                error_json = e.response.json()
                error_detail = error_json.get("detail", error_detail)
            except:
                pass
            print(f"API Error generating strategy ({e.response.status_code}): {error_detail}")
            gr.Warning(
                f"API Error generating strategy: {e.response.status_code} - {error_detail[:200]}"
            )
            return {"error": f"API Error: {error_detail}"}
        except httpx.RequestError as e:
            print(f"Network Error generating strategy: {e}")
            gr.Error(f"Network Error connecting to API at {API_BASE_URL}.")
            return {"error": "Network error generating strategy."}
        except Exception as e:
            print(f"Unexpected error generating strategy: {e}")
            gr.Error("An unexpected error occurred generating the strategy.")
            return {"error": "Unexpected error generating strategy."}


# --- Gradio UI Functions ---


def format_search_results(papers_summary_data: list[dict[str, Any]]) -> pd.DataFrame:
    """Format paper summaries for display in a dataframe."""
    formatted_papers = []
    for p_summary in papers_summary_data:
        # Format date
        pub_date_str = p_summary.get("published_date")
        formatted_date = "N/A"
        if pub_date_str:
            try:
                dt_obj = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                formatted_date = dt_obj.strftime("%Y-%m-%d")
            except ValueError:
                pass

        authors = p_summary.get("authors", [])
        if len(authors) > 3:
            author_str = ", ".join(authors[:3]) + f" +{len(authors) - 3}"
        else:
            author_str = ", ".join(authors)

        abstract = p_summary.get("abstract", "")
        if len(abstract) > 200:
            abstract = abstract[:197] + "..."

        formatted_papers.append(
            {
                "ID": p_summary.get("paper_id", "N/A"),
                "Title": p_summary.get("title", "N/A"),
                "Authors": author_str,
                "Date": formatted_date,
                "Tags": ", ".join(p_summary.get("tags", [])[:3]),
                "Abstract": abstract,
            }
        )
    if not formatted_papers:
        return pd.DataFrame(columns=["ID", "Title", "Authors", "Date", "Tags", "Abstract"])

    return pd.DataFrame(formatted_papers)


def format_papers_for_checkboxgroup(
    papers_summary_data: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Formats paper data for CheckboxGroup choices (label, value)."""
    choices = []
    for p in papers_summary_data:
        paper_id = p.get("paper_id", "N/A")
        title = p.get("title", "Unknown Title")
        # Truncate title if too long for label
        display_title = title[:70] + "..." if len(title) > 70 else title
        label = f"{display_title} ({paper_id})"
        choices.append((label, paper_id))
    # Sort by paper ID (often corresponds to date)
    choices.sort(key=lambda x: x[1], reverse=True)
    return choices


# --- Gradio UI Component Actions ---


# Keep handle_search largely the same, maybe update info message
async def handle_search(
    query: str, max_results: int
) -> tuple[pd.DataFrame, gr.Markdown, list[str]]:
    """Action for the search button click."""
    if not query.strip():
        empty_df = format_search_results([])
        # Return empty list for paper_ids state as well
        return empty_df, gr.update(visible=False, value=""), []

    gr.Info(f"Searching for '{query}'... Processing happens in background.")
    papers_summary_list = await search_papers_api(query, max_results)

    if not papers_summary_list:
        empty_df = format_search_results([])
        return empty_df, gr.update(visible=True, value="No papers found matching your query."), []

    df = format_search_results(papers_summary_list)
    # Extract paper IDs - This might be less crucial now if we rely on the CheckboxGroup
    paper_ids = [p.get("paper_id", "") for p in papers_summary_list if p.get("paper_id")]

    gr.Info(
        f"Found {len(paper_ids)} papers. Select a row to view details or refresh 'Available Papers' list."
    )
    # Return paper_ids, although it might not be directly used as state anymore
    return df, gr.update(visible=False, value=""), paper_ids


# Keep handle_select_paper the same for displaying details of a single clicked paper
async def handle_select_paper(
    evt: gr.SelectData, papers_df: pd.DataFrame
) -> tuple[str, str, str, str, str, str, str]:
    """Action when a paper row is selected in the dataframe to show details."""
    if not evt.index or not isinstance(papers_df, pd.DataFrame) or papers_df.empty:
        # Return blank fields and None for the selected paper ID state
        return "No paper selected", "", "", "", "", "", None

    selected_row_index = evt.index[0]
    if selected_row_index >= len(papers_df):
        return "Selection index out of bounds", "", "", "", "", "", None

    selected_paper_id = papers_df.iloc[selected_row_index]["ID"]
    selected_paper_title = papers_df.iloc[selected_row_index]["Title"]

    if not selected_paper_id or selected_paper_id == "N/A":
        return "Invalid paper ID selected", "", "", "", "", "", None

    gr.Info(f"Loading details for paper {selected_paper_id}...")
    paper_details = await get_paper_details_api(selected_paper_id)

    if not paper_details:
        gr.Warning(
            f"Could not load details for {selected_paper_id}. It might still be processing, or an error occurred."
        )
        abstract = papers_df.iloc[selected_row_index]["Abstract"]
        # Return details view with loading state, and the selected ID
        return (
            f"# {selected_paper_title}",
            abstract,
            "Loading...",
            "Loading...",
            "Loading...",
            "Loading...",
            selected_paper_id,
        )

    metadata = paper_details.get("metadata", {})
    content = paper_details.get("content", {})
    structured_summary = content.get("structured_summary", {}) if content else {}

    title = metadata.get("title", "Title not found")
    abstract = metadata.get("abstract", "Abstract not available")
    comprehensive_summary = (
        content.get("comprehensive_summary", "Summary not available")
        if content
        else "Summary not available"
    )

    # Format structured summary lists if they exist
    objective = structured_summary.get("objective", "N/A")
    methods = (
        "\n- ".join(structured_summary.get("methods", ["N/A"]))
        if isinstance(structured_summary.get("methods"), list)
        else structured_summary.get("methods", "N/A")
    )
    results = (
        "\n- ".join(structured_summary.get("results", ["N/A"]))
        if isinstance(structured_summary.get("results"), list)
        else structured_summary.get("results", "N/A")
    )
    conclusions = (
        "\n- ".join(structured_summary.get("conclusions", ["N/A"]))
        if isinstance(structured_summary.get("conclusions"), list)
        else structured_summary.get("conclusions", "N/A")
    )

    # Prepend bullet if lists were joined
    methods = "- " + methods if "\n-" in methods else methods
    results = "- " + results if "\n-" in results else results
    conclusions = "- " + conclusions if "\n-" in conclusions else conclusions

    display_summary = (
        comprehensive_summary if comprehensive_summary != "Summary not available" else abstract
    )

    gr.Info(f"Details loaded for {selected_paper_id}.")
    return (
        f"# {title}",
        display_summary,
        objective,
        methods,
        results,
        conclusions,
        selected_paper_id,  # Pass the ID to the state (still useful for maybe showing which paper details are displayed)
    )


# New handler for refreshing the CheckboxGroup
async def handle_refresh_available_papers():
    """Action for the Refresh Available Papers button."""
    gr.Info("Refreshing list of available papers from cache...")
    cached_papers = await list_papers_api()
    if not cached_papers:
        gr.Warning("No cached papers found or failed to fetch list.")
        return gr.update(choices=[], value=[])  # Clear choices and selection

    choices = format_papers_for_checkboxgroup(cached_papers)
    gr.Info(f"Found {len(choices)} available papers.")
    # Update choices, keep current selection if possible (Gradio handles this reasonably well)
    return gr.update(choices=choices)


# Updated handler for Q&A - takes list of IDs from CheckboxGroup
async def handle_ask_question(question: str, selected_paper_ids: list[str]) -> str:
    """Action for the Ask button in Q&A tab."""
    if not question.strip():
        return "Please enter a question."
    if not selected_paper_ids:  # Check if the list is empty
        return "Please select one or more papers from the 'Available Papers' list below the search results."

    gr.Info(f"Asking question about {len(selected_paper_ids)} selected papers...")
    answer = await ask_question_api(question, selected_paper_ids)
    return answer


# Updated handler for Strategy - takes list of IDs from CheckboxGroup
async def handle_generate_strategy(
    selected_paper_ids: list[str], strategy_prompt: str
) -> tuple[str, str, str, str]:
    """Action for the Generate button in Strategy tab."""
    if not strategy_prompt.strip():
        no_output = "Please enter a prompt describing the desired strategy."
        return no_output, "", "", ""
    if not selected_paper_ids:  # Check if the list is empty
        no_output = "Please select one or more papers from the 'Available Papers' list below the search results to base the strategy on."
        return no_output, "", "", ""

    gr.Info(f"Generating strategy based on {len(selected_paper_ids)} selected papers...")
    strategy_result = await generate_strategy_api_structured(selected_paper_ids, strategy_prompt)

    if not strategy_result or "error" in strategy_result:
        error_msg = strategy_result.get("error", "Unknown error during generation.")
        gr.Warning(f"Strategy generation failed: {error_msg}")
        # Return error message to the first output field
        return f"**Error:** {error_msg}", "", "", ""

    # Extract parts from the structured result
    desc = strategy_result.get("strategy_description", "N/A")
    pseudo = strategy_result.get("pseudocode", "N/A")
    usage = strategy_result.get("how_to_use", "N/A")
    code = strategy_result.get("python_code", "# No code generated")

    return desc, pseudo, usage, code


# --- Build Gradio Interface ---

css = """
.gradio-container { font-family: 'IBM Plex Sans', sans-serif; }
.gr-button { color: white; border-color: black; background: black; }
footer { display: none !important; }
.gr-checkboxgroup .gr-form { /* Try to make checkbox group scrollable */
    max-height: 250px;
    overflow-y: auto;
    border: 1px solid #e0e0e0;
    padding: 5px;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="slate", secondary_hue="blue"), css=css, title=DEMO_TITLE
) as demo:
    gr.Markdown(f"<h1 style='text-align: center; margin-bottom: 1rem'>{DEMO_TITLE}</h1>")
    gr.Markdown(DEMO_DESCRIPTION)

    # Keep state for the single paper ID whose details are currently displayed
    displayed_paper_id_state = gr.State(None)

    with gr.Tabs():
        with gr.TabItem("Search & Explore Papers"):
            # --- Search Input ---
            with gr.Row():
                with gr.Column(scale=4):
                    search_input = gr.Textbox(
                        label="Search Query",
                        placeholder='Enter keywords (e.g., "volatility forecasting using lstm")...',
                        lines=1,
                    )
                with gr.Column(scale=1):
                    max_results_slider = gr.Slider(
                        minimum=1,  # Allow searching for 1
                        maximum=settings.MAX_PAPERS_FETCH,
                        value=settings.DEFAULT_PAPERS_FETCH,
                        step=1,
                        label="Max Papers to Fetch",
                    )
                with gr.Column(scale=1):
                    search_button = gr.Button("Search Papers", variant="primary")

            search_info = gr.Markdown(
                "", visible=False
            )  # Placeholder for status messages like "No results"

            # --- Search Results Table ---
            papers_dataframe = gr.DataFrame(
                headers=["ID", "Title", "Authors", "Date", "Tags", "Abstract"],
                datatype=["str", "str", "str", "str", "str", "str"],
                interactive=True,  # Allow row selection for detail view
                wrap=True,
                label="Search Results (Click Row to Load Details Below)",
            )

            # --- Available Papers for Context Selection ---
            with gr.Group():  # Group these elements
                gr.Markdown("### Available Papers for Context")
                with gr.Row():
                    refresh_papers_btn = gr.Button("Refresh List from Cache")
                available_papers_cbg = gr.CheckboxGroup(
                    label="Select papers to use for Q&A and Strategy Generation",
                    choices=[],  # Initially empty, populated by refresh
                    scale=1,  # Relative scale
                )

            # --- Single Paper Detail View ---
            gr.Markdown("### Selected Paper Details (from table click)")
            with gr.Row():
                paper_title_md = gr.Markdown("# Select a paper from the table above")
            with gr.Row():
                paper_summary_md = gr.Markdown(label="Summary / Abstract")

            with gr.Accordion("Structured Analysis", open=False):
                with gr.Row():
                    with gr.Column(scale=1):
                        objective_md = gr.Markdown(label="Objective")
                    with gr.Column(scale=1):
                        methods_md = gr.Markdown(label="Methods")
                with gr.Row():
                    with gr.Column(scale=1):
                        results_md = gr.Markdown(label="Results")
                    with gr.Column(scale=1):
                        conclusions_md = gr.Markdown(label="Conclusions")

        with gr.TabItem("Q&A"):
            gr.Markdown(
                "Ask a question based on the papers selected in the **'Available Papers for Context'** list on the 'Search & Explore' tab."
            )
            with gr.Row():
                qa_question_input = gr.Textbox(
                    label="Your Question",
                    lines=2,
                    placeholder="e.g., Compare the methodologies used in the selected papers for volatility prediction.",
                    scale=4,
                )
                qa_button = gr.Button("Ask Question", variant="primary", scale=1)
            qa_answer_output = gr.Markdown(label="Answer")

        with gr.TabItem("Strategy Generation"):
            gr.Markdown(
                "Generate a Python strategy outline based on the papers selected in the **'Available Papers for Context'** list on the 'Search & Explore' tab."
            )
            strategy_prompt_input = gr.Textbox(
                label="Strategy Prompt",
                lines=4,
                placeholder="""Describe the strategy you want. Reference concepts from the selected papers. 
e.g., "Combine the LSTM approach from [ID1] with the risk management technique from [ID2] for a daily stock trading strategy.""",  # Use triple quotes
            )
            strategy_button = gr.Button("Generate Strategy Outline", variant="primary")
            with gr.Accordion("Generated Strategy Details", open=True):
                strategy_description_md = gr.Markdown(label="Description")
                strategy_pseudocode_md = gr.Markdown(label="Pseudocode / Logic")
                strategy_usage_md = gr.Markdown(label="How to Use / Limitations")
                strategy_python_code = gr.Code(
                    label="Python Code Outline", language="python", interactive=False
                )

    # --- Component Wiring ---

    # Search Action
    search_button.click(
        handle_search,
        inputs=[search_input, max_results_slider],
        # Output DF, info message, and the list of paper IDs (though state not directly used now)
        outputs=[papers_dataframe, search_info, gr.State([])],  # Pass dummy state for now
    )

    # Paper Selection Action (for Detail View)
    papers_dataframe.select(
        handle_select_paper,
        inputs=[papers_dataframe],
        outputs=[
            paper_title_md,
            paper_summary_md,
            objective_md,
            methods_md,
            results_md,
            conclusions_md,
            displayed_paper_id_state,  # Update state for the *displayed* paper
        ],
    )

    # Refresh Available Papers Action
    refresh_papers_btn.click(
        handle_refresh_available_papers, inputs=None, outputs=[available_papers_cbg]
    )

    # Q&A Action - Input is now the CheckboxGroup
    qa_button.click(
        handle_ask_question,
        inputs=[qa_question_input, available_papers_cbg],  # Use CheckboxGroup value
        outputs=[qa_answer_output],
    )

    # Strategy Generation Action - Input is now the CheckboxGroup
    strategy_button.click(
        handle_generate_strategy,
        inputs=[available_papers_cbg, strategy_prompt_input],  # Use CheckboxGroup value
        outputs=[
            strategy_description_md,
            strategy_pseudocode_md,
            strategy_usage_md,
            strategy_python_code,
        ],
    )

    # Load available papers when the app starts
    demo.load(handle_refresh_available_papers, None, [available_papers_cbg])

    gr.Markdown(DEMO_ARTICLE)


# --- Launch ---

if __name__ == "__main__":
    print(f"Connecting Gradio frontend to API at: {API_BASE_URL}")
    # Use settings for server name and port
    demo.launch(server_name=settings.GRADIO_SERVER_NAME, server_port=settings.GRADIO_SERVER_PORT, share=True)

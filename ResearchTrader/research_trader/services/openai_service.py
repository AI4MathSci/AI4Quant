"""
Service for interacting with OpenAI API for paper analysis and generation tasks.
"""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from research_trader.config import settings
from research_trader.models.paper import (  # Import StrategyOutput
    Paper,
    StrategyOutput,
)
from research_trader.utils.errors import ServiceError

logger = logging.getLogger(__name__)


class OpenAIService:
    """Service interacting with the OpenAI API."""

    def __init__(
        self,
        api_key: str = settings.OPENAI_API_KEY,
        model: str = settings.OPENAI_MODEL,
        base_url: str = settings.OPENAI_BASE_URL,
        timeout: float = settings.OPENAI_TIMEOUT,
    ):
        if not api_key:
            raise ValueError("OpenAI API key is not configured.")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    async def _make_request(
        self, endpoint: str, data: dict[str, Any], stream: bool = False
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """
        Make a request to the OpenAI API.

        Args:
            endpoint: API endpoint (e.g., 'chat/completions').
            data: Request payload.
            stream: Whether to stream the response.

        Returns:
            API response as a dictionary or an async generator for streaming.

        Raises:
            ServiceError: If the API request fails.
        """
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        url = f"{self.base_url}/{endpoint}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                if stream:

                    async def _stream_generator():
                        async with client.stream(
                            "POST", url, headers=headers, json=data
                        ) as response:
                            # Check for initial error before streaming
                            if response.status_code >= 400:
                                try:
                                    error_body = await response.aread()
                                    error_data = json.loads(error_body)
                                    error_message = error_data.get("error", {}).get(
                                        "message", "Unknown OpenAI streaming error"
                                    )
                                    logger.error(
                                        f"OpenAI stream error ({response.status_code}): {error_message}"
                                    )
                                    raise ServiceError(
                                        detail=f"OpenAI API error: {error_message}",
                                        service="openai",
                                        status_code=503,
                                    )
                                except (json.JSONDecodeError, Exception) as e:
                                    logger.error(
                                        f"OpenAI stream error ({response.status_code}): Failed to parse error body - {e}"
                                    )
                                    raise ServiceError(
                                        detail=f"OpenAI API error: {response.status_code} {response.reason_phrase}",
                                        service="openai",
                                        status_code=503,
                                    )

                            async for line in response.aiter_lines():
                                if line.startswith("data: "):
                                    line_data = line[len("data: ") :]
                                    if line_data.strip() == "[DONE]":
                                        break
                                    try:
                                        chunk = json.loads(line_data)
                                        yield chunk
                                    except json.JSONDecodeError:
                                        logger.warning(
                                            f"Failed to decode stream chunk: {line_data}"
                                        )
                                        continue

                    return _stream_generator()
                else:
                    response = await client.post(url, headers=headers, json=data)
                    response.raise_for_status()  # Raise HTTPStatusError for bad responses (4xx or 5xx)
                    return response.json()
        except httpx.HTTPStatusError as e:
            error_message = f"OpenAI API error: {e.response.status_code}"
            try:
                error_data = e.response.json()
                if "error" in error_data and "message" in error_data["error"]:
                    error_message = f"OpenAI API error: {error_data['error']['message']}"
            except (json.JSONDecodeError, AttributeError):
                error_message = (
                    f"OpenAI API error: {e.response.status_code} {e.response.reason_phrase}"
                )
            logger.error(error_message, exc_info=True)
            raise ServiceError(
                detail=error_message, service="openai", status_code=e.response.status_code
            )
        except httpx.RequestError as e:
            error_message = f"OpenAI request failed: {e.__class__.__name__}"
            logger.error(error_message, exc_info=True)
            raise ServiceError(detail=error_message, service="openai", status_code=503)
        except Exception as e:
            error_message = (
                f"An unexpected error occurred during OpenAI request: {e.__class__.__name__}"
            )
            logger.error(error_message, exc_info=True)
            raise ServiceError(detail=error_message, service="openai", status_code=500)

    async def _extract_json_response(self, response: dict[str, Any]) -> dict[str, Any] | None:
        """Safely extracts and parses JSON content from the OpenAI response."""
        try:
            content = response["choices"][0]["message"]["content"]
            # Sometimes the model might wrap the JSON in ```json ... ```
            if content.startswith("```json\n"):
                content = content[len("```json\n") : -len("\n```")]
            elif content.startswith("```"):
                content = content[len("```") : -len("```")]
            return json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError, TypeError) as e:
            logger.error(
                f"Failed to extract or parse JSON from OpenAI response: {e}. Response: {response}"
            )
            return None

    async def generate_structured_summary(
        self, paper_text: str, title: str, authors: list[str]
    ) -> dict[str, str | list[str]] | None:
        """
        Generates a structured summary (Objective, Methods, Results, Conclusions).

        Args:
            paper_text: The full text or abstract of the paper.
            title: Paper title.
            authors: List of author names.

        Returns:
            A dictionary containing structured summary fields (objective: str, methods: List[str], results: List[str], conclusions: List[str]) or None if generation fails.
        """
        system_prompt = """
You are an expert financial analyst specializing in extracting key structured information from quantitative finance research papers.
Focus ONLY on the following sections: Objective, Methods, Results, Conclusions.
Format the output as a JSON object with keys: "objective", "methods", "results", "conclusions".
- The value for "objective" should be a single, concise string stating the paper's main goal.
- The values for "methods", "results", and "conclusions" should be lists of strings, where each string is a specific point or finding (e.g., a specific algorithm, a key performance metric, a main takeaway).
Keep the points concise and directly extracted or inferred from the text.
Example for list values: "methods": ["Used LSTM model", "Applied Kalman filter", "Performed Monte Carlo simulation"]
"""
        user_prompt = f"""
Analyze the following paper extract:

Title: {title}
Authors: {", ".join(authors)}

Text:
{paper_text[: settings.MAX_TEXT_LENGTH_FOR_SUMMARY]}...

Please provide the structured summary as JSON, ensuring 'methods', 'results', and 'conclusions' are lists of strings.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self._make_request(
                "chat/completions",
                {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                },
            )
            raw_data = await self._extract_json_response(response)

            if not raw_data or not all(
                k in raw_data for k in ["objective", "methods", "results", "conclusions"]
            ):
                logger.warning(
                    f"Failed to get complete structured summary fields for '{title}'. Response: {raw_data}"
                )
                return None

            # Process and validate the response structure
            processed_data: dict[str, str | list[str]] = {}

            # Objective should be a string
            obj_val = raw_data.get("objective")
            processed_data["objective"] = str(obj_val) if obj_val is not None else "N/A"

            # Methods, Results, Conclusions should be lists of strings
            for key in ["methods", "results", "conclusions"]:
                val = raw_data.get(key)
                if isinstance(val, list):
                    # Ensure all items in list are strings
                    processed_data[key] = [str(item) for item in val]
                elif isinstance(val, str):
                    # Attempt to split string into list (e.g., if LLM used bullet points)
                    items = [
                        item.strip()
                        for item in val.split("\n")
                        if item.strip() and item.strip() not in ["-", "*", "•"]
                    ]
                    # Remove potential leading bullet point characters if splitting failed cleanly
                    items = [item.lstrip("-*• ") for item in items]
                    processed_data[key] = (
                        items if items else [val]
                    )  # Keep original string if split fails
                else:
                    # Handle unexpected type
                    processed_data[key] = [str(val)] if val is not None else []

            logger.info(f"Successfully processed structured summary for '{title}'")
            return processed_data

        except ServiceError as e:
            logger.error(f"Service error during structured summary generation for '{title}': {e}")
            return None

    async def generate_comprehensive_summary(
        self, paper_text: str, title: str, authors: list[str]
    ) -> str | None:
        """
        Generates a comprehensive, human-readable summary for a quant audience.

        Args:
            paper_text: The full text or abstract of the paper.
            title: Paper title.
            authors: List of author names.

        Returns:
            A string containing the summary or None if generation fails.
        """
        system_prompt = """
You are an expert financial analyst skilled at summarizing complex quantitative finance research for a quant trading audience.
Write a concise (150-250 words) summary covering the paper's core objective, methodology, key findings, and potential trading applications.
Focus on clarity and actionable insights. Avoid jargon where possible or explain it briefly.
Output ONLY the summary text, without any preamble or headings.
"""
        user_prompt = f"""
Summarize the following paper:

Title: {title}
Authors: {", ".join(authors)}

Text:
{paper_text[: settings.MAX_TEXT_LENGTH_FOR_SUMMARY]}...

Please provide the comprehensive summary.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self._make_request(
                "chat/completions",
                {"model": self.model, "messages": messages, "temperature": 0.3, "max_tokens": 400},
            )
            summary = response["choices"][0]["message"]["content"]
            return summary.strip() if summary else None
        except (ServiceError, KeyError, IndexError, TypeError) as e:
            logger.error(f"Failed to generate comprehensive summary for '{title}': {e}")
            return None

    def _format_paper_context(self, papers: list[Paper]) -> str:
        """Formats a list of papers into a string context, applying per-paper length limits for QA."""
        full_context_str = ""
        for i, paper in enumerate(papers):
            paper_context_parts = []
            # Basic metadata
            paper_context_parts.append(
                f"--- Paper {i + 1}: {paper.metadata.title} (ID: {paper.paper_id}) ---"
            )
            paper_context_parts.append(f"Authors: {', '.join(paper.metadata.authors)}")
            paper_context_parts.append(
                f"Published: {paper.metadata.published_date.strftime('%Y-%m-%d') if paper.metadata.published_date else 'N/A'}"
            )
            paper_context_parts.append(f"Abstract: {paper.metadata.abstract}")

            # Content (Summaries)
            if paper.content:
                if paper.content.structured_summary:
                    structured_summary = paper.content.structured_summary
                    paper_context_parts.append(
                        f"Objective: {structured_summary.get('objective', 'N/A')}"
                    )

                    # Format lists with bullet points
                    for key in ["methods", "results", "conclusions"]:
                        items = structured_summary.get(key, [])
                        if isinstance(items, list) and items:
                            formatted_items = "\n  - " + "\n  - ".join(items)
                            paper_context_parts.append(f"{key.capitalize()}: {formatted_items}")
                        elif isinstance(items, str):  # Fallback if somehow it's still a string
                            paper_context_parts.append(f"{key.capitalize()}: {items}")
                        # else: omit if empty list or None

                if paper.content.comprehensive_summary:
                    paper_context_parts.append(
                        f"Generated Summary: {paper.content.comprehensive_summary}"
                    )
                # Optionally include a snippet of full text if desired, but keep it short
                # if paper.content.full_text:
                #     paper_context_parts.append(f"Full Text Snippet: {paper.content.full_text[:200]}...")

            # Join parts for this paper and truncate to the per-paper limit
            single_paper_context = "\n".join(paper_context_parts)
            truncated_paper_context = single_paper_context[: settings.MAX_CONTEXT_PER_PAPER_FOR_QA]

            full_context_str += truncated_paper_context + "\n\n"  # Add separator

        return full_context_str.strip()

    async def answer_question(self, question: str, papers: list[Paper]) -> str | None:
        """
        Answers a question based on the content of provided papers.

        Args:
            question: The user's question.
            papers: A list of Paper objects containing metadata and potentially content.

        Returns:
            The answer string or None if generation fails.
        """
        if not papers:
            return "No papers provided to answer the question."

        # Formatting context now handles per-paper limits
        context = self._format_paper_context(papers)

        system_prompt = """
You are a financial research assistant. Answer the user's question based *only* on the provided context from research papers.
Synthesize information across papers if necessary. Cite the relevant paper IDs (e.g., [ID: 1234.5678]) where appropriate.
If the answer cannot be found in the context, state that explicitly.
Be concise and direct.
"""
        # No longer need to truncate the final context here
        user_prompt = f"""
Context:
{context}

Question: {question}

Answer:
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self._make_request(
                "chat/completions",
                {"model": self.model, "messages": messages, "temperature": 0.2, "max_tokens": 500},
            )
            answer = response["choices"][0]["message"]["content"]
            return answer.strip() if answer else None
        except (ServiceError, KeyError, IndexError, TypeError) as e:
            logger.error(f"Failed to answer question '{question[:50]}...': {e}")
            return None

    async def generate_strategy(
        self, papers: list[Paper], strategy_prompt: str
    ) -> StrategyOutput | None:
        """
        Generates a structured trading strategy outline based on provided papers and a prompt.

        Args:
            papers: List of Paper objects to base the strategy on.
            strategy_prompt: User prompt describing the desired strategy aspects.

        Returns:
            A StrategyOutput object containing the structured strategy details, or None.
        """
        if not papers:
            logger.warning("No papers provided to generate strategy.")
            # Return an empty object or None? Returning None seems consistent.
            return None

        context = self._format_paper_context(papers)

        system_prompt = """
You are an AI assistant specialized in generating structured trading strategy outlines based on financial research papers.
Generate a conceptual strategy based on the user's request and the provided paper context.

Provide the following four sections in a JSON object:
1.  `strategy_description`: A clear explanation of the strategy, its core logic, the financial concepts involved, and how it relates to the provided paper context (cite paper IDs like [ID: 1234.5678]).
2.  `pseudocode`: High-level, step-by-step logic of the strategy in pseudocode format or clear bullet points.
3.  `python_code`: A conceptual Python code outline for the strategy using common libraries (e.g., pandas, numpy, placeholder functions for indicators/execution). Focus on core logic (signal generation, entry/exit). Include comments explaining the code and linking back to papers [ID: ...]. Keep it illustrative, not production-ready.
4.  `how_to_use`: Practical notes including potential data requirements, key parameters to tune, assumptions made, and important limitations or risks.

Output ONLY the JSON object with these four keys.
Example JSON structure: 
{ "strategy_description": "...". "pseudocode": "1. Check condition X...", "python_code": "import pandas as pd\n...", "how_to_use": "Requires daily price data..." }
"""
        user_prompt = f"""
Paper Context:
{context[: settings.MAX_CONTEXT_LENGTH_FOR_STRATEGY]}

Strategy Request: {strategy_prompt}

Generate the structured strategy outline as a JSON object with keys: strategy_description, pseudocode, python_code, how_to_use.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self._make_request(
                "chat/completions",
                {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.4,
                    "max_tokens": 2000,  # Increased tokens for more detailed output
                    "response_format": {"type": "json_object"},  # Request JSON output
                },
            )

            # Parse the JSON response
            strategy_data = await self._extract_json_response(response)

            if not strategy_data or not all(
                k in strategy_data
                for k in ["strategy_description", "pseudocode", "python_code", "how_to_use"]
            ):
                logger.error(
                    f"Failed to get complete structured strategy output for prompt '{strategy_prompt[:50]}...'. Response: {strategy_data}"
                )
                return None

            # Validate and create StrategyOutput object
            # Basic validation - ensure keys exist and are roughly strings (more robust validation possible)
            return StrategyOutput(
                strategy_description=str(strategy_data.get("strategy_description", "")),
                pseudocode=str(strategy_data.get("pseudocode", "")),
                python_code=str(strategy_data.get("python_code", "")),
                how_to_use=str(strategy_data.get("how_to_use", "")),
            )

        except (ServiceError, KeyError, IndexError, TypeError) as e:
            logger.error(
                f"Failed to generate structured strategy for prompt '{strategy_prompt[:50]}...': {e}"
            )
            return None

"""
Research Agent — Deal Intelligence Pipeline

Responsibility: Search for recent company intelligence using the Anthropic web search tool.
This agent has a single job: populate a structured research output for a given company.

It is one of two agents that run in parallel at the start of the pipeline.
The orchestrator dispatches it alongside the CRM Agent and waits for both before proceeding.

Building blocks demonstrated:
- Focus: single responsibility, scoped toolset, explicit persona
- Tools: web search tool with directive description
- Guardrails: structured JSON output contract, confidence signaling, no fabrication rule
- Memory: original goal injected into every call
"""

import json
import anthropic


SYSTEM_PROMPT = """You are a senior sales research analyst preparing intelligence for an enterprise sales rep before a high-value account call.

Your job is to research the company provided and return a structured JSON output. Nothing else. No preamble, no explanation, no markdown formatting around the JSON.

Use the web search tool to find current, reliable information. Search multiple times if needed to populate all required fields. Never answer from memory or training data. Always call the search tool before producing output.

Research the following and populate each field:
- company_summary: 2-3 sentence description of what the company does and their current market position
- recent_news: list of up to 3 significant developments from the last 90 days (funding, product launches, leadership changes, partnerships, layoffs)
- funding_status: most recent funding round, amount, date, and lead investor if available
- leadership_changes: any executive hires, departures, or role changes in the last 6 months
- market_signals: specific signals relevant to the call objective provided (expansion plans, technology investments, competitive moves, pain points)
- growth_indicators: any signals suggesting the company is growing, contracting, or pivoting
- confidence: overall confidence in the research output — high (strong reliable sources found), medium (partial information found), or low (limited reliable information available)
- low_confidence_fields: list of any specific fields where data was missing, unreliable, or not found

Rules:
- If you cannot find reliable information for a field, set that field to null
- Never fabricate or infer information not found through search
- Set confidence to low if more than two fields are null
- recent_news should be a list of objects with fields: headline, date, significance
- market_signals should be specific to the call objective, not generic company information
"""


async def run_research_agent(
    company_name: str,
    call_objective: str,
    run_id: str
) -> dict:
    """
    Run the Research Agent for a given company and call objective.

    This agent uses the Anthropic web search tool to find recent company intelligence.
    It runs in parallel with the CRM Agent — neither waits for the other.

    Args:
        company_name: The name of the company to research
        call_objective: The rep's stated objective for the call (shapes market_signals)
        run_id: Unique identifier for this pipeline run, used for tracing

    Returns:
        dict with research output fields and metadata, or error structure on failure
    """
    client = anthropic.AsyncAnthropic()

    # The user message passes company name, call objective, and original goal.
    # Original goal is injected explicitly — not inferred — so the agent
    # calibrates market_signals to what the rep actually needs.
    user_message = f"""Research this company and return the structured JSON output as specified.

Company: {company_name}
Call objective: {call_objective}
Original goal: Prepare the sales rep to have a high-quality, relevant conversation with this account that advances the relationship toward a close.

Focus market_signals specifically on what is relevant to the call objective above.
Run ID: {run_id}"""

    try:
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            tools=[
                {
                    "type": "web_search_20250305",
                    "name": "web_search"
                }
            ],
            messages=[
                {"role": "user", "content": user_message}
            ]
        )

        # Extract the final text output from the response.
        # The response may contain tool use blocks before the final text block.
        # We find the last text block which contains the structured JSON output.
        output_text = None
        for block in response.content:
            if block.type == "text":
                output_text = block.text

        if not output_text:
            return _error_output(company_name, run_id, "Research Agent returned no text output")

        # Parse and validate the JSON output.
        # If parsing fails the agent produced malformed output — treat as a low confidence result.
        try:
            # Strip any accidental markdown formatting before parsing
            cleaned = output_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            research_output = json.loads(cleaned.strip())
        except json.JSONDecodeError as e:
            return _error_output(company_name, run_id, f"Research Agent returned malformed JSON: {str(e)}")

        # Attach pipeline metadata to the output for orchestrator tracing
        research_output["agent"] = "research"
        research_output["run_id"] = run_id
        research_output["company_name"] = company_name
        research_output["status"] = "completed"

        # Ensure required fields exist even if agent omitted them
        required_fields = [
            "company_summary", "recent_news", "funding_status",
            "leadership_changes", "market_signals", "growth_indicators",
            "confidence", "low_confidence_fields"
        ]
        for field in required_fields:
            if field not in research_output:
                research_output[field] = None
                if research_output.get("low_confidence_fields") is None:
                    research_output["low_confidence_fields"] = []
                if isinstance(research_output.get("low_confidence_fields"), list):
                    research_output["low_confidence_fields"].append(field)

        # Downgrade confidence if too many fields are null
        null_fields = [f for f in required_fields[:-2] if research_output.get(f) is None]
        if len(null_fields) > 2 and research_output.get("confidence") != "low":
            research_output["confidence"] = "medium" if len(null_fields) <= 4 else "low"

        return research_output

    except anthropic.APIError as e:
        return _error_output(company_name, run_id, f"Anthropic API error: {str(e)}")
    except Exception as e:
        return _error_output(company_name, run_id, f"Unexpected error: {str(e)}")


def _error_output(company_name: str, run_id: str, error_message: str) -> dict:
    """
    Return a structured error output when the Research Agent fails.

    This ensures the Validation Agent always receives a well-formed object
    even on failure, so validation can route correctly rather than crashing.
    """
    return {
        "agent": "research",
        "run_id": run_id,
        "company_name": company_name,
        "status": "failed",
        "error": error_message,
        "company_summary": None,
        "recent_news": None,
        "funding_status": None,
        "leadership_changes": None,
        "market_signals": None,
        "growth_indicators": None,
        "confidence": "low",
        "low_confidence_fields": ["all fields — agent failed"]
    }

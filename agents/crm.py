"""
CRM Agent — Deal Intelligence Pipeline

Responsibility: Retrieve and summarize interaction history for a given company
from the CRM data file. This agent has a single job: populate a structured CRM
output based on what is actually in the data — nothing more.

It runs in parallel with the Research Agent. Neither waits for the other.
The orchestrator dispatches both simultaneously and waits for both to complete
before passing outputs to the Validation Agent.

Building blocks demonstrated:
- Focus: single responsibility, scoped toolset, no reasoning beyond CRM data
- Tools: read_crm function tool with directive description, episodic memory check
- Guardrails: structured JSON output contract, no fabrication rule, unknown state handling
- Memory: episodic memory check for prior briefs, original goal injected into every call
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import anthropic


# Paths resolved relative to this file so the project runs from any directory
DATA_PATH = Path(__file__).parent.parent / "data" / "crm_data.json"
MEMORY_PATH = Path(__file__).parent.parent / "memory" / "runs_log.json"


SYSTEM_PROMPT = """You are a CRM analyst preparing account context for an enterprise sales rep before a high-value call.

Your job is to retrieve and summarize interaction history for the company provided using the read_crm tool. Return a structured JSON output only. No preamble, no explanation, no markdown formatting around the JSON.

Always call the read_crm tool before producing any output. Never answer from memory.

Retrieve and populate each field:
- last_contact_date: date of most recent interaction (YYYY-MM-DD format)
- last_contact_summary: 2-3 sentence summary of what was discussed and what the outcome was
- open_opportunities: list of active deals with name, stage, value, and close date
- relationship_status: active, dormant, at-risk, or unknown — based on recency and interaction notes
- key_contacts: list of contacts with name, role, and a one-line behavioral note for the rep
- internal_flags: list of flags the rep must know before the call
- prior_brief_exists: true or false — set by the system before you run, do not change this value
- days_since_last_contact: integer number of days since last_contact_date, calculated from today
- account_health: strong, developing, at-risk, or unknown — taken directly from CRM data

Rules:
- Only use information from the CRM data. Never infer or fabricate.
- If no record exists for the company, return null for all fields and set relationship_status to unknown.
- key_contacts should include behavioral notes that help the rep calibrate their approach — not just names and roles.
- internal_flags are critical — include all of them exactly as they appear in the CRM data.
- days_since_last_contact should be an integer. If last_contact_date is null, set to null.
"""


def _load_crm_data() -> dict:
    """
    Load the CRM data file from disk.
    Returns empty accounts list if file does not exist.
    """
    if not DATA_PATH.exists():
        return {"accounts": []}
    with open(DATA_PATH, "r") as f:
        return json.load(f)


def _check_episodic_memory(company_name: str) -> bool:
    """
    Check the episodic memory log for prior briefs on this company
    within the last 30 days.

    This is the memory layer in practice: before the CRM Agent runs,
    the system checks whether this account has been through the pipeline
    recently so the Synthesis Agent can reference it in the brief.

    Returns True if a prior brief exists within 30 days, False otherwise.
    """
    if not MEMORY_PATH.exists():
        return False

    try:
        with open(MEMORY_PATH, "r") as f:
            runs = json.load(f)

        cutoff = datetime.now() - timedelta(days=30)

        for run in runs:
            if run.get("company_name", "").lower() == company_name.lower():
                run_time = datetime.fromisoformat(run.get("timestamp", "2000-01-01"))
                if run_time > cutoff and run.get("pipeline_status") == "completed":
                    return True

        return False

    except (json.JSONDecodeError, ValueError):
        return False


def _get_crm_record(company_name: str) -> dict | None:
    """
    Find the CRM record for a given company name.
    Case-insensitive match on company_name field.
    Returns None if no record found.
    """
    crm_data = _load_crm_data()
    for account in crm_data.get("accounts", []):
        if account.get("company_name", "").lower() == company_name.lower():
            return account
    return None


async def run_crm_agent(
    company_name: str,
    run_id: str
) -> dict:
    """
    Run the CRM Agent for a given company.

    This agent retrieves interaction history from the CRM data file using
    a function tool. It checks episodic memory for prior briefs before running
    and includes that flag in its output for the Synthesis Agent.

    Runs in parallel with the Research Agent — neither waits for the other.

    Args:
        company_name: The name of the company to retrieve CRM data for
        run_id: Unique identifier for this pipeline run, used for tracing

    Returns:
        dict with CRM output fields and metadata, or error structure on failure
    """
    client = anthropic.AsyncAnthropic()

    # Check episodic memory before running the agent.
    # This flag travels through the pipeline and surfaces in the final brief
    # so the rep knows if they have prepared for this account recently.
    prior_brief_exists = _check_episodic_memory(company_name)

    # Define the read_crm tool as a function tool.
    # The agent calls this tool with the company name and receives the raw CRM record.
    # Directive description ensures the agent always calls it before producing output.
    tools = [
        {
            "name": "read_crm",
            "description": "Use this tool to retrieve interaction history and account data for the company from the CRM data file. Always call this tool before producing any output. Never answer from memory.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "The exact company name to look up in the CRM"
                    }
                },
                "required": ["company_name"]
            }
        }
    ]

    user_message = f"""Retrieve and summarize the CRM data for this company and return the structured JSON output as specified.

Company: {company_name}
Prior brief exists (last 30 days): {prior_brief_exists}
Today's date: {datetime.now().strftime("%Y-%m-%d")}
Original goal: Prepare the sales rep with full relationship context so they can have a high-quality, relevant conversation that advances the relationship.

Set prior_brief_exists to {str(prior_brief_exists).lower()} in your output — this value is provided by the system.
Run ID: {run_id}"""

    messages = [{"role": "user", "content": user_message}]

    try:
        # Agentic loop: keep running until the agent produces a final text output.
        # The agent will call read_crm, receive the CRM data, then produce its JSON output.
        while True:
            response = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                tools=tools,
                messages=messages
            )

            # If the agent produced a final text response, extract and return it
            if response.stop_reason == "end_turn":
                output_text = None
                for block in response.content:
                    if hasattr(block, "text"):
                        output_text = block.text

                if not output_text:
                    return _error_output(company_name, run_id, prior_brief_exists,
                                        "CRM Agent returned no text output")

                # Parse the JSON output
                try:
                    cleaned = output_text.strip()
                    if cleaned.startswith("```"):
                        cleaned = cleaned.split("```")[1]
                        if cleaned.startswith("json"):
                            cleaned = cleaned[4:]
                    crm_output = json.loads(cleaned.strip())
                except json.JSONDecodeError as e:
                    return _error_output(company_name, run_id, prior_brief_exists,
                                        f"CRM Agent returned malformed JSON: {str(e)}")

                # Attach pipeline metadata
                crm_output["agent"] = "crm"
                crm_output["run_id"] = run_id
                crm_output["company_name"] = company_name
                crm_output["status"] = "completed"
                crm_output["prior_brief_exists"] = prior_brief_exists

                # Ensure required fields exist
                required_fields = [
                    "last_contact_date", "last_contact_summary", "open_opportunities",
                    "relationship_status", "key_contacts", "internal_flags",
                    "days_since_last_contact", "account_health"
                ]
                for field in required_fields:
                    if field not in crm_output:
                        crm_output[field] = None

                return crm_output

            # If the agent made a tool call, process it and continue the loop
            if response.stop_reason == "tool_use":
                # Add the assistant's tool use response to the message history
                messages.append({"role": "assistant", "content": response.content})

                # Process each tool call in the response
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use" and block.name == "read_crm":
                        # Execute the read_crm tool with the company name the agent provided
                        requested_company = block.input.get("company_name", company_name)
                        crm_record = _get_crm_record(requested_company)

                        if crm_record:
                            tool_result = json.dumps(crm_record, indent=2)
                        else:
                            tool_result = json.dumps({
                                "found": False,
                                "message": f"No CRM record found for {requested_company}",
                                "company_name": requested_company
                            })

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": tool_result
                        })

                # Add tool results to the message history and continue the loop
                messages.append({"role": "user", "content": tool_results})

            else:
                # Unexpected stop reason — return error
                return _error_output(company_name, run_id, prior_brief_exists,
                                    f"Unexpected stop reason: {response.stop_reason}")

    except anthropic.APIError as e:
        return _error_output(company_name, run_id, prior_brief_exists,
                            f"Anthropic API error: {str(e)}")
    except Exception as e:
        return _error_output(company_name, run_id, prior_brief_exists,
                            f"Unexpected error: {str(e)}")


def _error_output(
    company_name: str,
    run_id: str,
    prior_brief_exists: bool,
    error_message: str
) -> dict:
    """
    Return a structured error output when the CRM Agent fails.

    Ensures the Validation Agent always receives a well-formed object
    even on failure so the pipeline routes correctly rather than crashing.
    """
    return {
        "agent": "crm",
        "run_id": run_id,
        "company_name": company_name,
        "status": "failed",
        "error": error_message,
        "last_contact_date": None,
        "last_contact_summary": None,
        "open_opportunities": None,
        "relationship_status": "unknown",
        "key_contacts": None,
        "internal_flags": None,
        "prior_brief_exists": prior_brief_exists,
        "days_since_last_contact": None,
        "account_health": "unknown"
    }

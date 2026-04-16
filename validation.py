"""
Validation Agent — Deal Intelligence Pipeline

Responsibility: Quality check both upstream outputs before synthesis runs.
This agent is the guardrail between the parallel execution stage and the
Synthesis Agent. It ensures the Synthesis Agent never receives incomplete
or failed inputs.

This agent has no tools. It reasons only over what it receives.
It is the judge node in the pipeline — a dedicated validation step
that catches errors at the handoff rather than letting them compound
into the final brief.

Building blocks demonstrated:
- Focus: single responsibility, no tools, pure validation logic
- Guardrails: between-agent quality check, explicit minimum standards,
  routing signal for the orchestrator
- Cooperation: structured output contract that the orchestrator routes on
- Orchestration: validation result determines whether pipeline continues
  to synthesis or routes to error handling
"""

import json
import anthropic


SYSTEM_PROMPT = """You are a quality validation agent in a multi-agent sales intelligence pipeline.

You will receive two JSON outputs from upstream agents: a Research Agent output and a CRM Agent output.
Your job is to check whether both meet the minimum quality standard required for synthesis.

Return a structured JSON object only. No preamble, no explanation, no markdown formatting around the JSON.

Minimum standard for Research Agent output:
- status must be "completed" (not "failed")
- company_summary must be populated (not null)
- At least one of recent_news or market_signals must be populated (not null)
- confidence must not be null

Minimum standard for CRM Agent output:
- status must be "completed" (not "failed")
- relationship_status must be populated (not null or missing)

Validation output structure:
{
  "validation": "pass" or "fail",
  "research_passed": true or false,
  "crm_passed": true or false,
  "failed_agent": "research", "crm", "both", or null,
  "reason": "specific reason for failure" or null if passed,
  "warnings": ["list of non-blocking quality concerns the Synthesis Agent should be aware of"],
  "quality_notes": {
    "research_confidence": "high", "medium", or "low",
    "crm_completeness": "complete", "partial", or "empty",
    "prior_brief_exists": true or false
  }
}

Warnings are non-blocking — the pipeline continues but the Synthesis Agent should
be aware of them. Examples of warnings:
- Research confidence is low but minimum fields are populated
- CRM record exists but last contact was more than 90 days ago
- No open opportunities found in CRM
- Several research fields are null but minimums are met

Rules:
- A failed agent status always results in validation fail regardless of field population
- Both agents must pass for overall validation to be "pass"
- Be specific in the reason field — the rep and the system need to know exactly what failed
- quality_notes must always be populated regardless of pass or fail
"""


async def run_validation_agent(
    research_output: dict,
    crm_output: dict,
    run_id: str
) -> dict:
    """
    Run the Validation Agent against both upstream outputs.

    This is the guardrail between parallel execution and synthesis.
    The orchestrator routes on the validation result:
    - pass: proceed to Synthesis Agent
    - fail: return error response, do not proceed to synthesis

    Args:
        research_output: Structured output from the Research Agent
        crm_output: Structured output from the CRM Agent
        run_id: Unique identifier for this pipeline run

    Returns:
        dict with validation result that the orchestrator routes on
    """
    client = anthropic.AsyncAnthropic()

    # Pass both outputs to the validation agent as a structured user message.
    # The agent has no tools — it reasons only over what it receives here.
    user_message = f"""Validate these two agent outputs against the minimum quality standards.

Research Agent Output:
{json.dumps(research_output, indent=2)}

CRM Agent Output:
{json.dumps(crm_output, indent=2)}

Run ID: {run_id}

Return the validation JSON object as specified."""

    try:
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )

        # Extract text output
        output_text = None
        for block in response.content:
            if hasattr(block, "text"):
                output_text = block.text

        if not output_text:
            # If the validation agent itself fails, fail safe — do not proceed to synthesis
            return _fail_safe_output(run_id, "Validation Agent returned no output")

        # Parse the JSON output
        try:
            cleaned = output_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            validation_output = json.loads(cleaned.strip())
        except json.JSONDecodeError as e:
            return _fail_safe_output(run_id, f"Validation Agent returned malformed JSON: {str(e)}")

        # Attach pipeline metadata
        validation_output["agent"] = "validation"
        validation_output["run_id"] = run_id
        validation_output["status"] = "completed"

        # Ensure required fields exist
        if "validation" not in validation_output:
            return _fail_safe_output(run_id, "Validation Agent output missing validation field")

        if "warnings" not in validation_output:
            validation_output["warnings"] = []

        if "quality_notes" not in validation_output:
            validation_output["quality_notes"] = {
                "research_confidence": research_output.get("confidence", "unknown"),
                "crm_completeness": "unknown",
                "prior_brief_exists": crm_output.get("prior_brief_exists", False)
            }

        return validation_output

    except anthropic.APIError as e:
        return _fail_safe_output(run_id, f"Anthropic API error: {str(e)}")
    except Exception as e:
        return _fail_safe_output(run_id, f"Unexpected error: {str(e)}")


def _fail_safe_output(run_id: str, error_message: str) -> dict:
    """
    Return a fail-safe validation output when the Validation Agent itself fails.

    When the validator cannot run, we fail the pipeline rather than proceeding
    to synthesis with unvalidated inputs. Failing safe is the correct behavior
    here — an unvalidated brief reaching a sales rep is worse than no brief.

    This is the system-level guardrail: even if the validation agent fails,
    the pipeline does not proceed to synthesis.
    """
    return {
        "agent": "validation",
        "run_id": run_id,
        "status": "failed",
        "validation": "fail",
        "research_passed": False,
        "crm_passed": False,
        "failed_agent": "both",
        "reason": f"Validation Agent failed to run: {error_message}",
        "warnings": [],
        "quality_notes": {
            "research_confidence": "unknown",
            "crm_completeness": "unknown",
            "prior_brief_exists": False
        }
    }

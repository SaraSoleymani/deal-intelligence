"""
Synthesis Agent — Deal Intelligence Pipeline

Responsibility: Resolve conflicts between upstream outputs and assemble
the final deal intelligence brief. This is the last reasoning step in
the pipeline before the output reaches the sales rep.

This agent has no tools. It receives validated, structured outputs from
both upstream agents and produces a human-readable brief with four
sections plus a confidence note.

Building blocks demonstrated:
- Focus: single responsibility, explicit conflict resolution rules, clear persona
- Guardrails: explicit conflict resolution logic, never resolve conflicts silently,
  confidence note surfacing gaps for the rep
- Cooperation: receives structured outputs from both upstream agents,
  applies source hierarchy rules to resolve contradictions
- Memory: original goal injected explicitly, prior brief flag surfaced in output
- HITL: confidence note and gap flags give the rep clear signals on where
  to verify before the call
"""

import json
import anthropic


SYSTEM_PROMPT = """You are a senior GTM strategist preparing a pre-call intelligence brief for an enterprise sales rep.

You will receive validated outputs from a Research Agent and a CRM Agent. Your job is to synthesize them into a clear, actionable brief that helps the rep have the best possible conversation with this account.

Return a structured JSON object only. No preamble, no explanation, no markdown formatting around the JSON.

Conflict resolution rules — apply these explicitly, never resolve conflicts silently:
1. For account history, relationship status, and prior interactions: trust CRM data over research data
2. For recent company news, market signals, funding, and product updates: trust research data over CRM data
3. For genuine conflicts where neither source is clearly authoritative: include both perspectives in the brief and flag the conflict explicitly so the rep can verify before the call

Output structure:
{
  "account_snapshot": "2-3 sentences on what the company does and their current situation based on both sources",
  "relationship_status": "Current relationship status, last contact summary, key contacts with behavioral notes, and any open opportunities",
  "talking_points": [
    {
      "point": "specific talking point",
      "source": "research or crm or both",
      "relevance": "why this matters for the call objective"
    }
  ],
  "recommended_approach": "1-2 paragraphs on how to open the call, what to emphasize, what to avoid, and what outcome to aim for",
  "confidence_note": {
    "overall_confidence": "high, medium, or low",
    "gaps": ["list of specific information gaps the rep should be aware of"],
    "conflicts_flagged": ["list of any conflicts between research and CRM data that were surfaced rather than resolved"],
    "prior_brief_note": "note if a brief was generated for this account in the last 30 days",
    "warnings": ["list of warnings passed from the validation agent"]
  },
  "internal_flags": ["critical flags from CRM the rep must know before the call"]
}

Rules:
- talking_points must be specific and actionable, not generic sales advice
- talking_points must be calibrated to the call objective provided
- recommended_approach must reflect the relationship status — approach an active deal differently from a dormant account
- confidence_note.gaps must list every null or low-confidence field from both upstream outputs
- internal_flags must include all flags from the CRM output exactly as provided — never omit them
- prior_brief_note must acknowledge if a brief was generated recently and suggest the rep review it
- If research confidence was low, surface that explicitly in confidence_note
- Never fabricate information not present in the upstream outputs
- The original goal must shape every section of the brief
"""


async def run_synthesis_agent(
    research_output: dict,
    crm_output: dict,
    validation_output: dict,
    company_name: str,
    call_objective: str,
    run_id: str
) -> dict:
    """
    Run the Synthesis Agent to produce the final deal intelligence brief.

    This agent only runs after the Validation Agent has returned a pass result.
    It receives all upstream outputs plus the validation quality notes,
    applies explicit conflict resolution rules, and produces a structured brief.

    Args:
        research_output: Validated output from the Research Agent
        crm_output: Validated output from the CRM Agent
        validation_output: Output from the Validation Agent including warnings and quality notes
        company_name: The company being researched
        call_objective: The rep's stated objective for the call
        run_id: Unique identifier for this pipeline run

    Returns:
        dict with the structured brief and pipeline metadata
    """
    client = anthropic.AsyncAnthropic()

    # Extract quality notes and warnings from validation output.
    # These travel into the synthesis context so the agent can surface them
    # in the confidence note without needing to re-derive them.
    quality_notes = validation_output.get("quality_notes", {})
    warnings = validation_output.get("warnings", [])
    prior_brief_exists = quality_notes.get("prior_brief_exists", False)

    # The original goal is injected explicitly as a standalone line.
    # This is explicit goal anchoring — it ensures the synthesis output
    # stays calibrated to what the rep actually needs regardless of how
    # much context is in the rest of the message.
    user_message = f"""Synthesize these validated agent outputs into a deal intelligence brief.

Company: {company_name}
Call objective: {call_objective}
Original goal: Prepare the sales rep to have a high-quality, relevant conversation with this account that advances the relationship toward a close.
Prior brief exists (last 30 days): {prior_brief_exists}
Run ID: {run_id}

Research Agent Output:
{json.dumps(research_output, indent=2)}

CRM Agent Output:
{json.dumps(crm_output, indent=2)}

Validation Warnings (non-blocking quality concerns to surface in the brief):
{json.dumps(warnings, indent=2)}

Apply the conflict resolution rules from your instructions explicitly.
Flag any conflicts you encounter rather than resolving them silently.
Return the structured JSON brief as specified."""

    try:
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=3000,
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
            return _error_output(company_name, run_id, "Synthesis Agent returned no text output")

        # Parse the JSON output
        try:
            cleaned = output_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            brief = json.loads(cleaned.strip())
        except json.JSONDecodeError as e:
            return _error_output(company_name, run_id,
                                f"Synthesis Agent returned malformed JSON: {str(e)}")

        # Attach pipeline metadata
        brief["agent"] = "synthesis"
        brief["run_id"] = run_id
        brief["company_name"] = company_name
        brief["call_objective"] = call_objective
        brief["status"] = "completed"

        # Ensure required sections exist
        required_sections = [
            "account_snapshot", "relationship_status", "talking_points",
            "recommended_approach", "confidence_note", "internal_flags"
        ]
        for section in required_sections:
            if section not in brief:
                brief[section] = None

        # Ensure confidence_note structure is complete
        if brief.get("confidence_note") and isinstance(brief["confidence_note"], dict):
            confidence_note = brief["confidence_note"]
            if "gaps" not in confidence_note:
                confidence_note["gaps"] = []
            if "conflicts_flagged" not in confidence_note:
                confidence_note["conflicts_flagged"] = []
            if "warnings" not in confidence_note:
                confidence_note["warnings"] = warnings
            if "prior_brief_note" not in confidence_note and prior_brief_exists:
                confidence_note["prior_brief_note"] = (
                    "A brief was generated for this account within the last 30 days. "
                    "Review prior brief before the call for continuity."
                )

        return brief

    except anthropic.APIError as e:
        return _error_output(company_name, run_id, f"Anthropic API error: {str(e)}")
    except Exception as e:
        return _error_output(company_name, run_id, f"Unexpected error: {str(e)}")


def _error_output(company_name: str, run_id: str, error_message: str) -> dict:
    """
    Return a structured error output when the Synthesis Agent fails.

    At this stage the pipeline has already passed validation, so a synthesis
    failure is surfaced to the rep with a clear error message rather than
    silently returning an empty brief.
    """
    return {
        "agent": "synthesis",
        "run_id": run_id,
        "company_name": company_name,
        "status": "failed",
        "error": error_message,
        "account_snapshot": None,
        "relationship_status": None,
        "talking_points": [],
        "recommended_approach": None,
        "confidence_note": {
            "overall_confidence": "low",
            "gaps": ["Synthesis Agent failed — brief could not be generated"],
            "conflicts_flagged": [],
            "prior_brief_note": None,
            "warnings": []
        },
        "internal_flags": []
    }

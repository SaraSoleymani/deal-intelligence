# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Activate the virtual environment (required before running anything)
source venv/bin/activate

# Start the server (with hot reload)
python main.py

# Install dependencies
pip install -r requirements.txt
```

The server runs at `http://localhost:8000`. The web UI is served at `/`. API docs at `/docs`.

The `ANTHROPIC_API_KEY` environment variable must be set — the pipeline calls the Anthropic API on every request.

## Architecture

This is a four-agent deal intelligence pipeline built on FastAPI. `main.py` is the orchestrator; the four agents live in `agents/`. The orchestrator never generates content — it only coordinates agents, manages shared state, and routes on results.

**Pipeline sequence:**
1. `POST /api/pipeline` receives `{company_name, call_objective}` and generates a `run_id`
2. Research Agent and CRM Agent run **in parallel** via `asyncio.gather`
3. Validation Agent checks both outputs against minimum quality standards
4. Orchestrator routes: `validation == "pass"` → Synthesis Agent; `"fail"` → return error immediately
5. Synthesis Agent assembles the final brief
6. Full pipeline state is written to `memory/runs_log.json` (episodic memory)

### Agents

| Agent | Tool | Responsibility |
|---|---|---|
| **Research** (`agents/research.py`) | `web_search_20250305` (Anthropic built-in) | Searches for recent company intelligence; returns structured JSON with confidence rating |
| **CRM** (`agents/crm.py`) | `read_crm` (function tool, reads `data/crm_data.json`) | Retrieves interaction history; runs an agentic loop until `stop_reason == "end_turn"` |
| **Validation** (`agents/validation.py`) | None | Pure reasoning — checks both upstream outputs meet minimum standards; **fail-safe: if the validator itself errors, it returns `validation: "fail"` and blocks synthesis** |
| **Synthesis** (`agents/synthesis.py`) | None | Assembles the final brief from validated outputs; applies explicit conflict resolution rules (CRM wins on history; Research wins on news/market signals) |

All agents use `claude-haiku-4-5-20251001` and return structured JSON. Every agent output includes `agent`, `run_id`, `company_name`, and `status` fields injected by the caller after parsing.

### Shared State

The orchestrator builds a `state` dict progressively during a run. Agents receive only the slice they need — not the full object. The state shape:

```python
{
    "run_id": str,               # 8-char uppercase UUID prefix
    "company_name": str,
    "call_objective": str,
    "original_goal": str,        # injected into every agent call
    "research_output": dict,     # set after parallel stage
    "crm_output": dict,          # set after parallel stage
    "validation_result": dict,   # set after validation
    "final_brief": dict,         # set after synthesis
    "timestamp": str,
    "pipeline_status": "running" | "completed" | "failed"
}
```

### Memory

`memory/runs_log.json` is an append-only JSON array. Every pipeline run (completed or failed) is written here. The CRM Agent reads it at startup via `_check_episodic_memory()` to set `prior_brief_exists` — a flag that travels through to the final brief so the rep knows if they recently prepped for this account.

The `/api/memory` endpoint exposes the log, filterable by `company_name`.

### CRM Data

`data/crm_data.json` holds the mock CRM — a list of accounts with interaction history, key contacts (with behavioral notes), open opportunities, and internal flags. Case-insensitive name matching. If no record is found, the CRM Agent returns `relationship_status: "unknown"` with all fields null; this is a valid non-failing state.

## Guardrails

**Never bypass the validation gate.** The orchestrator checks `validation_result.get("validation") != "pass"` and returns immediately on failure. Synthesis must never run on unvalidated inputs.

**Validation fails safe.** If the Validation Agent itself errors (API failure, malformed output), `_fail_safe_output()` returns `validation: "fail"`, blocking the pipeline. This is intentional — an unvalidated brief reaching a rep is worse than no brief.

**Every agent returns a well-formed error structure on failure** (see `_error_output()` in each agent file). This ensures the Validation Agent always receives a typed object rather than crashing on `None`.

**Agents must not fabricate.** System prompts for Research and CRM agents explicitly forbid answering from training data. Research must call `web_search` before producing output; CRM must call `read_crm` before producing output. These are enforced via directive tool descriptions.

**Synthesis conflict resolution is explicit, never silent.** When Research and CRM data conflict, the agent surfaces both perspectives and flags the conflict in `confidence_note.conflicts_flagged`. The source hierarchy: CRM wins on history/relationships; Research wins on news/signals.

## What Not To Do

- **Do not add inter-agent communication.** Agents do not call each other. All coordination flows through the orchestrator.
- **Do not give agents tools outside their scope.** Research has web search only. CRM has `read_crm` only. Validation and Synthesis have no tools.
- **Do not let the orchestrator reason or generate content.** It dispatches, routes, and manages state. Any reasoning belongs in an agent.
- **Do not skip or conditionally run the Validation Agent.** It runs on every request, always, before synthesis.
- **Do not swallow memory write failures silently for anything other than the episodic log.** The `_write_episodic_memory` exception handler is intentionally silent because a memory failure must never crash a completed pipeline. Other errors should surface.
- **Do not change `prior_brief_exists` inside the CRM Agent.** This value is set by `_check_episodic_memory()` before the agent call and injected explicitly into the user message. The agent is instructed not to override it.

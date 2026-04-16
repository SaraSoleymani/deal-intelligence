"""
Orchestrator — Deal Intelligence Pipeline

This is the central orchestrator and FastAPI application. It manages the full
pipeline from input to final brief. It does not reason or generate content.
Its only job is to coordinate the four agents, manage shared state, handle
routing decisions, and write episodic memory.

Pipeline sequence:
1. Receive input from webhook, generate run_id
2. Dispatch Research Agent and CRM Agent in parallel
3. Pass both outputs to Validation Agent
4. Route on validation result:
   - pass: proceed to Synthesis Agent
   - fail: return error response, do not proceed
5. Pass validated outputs to Synthesis Agent
6. Write full pipeline trace to episodic memory log
7. Return final brief to web UI

Building blocks demonstrated:
- Orchestration: centralized orchestrator, parallel execution, state management,
  conditional routing, HITL checkpoint in the UI
- Memory: shared state object built progressively, episodic memory logging
- Cooperation: original goal injected into every agent call, structured state
  passed selectively to each agent
- Guardrails: validation gate enforced by orchestrator, never bypassed
"""

from __future__ import annotations

import asyncio
from dotenv import load_dotenv
load_dotenv()
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from agents.research import run_research_agent
from agents.crm import run_crm_agent
from agents.validation import run_validation_agent
from agents.synthesis import run_synthesis_agent


# Paths
MEMORY_PATH = Path(__file__).parent / "memory" / "runs_log.json"
STATIC_PATH = Path(__file__).parent / "static"

app = FastAPI(
    title="Deal Intelligence Pipeline",
    description="Multi-agent deal intelligence for enterprise sales reps",
    version="1.0.0"
)

# Serve static files for the web UI
app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")


class PipelineRequest(BaseModel):
    """Input model for the pipeline endpoint."""
    company_name: str
    call_objective: str


class PipelineResponse(BaseModel):
    """Response model for the pipeline endpoint."""
    run_id: str
    status: str
    company_name: str
    call_objective: str
    brief: Optional[dict] = None
    error: Optional[str] = None
    pipeline_trace: Optional[dict] = None


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the web UI."""
    index_path = STATIC_PATH / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    with open(index_path, "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/pipeline", response_model=PipelineResponse)
async def run_pipeline(request: PipelineRequest):
    """
    Run the full deal intelligence pipeline.

    This is the main orchestration endpoint. It manages the complete
    pipeline from input to final brief, handling parallel execution,
    validation routing, and episodic memory logging.
    """
    # Generate a unique run ID for tracing this pipeline execution.
    # This ID travels through every agent call and ties together
    # all logs, tool calls, and state changes for this run.
    run_id = str(uuid.uuid4())[:8].upper()

    # Initialize the shared state object.
    # This is built progressively as the pipeline runs.
    # Each agent receives only the slice it needs — not the full object.
    state = {
        "run_id": run_id,
        "company_name": request.company_name,
        "call_objective": request.call_objective,
        "original_goal": "Prepare the sales rep to have a high-quality, relevant conversation with this account that advances the relationship toward a close.",
        "research_output": None,
        "crm_output": None,
        "validation_result": None,
        "final_brief": None,
        "timestamp": datetime.now().isoformat(),
        "pipeline_status": "running"
    }

    try:
        # Stage 2: Parallel execution.
        # Research Agent and CRM Agent run simultaneously.
        # Neither waits for the other. asyncio.gather dispatches both
        # and returns when both complete.
        research_output, crm_output = await asyncio.gather(
            run_research_agent(
                company_name=request.company_name,
                call_objective=request.call_objective,
                run_id=run_id
            ),
            run_crm_agent(
                company_name=request.company_name,
                run_id=run_id
            )
        )

        # Update shared state with parallel outputs
        state["research_output"] = research_output
        state["crm_output"] = crm_output

        # Stage 3: Validation.
        # Both outputs are checked against minimum quality standards
        # before synthesis runs. This step is never skipped.
        validation_result = await run_validation_agent(
            research_output=research_output,
            crm_output=crm_output,
            run_id=run_id
        )

        state["validation_result"] = validation_result

        # Stage 4: Routing on validation result.
        # If validation fails, return error response immediately.
        # The pipeline never proceeds to synthesis with unvalidated inputs.
        if validation_result.get("validation") != "pass":
            state["pipeline_status"] = "failed"
            _write_episodic_memory(state)

            return PipelineResponse(
                run_id=run_id,
                status="failed",
                company_name=request.company_name,
                call_objective=request.call_objective,
                error=f"Validation failed — {validation_result.get('failed_agent', 'unknown')} agent: {validation_result.get('reason', 'unknown reason')}",
                pipeline_trace=_build_pipeline_trace(state)
            )

        # Stage 5: Synthesis.
        # Only runs after validation passes.
        # Receives validated outputs from both upstream agents plus
        # quality notes and warnings from the validation step.
        final_brief = await run_synthesis_agent(
            research_output=research_output,
            crm_output=crm_output,
            validation_output=validation_result,
            company_name=request.company_name,
            call_objective=request.call_objective,
            run_id=run_id
        )

        state["final_brief"] = final_brief

        # Check if synthesis itself failed
        if final_brief.get("status") == "failed":
            state["pipeline_status"] = "failed"
            _write_episodic_memory(state)

            return PipelineResponse(
                run_id=run_id,
                status="failed",
                company_name=request.company_name,
                call_objective=request.call_objective,
                error=f"Synthesis failed: {final_brief.get('error', 'unknown error')}",
                pipeline_trace=_build_pipeline_trace(state)
            )

        # Stage 6: Episodic memory logging.
        # Write the full pipeline trace to the runs log.
        # This is the memory layer — every completed run is stored
        # so the CRM Agent can check for prior briefs on future runs.
        state["pipeline_status"] = "completed"
        _write_episodic_memory(state)

        # Stage 7: Return final brief to web UI.
        # The UI surfaces confidence flags and warnings for the rep
        # to review before the call — this is the HITL checkpoint.
        return PipelineResponse(
            run_id=run_id,
            status="completed",
            company_name=request.company_name,
            call_objective=request.call_objective,
            brief=final_brief,
            pipeline_trace=_build_pipeline_trace(state)
        )

    except Exception as e:
        state["pipeline_status"] = "failed"
        _write_episodic_memory(state)

        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}"
        )


@app.get("/api/memory")
async def get_memory(company_name: Optional[str] = None):
    """
    Retrieve episodic memory log.

    Returns all pipeline runs, optionally filtered by company name.
    Useful for the rep to review prior briefs before a call.
    """
    if not MEMORY_PATH.exists():
        return JSONResponse(content={"runs": []})

    try:
        with open(MEMORY_PATH, "r") as f:
            runs = json.load(f)

        if company_name:
            runs = [
                r for r in runs
                if r.get("company_name", "").lower() == company_name.lower()
            ]

        # Return runs in reverse chronological order
        runs_sorted = sorted(runs, key=lambda x: x.get("timestamp", ""), reverse=True)

        return JSONResponse(content={"runs": runs_sorted, "total": len(runs_sorted)})

    except (json.JSONDecodeError, IOError) as e:
        raise HTTPException(status_code=500, detail=f"Memory read error: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


def _write_episodic_memory(state: dict):
    """
    Append the current pipeline state to the episodic memory log.

    This is the memory layer. Every pipeline run — completed or failed —
    is written to the log so the CRM Agent can check for prior briefs
    and the team has a full audit trail of every brief generated.

    Uses append-only writes to avoid overwriting prior runs.
    """
    try:
        # Ensure memory directory exists
        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Load existing runs
        existing_runs = []
        if MEMORY_PATH.exists():
            try:
                with open(MEMORY_PATH, "r") as f:
                    existing_runs = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing_runs = []

        # Create a memory entry with the full state.
        # We store the complete state so the pipeline has full traceability.
        # For production systems with high volume, consider storing only
        # the brief and metadata rather than full agent outputs.
        memory_entry = {
            "run_id": state.get("run_id"),
            "company_name": state.get("company_name"),
            "call_objective": state.get("call_objective"),
            "timestamp": state.get("timestamp"),
            "pipeline_status": state.get("pipeline_status"),
            "research_confidence": (
                state.get("research_output", {}) or {}
            ).get("confidence"),
            "crm_relationship_status": (
                state.get("crm_output", {}) or {}
            ).get("relationship_status"),
            "validation_result": (
                state.get("validation_result", {}) or {}
            ).get("validation"),
            "brief_generated": state.get("final_brief") is not None and
                               state.get("pipeline_status") == "completed",
            "full_state": state
        }

        existing_runs.append(memory_entry)

        with open(MEMORY_PATH, "w") as f:
            json.dump(existing_runs, f, indent=2, default=str)

    except Exception:
        # Memory write failure should never crash the pipeline.
        # Log silently and continue — the brief has already been generated.
        pass


def _build_pipeline_trace(state: dict) -> dict:
    """
    Build a pipeline trace summary for the API response.

    This gives the UI and the caller visibility into what happened
    at each stage of the pipeline without exposing the full state object.
    """
    return {
        "run_id": state.get("run_id"),
        "timestamp": state.get("timestamp"),
        "pipeline_status": state.get("pipeline_status"),
        "stages": {
            "research": {
                "status": (state.get("research_output") or {}).get("status"),
                "confidence": (state.get("research_output") or {}).get("confidence"),
                "low_confidence_fields": (
                    state.get("research_output") or {}
                ).get("low_confidence_fields", [])
            },
            "crm": {
                "status": (state.get("crm_output") or {}).get("status"),
                "relationship_status": (
                    state.get("crm_output") or {}
                ).get("relationship_status"),
                "prior_brief_exists": (
                    state.get("crm_output") or {}
                ).get("prior_brief_exists", False)
            },
            "validation": {
                "status": (state.get("validation_result") or {}).get("validation"),
                "warnings": (state.get("validation_result") or {}).get("warnings", []),
                "failed_agent": (
                    state.get("validation_result") or {}
                ).get("failed_agent")
            },
            "synthesis": {
                "status": (state.get("final_brief") or {}).get("status"),
                "overall_confidence": (
                    (state.get("final_brief") or {})
                    .get("confidence_note") or {}
                ).get("overall_confidence")
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

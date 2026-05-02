# Deal Intelligence Pipeline

A multi-agent deal intelligence system that prepares enterprise sales reps for high-value account calls. Built as a practical companion to **Article 7: Multi-Agent Systems** in the *Building with Agentic AI* series.

A rep submits a company name and call objective. Four specialized agents run in a coordinated pipeline and return a structured brief ready for human review before the call.

---

## What This Demonstrates

Every architectural decision in this project maps directly to a building block from the article:

| Building Block | How It Shows Up |
|---|---|
| **Orchestration** | `main.py` manages pipeline sequence, parallel dispatch, validation routing, and state logging |
| **Focus** | Each agent has one responsibility and receives only the context it needs |
| **Tools** | Research Agent uses Anthropic web search. CRM Agent uses a `read_crm` function tool |
| **Guardrails** | Validation Agent checks both outputs before synthesis. Failed validation never proceeds |
| **Cooperation** | Structured JSON contracts between all agents. Conflict resolution rules in Synthesis Agent |
| **Memory** | Episodic memory log persists every run. CRM Agent checks for prior briefs before running |
| **HITL** | Confidence note and gap flags surface in the UI for rep review before the call |

---

## Architecture

```
Webhook Input
     │
     ├──────────────────────┐
     ▼                      ▼
Research Agent         CRM Agent
(web search)       (read_crm tool)
     │                      │
     └──────────┬───────────┘
                ▼
        Validation Agent
        (quality check)
                │
         pass ──┤── fail → error response
                ▼
        Synthesis Agent
    (conflict resolution +
       brief assembly)
                │
                ▼
         Final Brief
     (web UI + memory log)
```

### The Four Agents

**Research Agent** (`agents/research.py`)
Uses the Anthropic web search tool to find recent company intelligence: news, funding, leadership changes, and market signals calibrated to the call objective.

**CRM Agent** (`agents/crm.py`)
Retrieves interaction history from the CRM data file. Checks episodic memory for prior briefs on the same account within the last 30 days.

**Validation Agent** (`agents/validation.py`)
Quality checks both upstream outputs before synthesis runs. Returns pass or fail with specific reasons. Failed validation routes to error response — never to synthesis.

**Synthesis Agent** (`agents/synthesis.py`)
Applies explicit conflict resolution rules and assembles the final brief. CRM data is authoritative for relationship context. Research data is authoritative for market signals. Genuine conflicts are flagged rather than resolved silently.

---

## Project Structure

```
deal-intelligence/
├── main.py                  # Orchestrator and FastAPI app
├── agents/
│   ├── research.py          # Research Agent
│   ├── crm.py               # CRM Agent
│   ├── validation.py        # Validation Agent
│   └── synthesis.py         # Synthesis Agent
├── data/
│   └── crm_data.json        # Sample CRM data (5 accounts)
├── memory/
│   └── runs_log.json        # Episodic memory log (auto-created)
├── static/
│   └── index.html           # Web UI
├── requirements.txt
├── CLAUDE.md                # Architecture guide for Claude Code
└── README.md
```

---

## Setup

### Requirements

- Python 3.10 or higher
- An Anthropic API key with access to Claude and the web search tool

### Installation

```bash
# Clone the repo
git clone https://github.com/your-username/deal-intelligence
cd deal-intelligence

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```bash
ANTHROPIC_API_KEY=your_api_key_here
```

Or export it directly:

```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

### Run

```bash
python main.py
```

Then open `http://localhost:8000` in your browser.

---

## Sample CRM Data

The project includes five sample accounts in `data/crm_data.json` covering the full range of relationship states:

| Company | Relationship Status | Deal Stage | Use For |
|---|---|---|---|
| Stripe | Active | Negotiation | Testing active deal with strong champion |
| Databricks | Active | Proposal | Testing multi-stakeholder deal |
| Figma | Dormant | Nurture | Testing re-engagement scenario |
| Notion | Active | Discovery | Testing early-stage cautious buyer |

To add your own accounts, follow the schema in `crm_data.json`.

---

## Extending This Project

**Swap the CRM data source**
Replace `_get_crm_record()` in `agents/crm.py` with a call to your actual CRM API (Salesforce, HubSpot, etc.). The agent interface stays the same.

**Add more agents**
Follow the pattern in any existing agent file: single responsibility, structured JSON output, confidence field, error output function. Register the new agent in `main.py`.

**Add a real notification layer**
Replace the UI confidence note with a Slack or email notification for low-confidence briefs that need human review before the call.

**Connect to a vector store**
Replace the JSON CRM file with a Pinecone or similar vector store for semantic retrieval over large interaction history. See Article 5 in the series for the RAG implementation patterns.

---

## The Series

This project is part of *Building with Agentic AI*, a 10-article series on building production-grade agentic systems for GTM and sales teams.

- Article 1: How to Pick the Right Problems for AI Agents and Automation
- Article 2: Building AI Agents in Practice: A Sales Outreach Agent with n8n and Claude
- Article 3: Bad Prompt, Good Prompt, Great Prompt: The Practical Guide to Prompt Engineering [+ Sales Agent Example]
- Article 4: The AI Meeting Prep Assistant: From Problem to a Full Product with n8n and v0
- Article 5: RAG for Revenue Teams: From Simple Retrieval to Agentic and Graph RAG
- Article 6: Evals for Agentic AI: How to Know If Your System Actually Works + Hands on n8n JSON Files
- **Article 7: Multi-Agent Systems (this project)**
- Article 8: Fine-Tuning and Intentional Knowledge Ingestion (coming next)

---

## License

MIT

# Tabi

A tool-calling travel companion agent, built as a practice project to close specific
gaps left by the Atelier project: dynamic tool-calling (ReAct-style), MCP, streaming,
durable DB-backed memory, and proactive (agent-initiated) behavior via Telegram.

## Blocks

- A. Core tool-calling agent (src/agent/)
- B. Toolbelt: weather, places, currency, documents, web search (src/tools/)
- C. Memory / DB layer: itinerary + preferences + durable checkpointer (src/memory/)
- D. "Design my day" planning capability (src/planner/)
- E. Telegram interface (src/interface/)
- F. Scheduler / proactive push (src/scheduler/)
- G. LangSmith observability (cross-cutting, via env vars)

## Setup

    python -m venv agent_env
    .\agent_env\Scripts\activate      # Windows
    pip install -r requirements.txt

Copy `.env.example` values into `.env` and fill in real keys.

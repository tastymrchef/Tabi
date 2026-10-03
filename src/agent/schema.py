"""
In Atelier, schema.py held Pydantic models that shaped an LLM's structured
output (e.g. RouterDecision) and TypedDicts described graph state.

create_agent doesn't need a Pydantic output model for Block A, because it
doesn't ask the LLM for one structured object the way our old nodes did —
it lets the model freely decide, turn by turn, whether to call a tool or
reply in plain text. There's no fixed shape to validate against yet.
(Pydantic isn't gone from the system, though — every LangChain message
object is itself a Pydantic model, and once Block B gives tools real
arguments, LangChain auto-builds a Pydantic model from each tool's function
signature to validate the LLM's tool-call arguments before running it.)

What still belongs here is graph state. create_agent ships its own default
AgentState (messages, plus a couple of internal fields it needs for its
middleware system), and lets you extend it with your own fields by
subclassing rather than building a fresh TypedDict from scratch. We add
user_id now so later blocks (C: memory, D: planner) can look up saved
preferences without changing workflow.py again.
"""

from langchain.agents import AgentState as _BaseAgentState


class AgentState(_BaseAgentState):
    # Placeholder for Block C — not used yet, wired in once memory exists.
    user_id: str

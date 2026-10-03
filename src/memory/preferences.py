"""
Block C -- long-term memory tools.

Unlike the checkpointer (which remembers one conversation thread), these
tools read/write the Store, which persists across every thread for a
given user -- a preference saved in one chat is still known in a brand
new one later. Namespaced per user_id so this doesn't need per-user
databases or anything heavier than one shared table.
"""

from langchain.tools import tool, ToolRuntime


@tool
async def remember_preference(key: str, value: str, runtime: ToolRuntime) -> str:
    """Save a fact or preference about the user that should be remembered
    in future conversations, not just this one -- e.g. key='seat_preference',
    value='window seat', or key='home_airport', value='DEL'. Use short,
    consistent keys so they can be looked up again later."""
    user_id = runtime.state["user_id"]
    await runtime.store.aput(("preferences", user_id), key, {"value": value})
    return f"Got it -- I'll remember {key} = {value}."


@tool
async def recall_preferences(runtime: ToolRuntime) -> str:
    """List everything currently remembered about the user's preferences,
    across all past conversations."""
    user_id = runtime.state["user_id"]
    items = await runtime.store.asearch(("preferences", user_id))
    if not items:
        return "I don't have any saved preferences for you yet."
    return "\n".join(f"{item.key}: {item.value.get('value')}" for item in items)
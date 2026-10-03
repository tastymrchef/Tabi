import asyncio, sys
sys.path.insert(0, ".")
from src.memory.db import postgres_backend, get_db_uri
from src.agent.workflow import build_agent_graph

def summarize(messages, label):
    print(f"\n===== {label} =====")
    for m in messages:
        role = m.__class__.__name__
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                print(f"[{role}] TOOL_CALL -> {tc['name']}({tc['args']})")
        content = getattr(m, "content", None)
        if content:
            text = content if isinstance(content, str) else str(content)
            print(f"[{role}] {text[:400]}")

async def main():
    async with postgres_backend(get_db_uri()) as (checkpointer, store):
        await checkpointer.setup()
        await store.setup()
        agent_graph = await build_agent_graph(checkpointer, store)
        config = {"configurable": {"thread_id": "fix-verify-thread-1"}}
        result1 = await agent_graph.ainvoke(
            {"messages": [{"role": "user", "content": "can you plan a day for me in Kyoto?"}], "user_id": "sahil"},
            config=config,
        )
        summarize(result1["messages"], "TURN 1 (vague request)")

asyncio.run(main())

import asyncio
import sys

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

        # Turn 1: vague day-plan request with no anchor.
        result1 = await agent_graph.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": "can you plan a day for me in Kyoto?"}
                ],
                "user_id": "sahil",
            },
            config=config,
        )
        summarize(result1["messages"], "TURN 1 (vague request)")

        # Turn 2: user narrows it down after the clarifying question, but
        # still vaguely -- this is expected to hit the recursion limit
        # again and prompt a further narrowing question (already confirmed
        # working).
        result2 = await agent_graph.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "something rural, not sure which area though",
                    }
                ],
                "user_id": "sahil",
            },
            config=config,
        )
        summarize(result2["messages"], "TURN 2 (still vague)")

        turn2_new_messages = result2["messages"][len(result1["messages"]):]
        called_plan_day_again_turn2 = any(
            getattr(m, "tool_calls", None)
            and any(tc["name"] == "plan_day" for tc in m.tool_calls)
            for m in turn2_new_messages
        )

        # Turn 3: user names a specific, concrete area -- the actual happy
        # path this whole fix chain is meant to unlock: plan_day should be
        # called again with a narrow, anchored request, and this time
        # actually succeed (return a real DayPlan) instead of hitting the
        # recursion limit.
        result3 = await agent_graph.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "let's do Arashiyama",
                    }
                ],
                "user_id": "sahil",
            },
            config=config,
        )
        summarize(result3["messages"], "TURN 3 (names a specific area)")

        turn3_new_messages = result3["messages"][len(result2["messages"]):]
        called_plan_day_again_turn3 = any(
            getattr(m, "tool_calls", None)
            and any(tc["name"] == "plan_day" for tc in m.tool_calls)
            for m in turn3_new_messages
        )
        plan_day_succeeded_turn3 = any(
            role_name == "ToolMessage"
            and "wasn't able to finish" not in (content or "")
            for role_name, content in (
                (m.__class__.__name__, getattr(m, "content", None))
                for m in turn3_new_messages
            )
            if role_name == "ToolMessage"
        )

        print(f"\n===== VERDICT =====")
        print(f"plan_day called again in turn 2: {called_plan_day_again_turn2}")
        print(f"plan_day called again in turn 3: {called_plan_day_again_turn3}")
        print(f"plan_day succeeded (real plan, no recursion-limit bailout) in turn 3: {plan_day_succeeded_turn3}")


if __name__ == "__main__":
    asyncio.run(main())

"""
A small, repeatable eval script for the trip-facts behaviors we tightened
by hand in conversation over the last few prompt changes.

This is what "error analysis" turns into once you've found and fixed a
real failure: instead of trusting that you'll remember to manually
re-ask the same question after the next prompt or model change, you
write the scenario down once as a script with a real assertion, and
re-run it any time something touching these prompts changes.

Deliberately simple, and deliberately NOT mocked -- it runs against the
real agent graph, the real Postgres store, and the real OpenAI model, so
each run costs a handful of actual API calls. That's the honest
tradeoff: a mocked version would be fast and free but couldn't actually
tell you whether gpt-4o-mini still does the right thing with the current
prompt wording.

Two kinds of checks on purpose:
  - Checks against the STORED STATE (store.asearch) for anything that
    has a structured, correct answer -- e.g. "was a trip fact saved with
    the right dates". This is the strongest kind of check: it can't be
    fooled by the model writing a nice-sounding reply that doesn't match
    what actually got persisted.
  - Checks against the final REPLY TEXT for behaviors that only show up
    in what the agent says, not in stored state -- e.g. "did the day
    plan actually anchor to Sapporo". Necessarily fuzzier (a substring
    check), but still far better than eyeballing it by hand in Telegram.

Uses a dedicated eval user_id so this never touches your real trip data
saved under your actual Telegram user id.
"""

import asyncio
import sys
import uuid

sys.path.insert(0, ".")

from src.memory.db import postgres_backend, get_db_uri
from src.agent.workflow import build_agent_graph

EVAL_USER_ID = "eval-user-trip-facts"


class EvalResult:
    def __init__(self, name):
        self.name = name
        self.passed = True
        self.notes = []

    def check(self, condition, message):
        self.notes.append(f"{'OK  ' if condition else 'FAIL'} - {message}")
        if not condition:
            self.passed = False


async def ask(agent_graph, thread_id, text, user_id=EVAL_USER_ID):
    config = {"configurable": {"thread_id": thread_id}}
    result = await agent_graph.ainvoke(
        {"messages": [{"role": "user", "content": text}], "user_id": user_id},
        config=config,
    )
    return result["messages"][-1].content or ""


async def seed_sapporo_fact(store):
    await store.aput(
        ("trip_facts", EVAL_USER_ID),
        "flight:2026-10-08:Sapporo, Japan",
        {
            "fact_type": "flight",
            "location": "Sapporo, Japan",
            "start_date": "2026-10-08",
            "end_date": "2026-10-13",
            "details": "Tokyo Narita to Sapporo New Chitose and back.",
        },
    )


async def eval_fact_storage(agent_graph, store):
    """Telling Tabi a concrete trip plan in plain text should produce a
    real, correctly-dated trip_facts record -- not just a reply that
    sounds right."""
    result = EvalResult("fact_storage")
    thread_id = f"eval-fact-storage-{uuid.uuid4().hex[:8]}"

    await ask(
        agent_graph, thread_id,
        "I'm flying from Delhi to Bangkok on 2027-01-05, returning on 2027-01-12.",
    )

    items = await store.asearch(("trip_facts", EVAL_USER_ID))
    matches = [
        item for item in items
        if item.value.get("start_date") == "2027-01-05"
        and "bangkok" in item.value.get("location", "").lower()
    ]
    result.check(len(matches) >= 1, "a trip fact for the Bangkok trip was saved")
    if matches:
        result.check(
            matches[0].value.get("end_date") == "2027-01-12",
            f"end_date is 2027-01-12 (got {matches[0].value.get('end_date')!r})",
        )
    return result


async def eval_summary_no_hallucination(agent_graph, store):
    """This is the exact production bug: asking for a trip summary after
    a day-plan was discussed earlier in the same conversation shouldn't
    blend that day-plan in as a fake travel leg with an invented date."""
    result = EvalResult("summary_no_hallucination")
    thread_id = f"eval-summary-{uuid.uuid4().hex[:8]}"

    await seed_sapporo_fact(store)

    # Recreate the real failure condition: a day-trip plan discussed
    # earlier in the SAME thread, before asking for the summary.
    await ask(
        agent_graph, thread_id,
        "Plan a day trip to Kamakura for me -- temples and the Great Buddha.",
    )
    reply = await ask(agent_graph, thread_id, "What's my trip itinerary so far?")

    result.check(
        "kamakura" not in reply.lower(),
        "summary doesn't mention Kamakura (that was a day-plan idea, not a trip fact)",
    )
    result.check("sapporo" in reply.lower(), "summary does mention the real Sapporo flight")
    result.notes.append(f"      reply: {reply[:300]}")
    return result


# Phrases that show up in plan_day's own failure fallback, or in the
# main agent's "I couldn't / too broad, can you clarify" response to a
# failed or declined plan_day call. A reply containing the destination
# name is NOT a pass if it's actually one of these -- that's a failure
# dressed up as a mention, not a working day plan. Caught this the hard
# way: the first version of this eval only checked for "sapporo" and
# scored a recursion-limit failure as a PASS because the failure message
# happened to name the city.
FAILURE_PHRASES = [
    "couldn't complete",
    "wasn't able to finish",
    "too broad",
    "could you specify",
    "narrower",
]


async def eval_home_base_anchoring(agent_graph, store):
    """plan_day should anchor to a known flight/accommodation window
    without being told the city again -- and actually produce a plan,
    not just mention the right city while failing."""
    result = EvalResult("home_base_anchoring")
    thread_id = f"eval-anchoring-{uuid.uuid4().hex[:8]}"

    await seed_sapporo_fact(store)

    reply = await ask(
        agent_graph, thread_id,
        "Plan a day for me on October 10, 2026.",  # city deliberately omitted
    )
    reply_lower = reply.lower()

    result.check("sapporo" in reply_lower, "anchored to Sapporo without being told the city")
    hit = next((p for p in FAILURE_PHRASES if p in reply_lower), None)
    result.check(
        hit is None,
        f"didn't fall back to a failure/clarifying response{f' (found: {hit!r})' if hit else ''}",
    )
    result.notes.append(f"      reply: {reply[:300]}")
    return result


async def main():
    async with postgres_backend(get_db_uri()) as (checkpointer, store):
        await checkpointer.setup()
        await store.setup()
        agent_graph = await build_agent_graph(checkpointer, store)

        evals = [eval_fact_storage, eval_summary_no_hallucination, eval_home_base_anchoring]
        results = []
        for eval_fn in evals:
            print(f"\n=== running {eval_fn.__name__} ===")
            res = await eval_fn(agent_graph, store)
            for note in res.notes:
                print(note)
            results.append(res)

        print("\n===== SUMMARY =====")
        all_passed = True
        for res in results:
            status = "PASS" if res.passed else "FAIL"
            all_passed = all_passed and res.passed
            print(f"{status} - {res.name}")

        if not all_passed:
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

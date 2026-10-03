"""
Block B -- trip-fact memory tools.

Tabi's goal here isn't to parse documents for their own sake -- it's to
keep a running timeline of a user's actual trip: flights, reservations,
and which place is "home base" for which stretch of days. That timeline
gets filled in two ways: the user just telling Tabi in conversation
("I'm staying in Sapporo for 5 days"), or forwarding a document/photo
(a booking confirmation, a ticket) that the main agent reads itself --
gpt-4o-mini is vision-capable, so no separate OCR/parsing library is
needed; the Telegram interface hands images straight to the agent as
part of the message content.

Like preferences.py, this reads/writes the long-term Store (persists
across every thread for a user), namespaced separately as "trip_facts"
so a day-plan's home-base/date-range info doesn't get mixed in with
freeform preferences.
"""

from langchain.tools import tool, ToolRuntime


@tool
async def remember_trip_fact(
    fact_type: str,
    location: str,
    start_date: str,
    end_date: str,
    details: str,
    runtime: ToolRuntime,
) -> str:
    """Save a concrete fact about the user's trip -- a flight, an
    accommodation booking, a reservation, or any other dated plan -- so
    it's remembered across conversations, not just this one.

    fact_type: one of 'flight', 'accommodation', 'transit', 'reservation',
        'other'.
    location: the place this fact is about -- for a trip or flight, use
        the DESTINATION (e.g. 'Sapporo, Japan'), not the departure
        point. For accommodation, the place you're actually staying.
    start_date: ISO date (YYYY-MM-DD) this fact starts applying -- a
        trip's outbound/departure date, or a stay's check-in date.
    end_date: ISO date (YYYY-MM-DD) this fact stops applying. For a
        ROUND TRIP (a flight there and back, a full stay), this is the
        return or check-out date -- it is NOT the same as start_date,
        even though the round trip is saved as a single fact. Only use
        the same value as start_date for a genuinely single-day,
        one-way event (a one-way flight with no return, a single
        reservation with no end).
    details: anything else worth keeping -- flight numbers for BOTH
        legs if it's a round trip, confirmation code, check-in/check-out
        time, address, cost, etc. Only include what you actually know;
        don't guess or fill in placeholders.

    Use this whenever the user tells you about a travel plan directly
    ('I'm staying in Sapporo for 5 days starting the 10th', 'flying to
    Bangkok on the 5th, back on the 12th'), or after reading a
    document/photo they send you (a booking confirmation, a ticket) --
    extract the real facts from it first, then call this. A round trip
    is always ONE call covering the full start_date-to-end_date range --
    never split a round trip into two separate calls for the outbound
    and return legs.
    """
    user_id = runtime.state["user_id"]
    fact_id = f"{fact_type}:{start_date}:{location}"
    await runtime.store.aput(
        ("trip_facts", user_id),
        fact_id,
        {
            "fact_type": fact_type,
            "location": location,
            "start_date": start_date,
            "end_date": end_date,
            "details": details,
        },
    )
    return (
        f"Saved: {fact_type} at {location} from {start_date} to {end_date}"
        f"{f' -- {details}' if details else ''}."
    )


@tool
async def recall_trip_facts(runtime: ToolRuntime) -> str:
    """List everything currently known about the user's trip -- flights,
    accommodations, reservations, and date ranges -- across all past
    conversations. Call this before planning a day or answering anything
    that depends on where the user is or what's already booked."""
    user_id = runtime.state["user_id"]
    items = await runtime.store.asearch(("trip_facts", user_id))
    if not items:
        return "I don't have any saved trip facts for you yet."

    facts = sorted(items, key=lambda item: item.value.get("start_date", ""))
    lines = []
    for item in facts:
        v = item.value
        line = (
            f"{v.get('fact_type')}: {v.get('location')} "
            f"({v.get('start_date')} to {v.get('end_date')})"
        )
        if v.get("details"):
            line += f" -- {v['details']}"
        lines.append(line)
    return "\n".join(lines)

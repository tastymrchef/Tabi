"""
Block D -- the day-planning tool.

plan_day is a tool the main Tabi agent can call, but under the hood
it's a small, separate agent of its own -- with its own system prompt
and its own structured response_format (DayPlan). Everything else Tabi
does stays plain free-text chat; only this one tool forces its answer
into a fixed shape, because a day plan is something you'd actually want
to reuse or display as structured data, not read as a paragraph.

It reuses the SAME real toolbelt tools (weather, currency, web search,
places) rather than reimplementing any research logic -- the LLM inside
this sub-agent still does the actual reasoning and tool-calling; we're
just wrapping its final answer in a schema.
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent

from src.tools.datetime_tool import get_current_datetime
from src.tools.weather import get_weather
from src.tools.currency import convert_currency
from src.tools.web_search import web_search
from src.tools.places import get_places_tools
from src.planner.schema import DayPlan

from langgraph.errors import GraphRecursionError

load_dotenv()

PLANNER_SYSTEM_PROMPT = (
    "You are a trip-planning specialist. Given a destination and whatever "
    "constraints the user gives you (dates, interests, pace, budget), "
    "research real options using your tools -- check the weather, look up "
    "actual nearby places for food/activities, and factor in currency "
    "conversions if costs come up. Don't invent specific venue names you "
    "haven't actually looked up.\n\n"
    "Build a realistic, time-ordered day plan: a handful of blocks "
    "(morning/afternoon/evening or specific times), each with a real "
    "activity and location. Keep it grounded in what your tools actually "
    "returned, not generic tourist-guide filler."
    "\n\nIf the request includes known preferences about the user, treat "
    "dietary restrictions as non-negotiable -- never suggest a restaurant "
    "or dish that violates one. Preferences about pace, interests, or "
    "style should shape your choices but aren't strict rules."
    "\n\nBefore calling query_nearby, decide on a small number of areas to "
    "build the day around. If the user describes a home base they're "
    "staying near plus one or more places they're visiting that day (e.g. "
    "a day trip out and back, or a home base plus a district across town), "
    "count only the places being visited toward the area limit -- at most "
    "2 -- not the home base itself. Otherwise, keep to at most 2 areas "
    "total, not scattered across the whole city. A relaxed day plan should "
    "minimize transit between distant neighborhoods, not maximize how much "
    "ground it covers."
    "\n\nAlways geocode an area with search_places before calling "
    "query_nearby in it -- never guess coordinates for a named place. Call "
    "search_places once per area, then reuse those coordinates for every "
    "query_nearby call in that area. Never repeat a query_nearby call for "
    "the same area and category combination more than once -- if you "
    "already looked up restaurants in an area, don't look them up again "
    "there for any reason, including after geocoding more precisely.\n\n"
    "For each area, pick only 1-2 amenity categories that are actually "
    "relevant to what the user described -- don't default to checking "
    "every category (cafe, bar, hotel, park, attraction) at every "
    "location. A day trip doesn't need hotel lookups; nothing about "
    "greenery or relaxing implies checking bars."
    "\n\nIf the request gives you no interests or categories to go on at "
    "all (not even a loose hint like 'relaxing' or 'shopping'), don't "
    "try to research your way to an answer by checking several "
    "categories to see what's good there -- that exploration is exactly "
    "what blows the research budget below. Default immediately to one "
    "sightseeing/attraction category plus one food category per area, "
    "build the plan from whatever those return, and move on."
    "\n\nKeep your research tightly bounded: look up at most 2 general "
    "areas OR at most 2 specific named sights total -- not both. Call "
    "query_nearby no more than 4-6 times in total across the whole day, "
    "counting each area/category combination only once. If the user asks "
    "for travel time or timing between areas, use a single web_search per "
    "leg of travel at most, and only if genuinely needed -- don't research "
    "transit options exhaustively. Once you have reasonable options for "
    "each part of the day, stop researching and build the plan from what "
    "you have -- don't keep discovering new places to check."
    "\n\nIf known trip facts are provided (flights, accommodation "
    "bookings, reservations), use each fact's location as the user's "
    "home base for every day inside its start_date/end_date range -- this "
    "applies to an accommodation booking just as much as a flight record "
    "that spans a trip to a destination (the range between its outbound "
    "and return legs is the window the user is actually there), even if "
    "the request itself doesn't repeat it. If more than one fact's range "
    "covers the day being planned (e.g. a whole-trip flight range and a "
    "more specific city visited partway through), use the narrower, more "
    "specific one -- that's the one that reflects where the user actually "
    "is that day. Call get_current_datetime if you need to work out what "
    "day is actually being planned (e.g. 'today' or 'tomorrow'). A flight "
    "landing or departing that day is also part of the day's reality -- "
    "don't plan activities that conflict with it (e.g. a full day of "
    "sightseeing on a day with a late-night departure should leave real "
    "time to get to the airport)."
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


async def build_planner_agent():
    places_tools = await get_places_tools()
    tools = [get_current_datetime, get_weather, convert_currency, web_search, *places_tools]

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        response_format=DayPlan,
    )


def _format_day_plan(plan: DayPlan) -> str:
    lines = [f"Day plan for {plan.destination} -- {plan.date_or_day}", plan.summary, ""]
    for block in plan.blocks:
        line = f"{block.time}: {block.activity} ({block.location})"
        if block.notes:
            line += f" -- {block.notes}"
        lines.append(line)
    return "\n".join(lines)


@tool
async def plan_day(request: str, runtime: ToolRuntime) -> str:
    """Build a full day itinerary for a trip. Use this when the user
    wants an actual plan for a day (or asks you to 'design my day',
    plan an itinerary, etc.), not for a single quick lookup. Pass the
    user's full request along, including destination and any
    preferences they mentioned (dates, interests, pace, budget)."""
    user_id = runtime.state["user_id"]
    saved = await runtime.store.asearch(("preferences", user_id))
    preferences_text = "; ".join(f"{item.key}: {item.value.get('value')}" for item in saved)

    trip_facts = await runtime.store.asearch(("trip_facts", user_id))
    trip_facts = sorted(trip_facts, key=lambda item: item.value.get("start_date", ""))
    trip_facts_text = "; ".join(
        f"{item.value.get('fact_type')} at {item.value.get('location')} "
        f"({item.value.get('start_date')} to {item.value.get('end_date')})"
        + (f" -- {item.value['details']}" if item.value.get("details") else "")
        for item in trip_facts
    )

    context_blocks = []
    if preferences_text:
        context_blocks.append(
            f"Known preferences about this user (always respect these unless "
            f"the request explicitly overrides them): {preferences_text}"
        )
    if trip_facts_text:
        context_blocks.append(
            f"Known trip facts (flights, accommodation, reservations) -- "
            f"figure out the user's home base for the day being planned "
            f"from whichever accommodation's date range covers it, rather "
            f"than asking: {trip_facts_text}"
        )

    full_request = request
    if context_blocks:
        full_request = "\n\n".join(context_blocks) + f"\n\nRequest: {request}"

    planner_agent = await build_planner_agent()
    try:
        result = await planner_agent.ainvoke(
            {"messages": [{"role": "user", "content": full_request}]},
            config={"recursion_limit": 24},
        )
    except GraphRecursionError:
        return (
            "I wasn't able to finish a full day plan within a reasonable "
            "amount of research -- the request may need to be narrower "
            "(e.g. one area, or fewer types of places) so I can stay "
            "focused rather than exploring too much."
        )

    plan: DayPlan = result["structured_response"]
    return _format_day_plan(plan)
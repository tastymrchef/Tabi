"""
Block D -- structured output shape for a day plan. Used as the
response_format for the planning sub-agent in planner/workflow.py, so
its final answer comes back as this, not a paragraph of prose.
"""

from pydantic import BaseModel, Field


class ActivityBlock(BaseModel):
    time: str = Field(description="Approximate time or time range, e.g. '9:00 AM' or 'Morning'")
    activity: str = Field(description="What to do")
    location: str = Field(description="Where, as a place name")
    notes: str = Field(default="", description="Optional practical notes -- cost, weather, travel time, etc.")


class DayPlan(BaseModel):
    destination: str = Field(description="The place this plan is for")
    date_or_day: str = Field(description="The date or day label this plan covers")
    summary: str = Field(description="One or two sentence overview of the day")
    blocks: list[ActivityBlock] = Field(description="The itinerary, in chronological order")
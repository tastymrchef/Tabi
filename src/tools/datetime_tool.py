"""
Originally defined inline in workflow.py (Block A). Pulled out into its
own file now that the Block D planner needs it too -- importing it
straight from workflow.py would create a circular import.
"""

from datetime import datetime
from langchain_core.tools import tool


@tool
def get_current_datetime() -> str:
    """Return the current date and time. Useful when the user asks
    what day/time it is, or for anything relative to 'today'."""
    return datetime.now().strftime("%A, %Y-%m-%d %H:%M")
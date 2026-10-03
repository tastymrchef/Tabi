"""
Block B, tool 4 -- web_search.

A general-purpose search tool for anything the other tools can't answer
directly: transport options, visa/entry rules, opening hours, local
customs, etc. Also useful later in Block D when the planner needs to
research things beyond weather/places/currency.

Uses Tavily (https://tavily.com) -- an API built specifically for LLM
agents, so results come back as clean text rather than raw HTML you'd
need to parse yourself. Requires a free API key (1000 searches/month on
the free tier) -- sign up at tavily.com and add it to your .env as
TAVILY_API_KEY.
"""

import os
import requests
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date information not covered by the other
    tools -- e.g. transport options between two places, visa/entry
    requirements, local customs, or anything else that needs a real,
    current answer rather than a guess. Keep the query specific and
    factual, e.g. 'train from Kyoto to Osaka duration' rather than a
    vague topic."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Web search failed: TAVILY_API_KEY is not set."

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "query": query,
                "search_depth": "basic",
                "max_results": 5,
                "include_answer": True,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return f"Web search failed: {e}"

    lines = []
    if data.get("answer"):
        lines.append(f"Summary: {data['answer']}")

    for result in data.get("results", []):
        lines.append(f"- {result['title']} ({result['url']}): {result['content']}")

    if not lines:
        return f"No results found for '{query}'."

    return "\n".join(lines)
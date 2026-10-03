"""
Block B, tool 3 -- convert_currency.

Same plain-HTTP pattern as get_weather: no API key, no MCP, just a
straightforward GET request. Deliberately simple -- the MCP integration
was the hard part, this one reinforces the "plain tool" shape before we
move on to Block C.

Uses the Frankfurter API (https://frankfurter.dev) -- free, no key
required, built on European Central Bank reference rates (updated once
per banking day). Good enough for travel budgeting; not for real-time
trading or anything time-sensitive to the minute.
"""

import requests
from langchain_core.tools import tool


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount from one currency to another using current
    exchange rates. Currencies are ISO 4217 codes, e.g. 'USD', 'JPY', 'EUR'.
    Example: convert_currency(100, 'USD', 'JPY') converts $100 to yen."""
    try:
        response = requests.get(
            "https://api.frankfurter.dev/v1/latest",
            params={
                "amount": amount,
                "from": from_currency.upper(),
                "to": to_currency.upper(),
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return f"Currency conversion failed: {e}"

    converted = data["rates"].get(to_currency.upper())
    if converted is None:
        return f"Couldn't find a rate for {to_currency.upper()}."

    return (
        f"{amount} {from_currency.upper()} = {converted:.2f} {to_currency.upper()} "
        f"(rate date: {data['date']})"
    )
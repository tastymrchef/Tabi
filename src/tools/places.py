"""
Block B, tool 2 -- and Tabi's first real MCP integration.

Unlike get_weather, this doesn't call an HTTP API directly. It connects to
a real MCP server (openstreetmap-mcp-server by cyanheads) and asks IT what
tools it offers, at runtime, rather than us writing find_nearby_places
ourselves.

The server wraps two free, no-key OpenStreetMap services: Nominatim
(geocoding) and the Overpass API (spatial queries). It ships as an npm
package, launched as a local subprocess via `npx`.

A real, confirmed problem: three of the server's six tools
(search_places, query_nearby, query_bbox) have a top-level `anyOf` in
their argument schema -- the server's valid way of saying "give me shape
A or shape B, never both." OpenAI's function-calling API rejects any tool
schema with oneOf/anyOf/allOf/enum/const/not at the top level, and
langchain.mcp (still beta) doesn't adapt the server's schema to fit that
restriction -- it passes it through unmodified. Confirmed directly: handing
all six tools to the agent got a 400 from OpenAI naming search_places by
exact schema complaint.

The fix: thin wrappers around the two of those three we actually need,
exposing a plain interface (ordinary str/float arguments, no combinators)
that calls the real MCP tool underneath with one fixed, valid argument
combination. The other three tools (reverse_geocode, lookup_objects,
query_raw) have clean schemas already and are passed straight through --
that's the actual "let the server define the tool" MCP pattern working as
intended, for the tools OpenAI's restriction doesn't get in the way of.
"""

from langchain_core.tools import tool
from langchain.mcp import MCPAdapter

MCP_CONFIG = {
    "mcpServers": {
        "openstreetmap": {
            "command": "npx",
            "args": ["-y", "@cyanheads/openstreetmap-mcp-server@latest"],
        }
    }
}

PASSTHROUGH_TOOL_NAMES = {
    "openstreetmap_reverse_geocode",
    "openstreetmap_lookup_objects",
    "openstreetmap_query_raw",
}


async def get_places_tools() -> list:
    async with MCPAdapter(MCP_CONFIG) as adapter:
        raw_tools = await adapter.list_tools()

    by_name = {t.name: t for t in raw_tools}

    @tool
    async def search_places(query: str) -> str:
        """Geocode a place name or address to coordinates, e.g.
        'Kinkaku-ji, Kyoto' or 'Eiffel Tower, Paris'. Give the name plus its
        city/region -- not a full street-level address hierarchy."""
        result = await by_name["openstreetmap_search_places"].ainvoke({"query": query})
        return str(result)

    @tool
    async def query_nearby(lat: float, lon: float, amenity: str, radius_meters: float = 1000) -> str:
        """Find places of one specific kind within radius_meters of (lat, lon).
        `amenity` is required -- the underlying service has no 'anything nearby'
        mode, you must commit to one category per call. Common values:
        'restaurant', 'cafe', 'bar', 'fast_food', 'atm', 'pharmacy', 'hospital',
        'hotel', 'bank', 'fuel'. Get lat/lon from search_places first. To cover
        more than one category, call this again with a different amenity."""
        result = await by_name["openstreetmap_query_nearby"].ainvoke(
            {"lat": lat, "lon": lon, "radius_meters": radius_meters, "amenity": amenity}
        )
        return str(result)

    passthrough_tools = [t for t in raw_tools if t.name in PASSTHROUGH_TOOL_NAMES]

    return [search_places, query_nearby, *passthrough_tools]
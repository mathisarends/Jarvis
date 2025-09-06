from datetime import datetime
from tavily import AsyncTavilyClient
from agents import Agent, Runner, function_tool
from dotenv import load_dotenv
from agent.realtime.tools.web_search.views import TavilyResponse
from agent.realtime.tools.location.location import get_current_location


@function_tool(
    docstring_style="google"
)  # auto-detect klappt oft, hier erzwingen wir "google"
async def _tavily_web_search(query: str, max_results: int = 2) -> str:
    """Perform a Tavily web search and return a compact, voice-friendly summary.

    Args:
        query: Refined, specific search string (include location/weekday if relevant).
        max_results: Max number of results to include in the summary (default 2).
    """
    client = AsyncTavilyClient()
    response_dict = await client.search(
        query=query,
        max_results=max_results,
        search_depth="basic",
    )
    response = TavilyResponse.model_validate(response_dict)
    lines = [
        f"• {r.title} — {r.content.strip().replace('\n', ' ')} ({r.url})"
        for r in response.results
        if r.content
    ]

    if not lines:
        return "Leider konnte ich keine relevanten Informationen finden."
    return "\n".join(lines)


if __name__ == "__main__":
    load_dotenv()
    import json

    print(_tavily_web_search.name)
    print(_tavily_web_search.description)
    schema_dict = _tavily_web_search.params_json_schema
    print(json.dumps(schema_dict, indent=2))

from datetime import datetime
from tavily import AsyncTavilyClient
from agents import Agent, Runner, function_tool
from dotenv import load_dotenv
from agent.realtime.tools.web_search.views import TavilyResponse
from agent.realtime.tools.location.location import get_current_location

load_dotenv()


# Idee auch hier per Generator arbeiten, um Zwischenergebnisse zu streamen?
async def run_web_search_agent(query: str) -> str:
    """
    Main entrypoint for web search.
    Delegates query refinement and execution to an OpenAI Agent with tools.
    """
    agent = Agent(
        name="WebSearchAgent",
        model="gpt-5-mini",
        instructions=(
            "You are a web search agent. Your primary goal is to answer queries with fresh, accurate facts.\n"
            "\n"
            "DECISION RULES (very important):\n"
            "1) If the user asks for opening hours, phone numbers, local businesses, events, or anything location-dependent,\n"
            "   you MUST first call get_current_location_tool() unless the query already contains an explicit location.\n"
            "2) If the query concerns opening hours/schedules and no weekday is specified, you MUST first call get_current_week_day().\n"
            "3) Refine the search query by appending the resolved location and/or weekday when relevant.\n"
            "4) Then call tavily_web_search(refined_query). Do not skip it for current/volatile info.\n"
            "\n"
            "REFINEMENT GUIDELINES:\n"
            "- Prefer precise entity names + city/region (e.g., 'Hafen 5 Aurich Öffnungszeiten Saturday').\n"
            "- Avoid verbose natural language in the query; keep it compact and keyword-focused.\n"
            "\n"
            "OUTPUT FORMAT (voice-friendly):\n"
            "- Respond with one short natural sentence.\n"
            "- If the query is about opening hours, use the template: 'Heute hat <Name> von <Startzeit> bis <Endzeit> geöffnet.'\n"
            "- Do not use bullet points, lists, or markdown; keep it natural and continuous so it can be read aloud.\n"
            "\n"
            "EXAMPLE:\n"
            "User: 'Öffnungszeiten Hafen 5'\n"
            "You: (call get_current_location_tool -> 'Aurich, Lower Saxony, Germany'; get_current_week_day -> 'Saturday')\n"
            "     (refine) 'Hafen 5 Aurich Öffnungszeiten Saturday'\n"
            "     (search) tavily_web_search(...)\n"
            "     (answer) 'Heute hat Hafen 5 von 10 bis 18 Uhr geöffnet.'\n"
            "\n"
            "If neither location nor weekday is relevant, skip those tools and just search.\n"
        ),
        tools=[_get_current_week_day, _get_current_location_tool, _tavily_web_search],
    )

    result = await Runner.run(agent, query)
    return result.final_output


@function_tool
async def _get_current_location_tool() -> str:
    """
    Use this to resolve the user's current location (city, region, country).
    Always call this BEFORE searching when a query is location-dependent or ambiguous
    (e.g., opening hours, 'in meiner Nähe', local businesses, events, pharmacies, supermarkets).
    Return a short one-line string suitable to append into a search query.
    """
    location = await get_current_location()
    return f"{location.city}, {location.region}, {location.country}"


@function_tool
async def _get_current_week_day() -> str:
    """
    Use this to disambiguate weekday-specific information (e.g., opening hours by weekday).
    Always call this BEFORE searching when the query mentions opening hours or schedules
    without an explicit weekday/date.
    Return the English weekday name, e.g., 'Saturday'.
    """
    now = datetime.now()
    return now.strftime("%A")  # e.g., "Saturday"


@function_tool
async def _tavily_web_search(query: str, max_results: int = 2) -> str:
    """
    Perform a web search via Tavily and return a compact, voice-friendly summary.
    Input: a refined, specific query that already includes location and weekday if relevant.
    Output: plain text, bullet-style lines: '• Title — key fact (source URL)'
    Keep it concise and readable aloud.
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

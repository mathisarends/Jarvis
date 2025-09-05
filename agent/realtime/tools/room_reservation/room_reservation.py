# Maybe event with human in the loop here would be precious
from collections.abc import AsyncGenerator
from browser_use import Agent, AgentHistoryList
from browser_use.browser.views import BrowserStateSummary
from browser_use.agent.views import AgentOutput
from browser_use.llm import ChatOpenAI, SystemMessage
import asyncio

_llm = ChatOpenAI(model="gpt-5-mini")


async def perform_browser_search(topic: str) -> AsyncGenerator[str, None]:
    """Performs a simulated browser search on a topic and streams the progress."""

    message_queue = asyncio.Queue()

    llm = ChatOpenAI(model="gpt-5-mini")
    task = "Find online information on the following topic: " + topic

    agent = Agent(
        llm=llm,
        task=task,
        register_new_step_callback=lambda bs, ao, sn: _new_step_callback(
            message_queue, bs, ao, sn
        ),
        register_done_callback=lambda ahl: _done_callback(message_queue, ahl),
        vision_detail_level="low",
    )

    agent_task = asyncio.create_task(agent.run())

    yield f"Starte Internet Recherche zu {topic}..."

    # Messages aus Queue yielden
    async for message in _process_message_queue(message_queue, agent_task):
        yield message


# Private helper functions below


async def _done_callback(
    message_queue: asyncio.Queue, _agent_history_list: AgentHistoryList
) -> None:
    """Callback when agent is done."""
    await message_queue.put("✅ Browser-Recherche abgeschlossen!")
    await message_queue.put(None)  # Signal für Ende


async def _new_step_callback(
    message_queue: asyncio.Queue,
    _browser_state_summary: BrowserStateSummary,
    agent_output: AgentOutput,
    step_number: int,
) -> None:
    """Callback for each new step."""
    # Status-Message in Queue einreihen
    status_message = f"🔍 Schritt {step_number}: {agent_output.current_state.thought}"
    await message_queue.put(status_message)


async def _process_message_queue(
    message_queue: asyncio.Queue, agent_task: asyncio.Task
) -> AsyncGenerator[str, None]:
    """Process messages from queue and yield them."""
    try:
        while True:
            done, pending = await _wait_for_message_or_completion(
                message_queue, agent_task
            )

            if agent_task in done:
                async for message in _yield_remaining_messages(message_queue):
                    yield message
                break

            message = await _extract_message_from_tasks(done, agent_task, pending)
            if message is None:
                return
            yield message

    except Exception as e:
        agent_task.cancel()
        yield f"❌ Fehler bei Browser-Automation: {str(e)}"


async def _wait_for_message_or_completion(
    message_queue: asyncio.Queue, agent_task: asyncio.Task
) -> tuple[set, set]:
    """Wait for next message or agent completion."""
    return await asyncio.wait(
        [asyncio.create_task(message_queue.get()), agent_task],
        return_when=asyncio.FIRST_COMPLETED,
    )


async def _yield_remaining_messages(
    message_queue: asyncio.Queue,
) -> AsyncGenerator[str, None]:
    """Yield all remaining messages when agent is done."""
    while not message_queue.empty():
        message = await message_queue.get()
        if message is not None:
            yield message


async def _extract_message_from_tasks(
    done: set, agent_task: asyncio.Task, pending: set
) -> str | None:
    for task in done:
        if task != agent_task:
            message = await task.result()
            for pending_task in pending:
                if pending_task != agent_task:
                    pending_task.cancel()
            return message
    return None

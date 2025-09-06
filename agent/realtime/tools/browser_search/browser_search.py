from textwrap import dedent
from collections.abc import AsyncGenerator
from browser_use import Agent, AgentHistoryList
from browser_use.browser.views import BrowserStateSummary
from browser_use.agent.views import AgentOutput
from browser_use.llm import ChatOpenAI, SystemMessage, UserMessage
import asyncio


class BrowserSearchManager:
    """Manages browser search operations with state management."""

    def __init__(self):
        self._llm = ChatOpenAI(model="gpt-5-mini")
        self.message_queue = None
        self.task = None
        self.agent_task = None

    async def search(self, topic: str) -> AsyncGenerator[str, None]:
        """Performs a browser search on a topic and streams the progress."""
        self.message_queue = asyncio.Queue()
        self.task = "Find online information on the following topic: " + topic

        agent = Agent(
            llm=self._llm,
            task=self.task,
            register_new_step_callback=self._new_step_callback,
            register_done_callback=self._done_callback,
            vision_detail_level="low",
        )

        self.agent_task = asyncio.create_task(agent.run())

        yield f"Starte Internet Recherche zu {topic}..."

        # Messages aus Queue yielden
        async for message in self._process_message_queue():
            yield message

    async def _done_callback(self, agent_history_list: AgentHistoryList) -> None:
        """Callback when agent is done."""
        result = agent_history_list.final_result()

        system_message = SystemMessage(
            content=dedent(
                """You are a research assistant tasked with summarizing web research results. 

            Your goal is to provide a clear, concise summary that directly answers the original research question. 

            Guidelines:
            - Focus on the key findings that address the original topic
            - Present information in a logical, easy-to-understand structure
            - Include the most important and relevant facts
            - Avoid unnecessary details or tangential information
            - If multiple perspectives exist, briefly mention them
            - Keep the summary comprehensive but concise
            - Write in a natural, conversational tone

            Summarize the research findings to directly answer what the user wanted to know about the topic."""
            )
        )

        human_message = UserMessage(
            content=f"Original research topic: {self.task}\n\nResearch results:\n{result}"
        )

        # Generiere Zusammenfassung
        summary_response = await self._llm.ainvoke(
            messages=[system_message, human_message]
        )
        summary = summary_response.completion

        await self.message_queue.put(summary)
        await self.message_queue.put(None)  # Signal für Ende

    async def _new_step_callback(
        self,
        browser_state_summary: BrowserStateSummary,
        agent_output: AgentOutput,
        step_number: int,
    ) -> None:
        """Callback for each new step."""
        agent_reasoning_trace = agent_output.current_state.thinking
        evaluation_previous_goal = agent_output.current_state.evaluation_previous_goal
        next_goal = agent_output.current_state.next_goal

        system_message = SystemMessage(
            content=dedent(
                """
            You are a personal assistant providing brief, natural status updates about a web research task. 
            Convert technical agent information into friendly, conversational updates in German.

            Guidelines:
            - Keep updates short and natural (1 sentence)
            - Use first person ("Ich...")
            - Sound human and engaging
            - Focus on what you're currently doing
            - Don't mention technical details like "step numbers" or "agent states"

            Examples:
            Technical: "Navigating to search engine, goal: find information about climate change"
            Natural: "Ich öffne gerade eine Suchmaschine"

            Technical: "Analyzing search results, found 10 articles about renewable energy"
            Natural: "Super, ich finde interessante Artikel zu erneuerbaren Energien"

            Technical: "Clicking on news article link, goal: gather recent information"
            Natural: "Ich schaue mir gerade aktuelle Nachrichten an"

            Technical: "Scrolling through page content, extracting relevant data"
            Natural: "Ich durchsuche die Seite nach wichtigen Informationen"

            Technical: "Opening new tab to cross-reference information"
            Natural: "Lass mich das noch in einer anderen Quelle überprüfen"

            Technical: "Loading Wikipedia page about artificial intelligence"
            Natural: "Ich schaue mir den Wikipedia-Artikel an"

            Technical: "Searching for recent news articles about topic"
            Natural: "Ich suche nach aktuellen Nachrichten"

            Technical: "Evaluating source credibility and relevance"
            Natural: "Ich prüfe gerade die Qualität der Quellen"

            Convert the following to a natural status update:
        """
            )
        )

        human_message = UserMessage(
            content=f"Step {step_number} - Evaluation previous goal: {evaluation_previous_goal}\nNext goal: {next_goal}\nReasoning: {agent_reasoning_trace}"
        )

        # Generiere natürliches Status-Update
        status_response = await self._llm.ainvoke(
            messages=[system_message, human_message]
        )
        natural_status = status_response.completion

        # Status-Message in Queue einreihen
        await self.message_queue.put(natural_status)

    async def _process_message_queue(self) -> AsyncGenerator[str, None]:
        """Process messages from queue and yield them."""
        try:
            while True:
                done, pending = await self._wait_for_message_or_completion()

                if self.agent_task in done:
                    async for message in self._yield_remaining_messages():
                        yield message
                    break

                message = self._extract_message_from_tasks(done, pending)
                if message is None:
                    return
                yield message

        except Exception as e:
            self.agent_task.cancel()
            yield f"❌ Fehler bei Browser-Automation: {str(e)}"

    async def _wait_for_message_or_completion(self) -> tuple[set, set]:
        """Wait for next message or agent completion."""
        return await asyncio.wait(
            [asyncio.create_task(self.message_queue.get()), self.agent_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

    async def _yield_remaining_messages(self) -> AsyncGenerator[str, None]:
        """Yield all remaining messages when agent is done."""
        while not self.message_queue.empty():
            message = await self.message_queue.get()
            if message is not None:
                yield message

    def _extract_message_from_tasks(self, done: set, pending: set) -> str | None:
        """Extract message from completed tasks."""
        for task in done:
            if task != self.agent_task:
                message = task.result()
                for pending_task in pending:
                    if pending_task != self.agent_task:
                        pending_task.cancel()
                return message
        return None


# Externe Funktion die die Klasse nutzt
async def perform_browser_search(topic: str) -> AsyncGenerator[str, None]:
    """Performs a simulated browser search on a topic and streams the progress."""
    search_manager = BrowserSearchManager()
    async for message in search_manager.search(topic):
        yield message

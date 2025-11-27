import asyncio

from agent.config import (
    ModelSettings,
    VoiceSettings,
    TranscriptionSettings,
    WakeWordSettings,
)
from agent.service import RealtimeAgent
from agent.realtime.events.client.session_update import (
    MCPRequireApprovalMode,
    MCPTool,
    NoiseReductionType,
)
from agent.realtime.views import AssistantVoice
from agent.wake_word import PorcupineWakeWord


async def main():
    """Run the voice assistant application."""

    mcp_tool = MCPTool(
        server_label="dmcp",
        server_url="https://dmcp-server.deno.dev/sse",
        require_approval=MCPRequireApprovalMode.NEVER,
    )

    # Configure settings
    model_settings = ModelSettings(
        instructions="Be concise and friendly. Answer in German. Always use tools if necessary.",
        temperature=0.8,
        mcp_tools=[mcp_tool],
    )

    voice_settings = VoiceSettings(
        assistant_voice=AssistantVoice.MARIN,
        speech_speed=1.3,
    )

    transcription_settings = TranscriptionSettings(
        enabled=False,
        noise_reduction_mode=NoiseReductionType.NEAR_FIELD,
    )

    wake_word_settings = WakeWordSettings(
        enabled=True,
        keyword=PorcupineWakeWord.PICOVOICE,
        sensitivity=0.7,
    )

    try:
        agent = RealtimeAgent(
            model_settings=model_settings,
            voice_settings=voice_settings,
            transcription_settings=transcription_settings,
            wake_word_settings=wake_word_settings,
        )

        await agent.start()

    except KeyboardInterrupt:
        pass  # Graceful shutdown
    except Exception:
        print("Critical application error")
        raise


if __name__ == "__main__":
    asyncio.run(main())

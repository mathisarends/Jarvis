import asyncio
import os
from typing import TypeVar, Generic

from agent.config.views import AgentConfig, WakeWordConfig
from agent.controller.service_factory import ServiceBundle, ServiceFactory
from agent.realtime.events.client.session_update import (
    InputAudioNoiseReductionConfig,
    NoiseReductionType,
    RealtimeModel,
    TranscriptionModel,
)
from agent.realtime.tools.tools import Tools
from agent.realtime.views import (
    AssistantVoice,
)
from audio.player.audio_strategy import AudioStrategy
from audio.wake_word_listener import PorcupineBuiltinKeyword
from shared.logging_mixin import LoggingMixin


Context = TypeVar("Context")


class RealtimeAgent(Generic[Context], LoggingMixin):
    def __init__(
        self,
        context: Context | None = None,
        model: RealtimeModel = RealtimeModel.GPT_REALTIME,
        instructions: str | None = None,
        response_temperature: float = 0.8,
        tools: Tools | None = None,
        tool_calling_model_name: str | None = None,
        assistant_voice: AssistantVoice = AssistantVoice.MARIN,
        speech_speed: float = 1.0,
        enable_transcription: bool = False,
        transcription_model: TranscriptionModel | None = None,
        transcription_language: str | None = None,
        transcription_prompt: str | None = None,
        noise_reduction_mode: NoiseReductionType | None = None,
        enable_wake_word: bool = False,
        wakeword: PorcupineBuiltinKeyword | None = None,
        wake_word_sensitivity: float = 0.7,
        audio_playback_strategy: AudioStrategy | None = None,
    ):
        # Store config (same as before)
        self.context = context
        self.model = model
        self.instructions = instructions
        self.response_temperature = response_temperature
        self.tools = tools if tools is not None else Tools()
        self.tool_calling_model_name = tool_calling_model_name
        self.assistant_voice = assistant_voice
        self.speech_speed = speech_speed
        self._validate_assistant_speech_speed()

        self.enable_transcription = enable_transcription

        # Standardmäßig None setzen; bei aktivierter Transkription ggf. überschreiben
        self.transcription_model = None
        if self.enable_transcription:
            self.transcription_model = (
                transcription_model
                if transcription_model is not None
                else TranscriptionModel.WHISPER_1
            )
        self.transcription_language = transcription_language
        self.transcription_prompt = transcription_prompt
        self._validate_transcription_config()

        self.noise_reduction_mode = noise_reduction_mode
        self.enable_wake_word = enable_wake_word
        if self.enable_wake_word:
            self.wakeword = (
                wakeword if wakeword is not None else PorcupineBuiltinKeyword.PICOVOICE
            )
            self.wake_word_sensitivity = (
                wake_word_sensitivity if wake_word_sensitivity is not None else 0.7
            )
        self._validate_wake_word_config()

        # Create services via factory
        agent_config = self._build_agent_config()
        wake_word_config = self._build_wake_word_config()

        self.audio_playback_strategy = audio_playback_strategy

        factory = ServiceFactory(agent_config, wake_word_config, self.tools)
        self.services: ServiceBundle = factory.create_services()

        # Application State
        self._running = False
        self._shutdown_event = asyncio.Event()

    async def run(self) -> None:
        """Run the voice assistant application."""
        if self._running:
            self.logger.warning("Agent already running")
            return

        try:
            self._assert_environment_variables()
            self.logger.info("Starting Voice Assistant")
            self._running = True

            # Play startup sound
            self.services.audio_manager.strategy.play_startup_sound()

            # Start the state machine
            await self.services.context.state.on_enter(self.services.context)

            # Main event-driven loop
            while self._running:
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=0.1)
                    break
                except asyncio.TimeoutError:
                    continue
                except KeyboardInterrupt:
                    self.logger.info("Shutdown requested by user")
                    break

        except KeyboardInterrupt:
            pass
        except Exception as e:
            self.logger.exception(f"Agent error: {e}")
            raise
        finally:
            await self._cleanup_all_services()

    async def stop(self) -> None:
        """Stop the voice assistant gracefully."""
        if not self._running:
            return

        self.logger.info("Stopping Voice Assistant")
        self._running = False
        self._shutdown_event.set()

    async def _cleanup_all_services(self) -> None:
        """Cleanup all services in parallel."""
        self.logger.info("Cleaning up all services...")

        cleanup_tasks = [
            self._cleanup_state_machine(),
            self._cleanup_wake_word_service(),
            self._cleanup_sound_service(),
        ]

        results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        service_names = ["state_machine", "wake_word", "sound"]
        for service_name, result in zip(service_names, results):
            if isinstance(result, Exception):
                self.logger.exception(f"Error cleaning up {service_name}: {result}")

        self.logger.info("All services cleaned up")

    async def _cleanup_state_machine(self) -> None:
        """Cleanup the state machine."""
        await self.services.context.state.on_exit(self.services.context)

    async def _cleanup_wake_word_service(self) -> None:
        """Cleanup wake word detection."""
        self.services.wake_word_listener.cleanup()

    async def _cleanup_sound_service(self) -> None:
        """Cleanup sound playback."""
        self.services.audio_manager.strategy.stop_sounds()

    def _build_agent_config(self) -> AgentConfig:
        """Build the agent configuration object from agent settings."""
        return AgentConfig(
            model=self.model,
            voice=self.assistant_voice,
            speed=self.speech_speed,
            instructions=self.instructions,
            temperature=self.response_temperature,
            input_audio_noise_reduction=InputAudioNoiseReductionConfig(
                type=self.noise_reduction_mode
            ),
            tool_calling_model_name=self.tool_calling_model_name,
        )

    def _build_wake_word_config(self) -> WakeWordConfig:
        """Build the wake word configuration object from agent settings."""
        return WakeWordConfig(
            keyword=self.wakeword or PorcupineBuiltinKeyword.PICOVOICE,
            sensitivity=self.wake_word_sensitivity,
        )

    def _validate_transcription_config(self) -> None:
        """Validate transcription configuration."""
        if not self.enable_transcription and self.transcription_model:
            self.logger.warning(
                "Transcription model provided but transcription is disabled. "
                "Model will be ignored. Set enable_transcription=True to use it."
            )

        if not self.enable_transcription and self.transcription_language:
            self.logger.warning(
                "Transcription language provided but transcription is disabled. "
                "Language will be ignored. Set enable_transcription=True to use it."
            )

        if not self.enable_transcription and self.transcription_prompt:
            self.logger.warning(
                "Transcription prompt provided but transcription is disabled. "
                "Prompt will be ignored. Set enable_transcription=True to use it."
            )

        # Validate transcription language format
        if self.enable_transcription and self.transcription_language:
            if not self._is_valid_language_code(self.transcription_language):
                self.logger.warning(
                    f"Invalid transcription language code: {self.transcription_language!r}. "
                    f"Expected ISO-639-1 format (e.g., 'en', 'de'). Using None instead."
                )
                self.transcription_language = None

    def _validate_wake_word_config(self) -> None:
        """Validate wake word configuration."""
        if not self.enable_wake_word and self.wakeword:
            self.logger.warning(
                "Wake word keyword provided but wake word detection is disabled. "
                "Keyword will be ignored. Set enable_wake_word=True to use it."
            )

        if not self.enable_wake_word and self.wake_word_sensitivity:
            self.logger.warning(
                "Wake word sensitivity provided but wake word detection is disabled. "
                "Sensitivity will be ignored. Set enable_wake_word=True to use it."
            )

    def _validate_assistant_speech_speed(self) -> None:
        min_speech_speed = 0.25
        max_speech_speed = 1.5

        """Validate and adjust speech speed to be within allowed range."""
        if self.speech_speed < min_speech_speed:
            self.logger.warning(
                "Speech speed (%.2f) below minimum (%.2f). Setting to minimum.",
                self.speech_speed,
                min_speech_speed,
            )
            self.speech_speed = min_speech_speed
        elif self.speech_speed > max_speech_speed:
            self.logger.warning(
                "Speech speed (%.2f) above maximum (%.2f). Setting to maximum.",
                self.speech_speed,
                max_speech_speed,
            )
            self.speech_speed = max_speech_speed

    def _is_valid_language_code(self, language: str) -> bool:
        """Validate language code format using early returns."""
        if not language or not isinstance(language, str):
            return False

        lang = language.strip().lower()
        if not lang:
            return False

        if len(lang) in (2, 3) and lang.isalpha():
            return True

        return False

    def _assert_environment_variables(self) -> None:
        """Validate that all required environment variables are set."""
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Missing OPENAI_API_KEY environment variable")

        if self.enable_wake_word and not os.getenv("PICO_ACCESS_KEY"):
            raise RuntimeError(
                "Missing PICO_ACCESS_KEY environment variable (required for wake word)"
            )

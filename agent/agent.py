import os
from typing import TypeVar, Generic

from agent.config.views import VoiceAssistantConfig, AgentConfig, WakeWordConfig
from agent.controller.voice_assistant_controller import VoiceAssistantController
from agent.realtime.events.client.session_update import (
    InputAudioNoiseReductionConfig,
    NoiseReductionType,
    RealtimeModel,
    TranscriptionModel,
)
from agent.realtime.tools.tool import Tool
from agent.realtime.views import (
    AssistantVoice,
)
from agents.models.interface import Model
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
        tools: list[Tool] = [],
        tool_calling_model=str | Model | None,
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
    ):
        self.context = context
        self.model = model
        self.instructions = instructions
        self.response_temperature = response_temperature
        self.tools = tools.copy() if tools else []
        self.tool_calling_model = tool_calling_model

        # Voice and Speech Settings
        self.assistant_voice = assistant_voice
        self.speech_speed = speech_speed
        self.min_speech_speed = 0.25
        self.max_speech_speed = 1.5
        self._validate_assistant_speech_speed()

        # Audio Transcription
        self.enable_transcription = enable_transcription

        if self.enable_transcription:
            self.transcription_model = (
                transcription_model
                if transcription_model is not None
                else TranscriptionModel.WHISPER_1
            )

        self.transcription_language = transcription_language
        self.transcription_prompt = transcription_prompt
        self._validate_transcription_config()

        # Audio Processing
        self.noise_reduction_mode = noise_reduction_mode

        # Wake Word Detection
        self.enable_wake_word = enable_wake_word

        if self.enable_wake_word:
            self.wakeword = (
                wakeword if wakeword is not None else PorcupineBuiltinKeyword.PICOVOICE
            )
            self.wake_word_sensitivity = (
                wake_word_sensitivity if wake_word_sensitivity is not None else 0.7
            )
        self._validate_wake_word_config()

        self._controller_config = self._get_voice_assistent_controller_config()
        self._controller = VoiceAssistantController(self._controller_config)

    async def run(self) -> None:
        """Run the voice assistant application."""
        try:
            self._assert_environment_variables()
            await self._controller.start()
        except KeyboardInterrupt:
            pass  # Graceful shutdown
        except Exception:
            print("Critical application error")
            raise

    # TODO: Diese Noise Reduction config hier gerne noch loswerden bitte
    def _get_voice_assistent_controller_config(self) -> VoiceAssistantConfig:
        """Build the configuration object from agent settings."""
        return VoiceAssistantConfig(
            agent=AgentConfig(
                model=self.model,
                voice=self.assistant_voice,
                speed=self.speech_speed,
                instructions=self.instructions,
                temperature=self.response_temperature,
                input_audio_noise_reduction=InputAudioNoiseReductionConfig(
                    type=self.noise_reduction_mode)
            ),
            wake_word=WakeWordConfig(
                keyword=self.wakeword or PorcupineBuiltinKeyword.PICOVOICE,
                sensitivity=self.wake_word_sensitivity,
            ),
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
        """Validate and adjust speech speed to be within allowed range."""
        if self.speech_speed < self.min_speech_speed:
            self.logger.warning(
                "Speech speed (%.2f) below minimum (%.2f). Setting to minimum.",
                self.speech_speed,
                self.min_speech_speed,
            )
            self.speech_speed = self.min_speech_speed
        elif self.speech_speed > self.max_speech_speed:
            self.logger.warning(
                "Speech speed (%.2f) above maximum (%.2f). Setting to maximum.",
                self.speech_speed,
                self.max_speech_speed,
            )
            self.speech_speed = self.max_speech_speed

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

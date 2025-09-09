from agent.realtime.event_bus import EventBus, on_event, on_event_with_data
from agent.realtime.views import ResponseOutputAudioDelta
from agent.state.base import VoiceAssistantEvent
from audio.player.audio_strategy import AudioStrategy
from shared.logging_mixin import LoggingMixin


class SoundEventHandler(LoggingMixin):
    """
    Handles all sound-related events from the EventBus.
    Now uses decorators for clean event handling without unused parameters.
    """

    def __init__(self, audio_strategy: AudioStrategy, event_bus: EventBus):
        self.audio_strategy = audio_strategy
        self.event_bus = event_bus

        # not instance based because decorators get resolved before init
        self.event_bus.register_handlers(self)
        self.logger.info("SoundEventHandler initialized and registered event handlers")

    @on_event_with_data(VoiceAssistantEvent.AUDIO_CHUNK_RECEIVED)
    def _handle_audio_chunk_event(self, response_output_audio_delta: ResponseOutputAudioDelta) -> None:
        """Handle AUDIO_CHUNK_RECEIVED events by adding the audio to the playback queue."""
        self.logger.debug("Received audio chunk via EventBus")
        self.audio_strategy.add_audio_chunk(response_output_audio_delta.delta)

    @on_event(VoiceAssistantEvent.WAKE_WORD_DETECTED)
    def _handle_wake_word_event(self) -> None:
        """Handle WAKE_WORD_DETECTED events by playing the wake word sound."""
        self.logger.debug("Playing wake word sound via EventBus")
        self.audio_strategy.play_wake_word_sound()

    @on_event(VoiceAssistantEvent.IDLE_TRANSITION)
    def _handle_idle_transition_event(self) -> None:
        """Handle IDLE_TRANSITION events by playing the return to idle sound."""
        self.logger.debug("Playing return to idle sound via EventBus")
        self.audio_strategy.play_return_to_idle_sound()

    @on_event(VoiceAssistantEvent.ERROR_OCCURRED)
    def _handle_error_event(self) -> None:
        """Handle ERROR_OCCURRED events by playing the error sound."""
        self.logger.debug("Playing error sound via EventBus")
        self.audio_strategy.play_error_sound()

    @on_event(VoiceAssistantEvent.USER_STARTED_SPEAKING)
    def _handle_user_started_speaking(self) -> None:
        """Handle USER_STARTED_SPEAKING events by clearing the audio queue."""
        if self.audio_strategy.is_currently_playing_chunks():
            self.event_bus.publish_sync(
                VoiceAssistantEvent.ASSISTANT_SPEECH_INTERRUPTED
            )

        self.logger.debug("User started speaking, clearing audio queue")
        self.audio_strategy.clear_queue_and_stop_chunks()
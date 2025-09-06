from agent.realtime.event_bus import EventBus
from agent.realtime.views import ResponseOutputAudioDelta
from agent.state.base import VoiceAssistantEvent
from audio.sound_player import SoundPlayer
from shared.logging_mixin import LoggingMixin


class SoundEventHandler(LoggingMixin):
    """
    Handles all sound-related events from the EventBus.
    Separated from SoundPlayer for better maintainability and separation of concerns.
    """

    def __init__(self, sound_player: SoundPlayer, event_bus: EventBus):
        self.sound_player = sound_player
        self.event_bus = event_bus
        
        self._subscribe_to_events()
        self.logger.info("SoundEventHandler initialized and subscribed to events")


    def _subscribe_to_events(self):
        """Subscribe to all sound-related events"""
        self.event_bus.subscribe(
            VoiceAssistantEvent.AUDIO_CHUNK_RECEIVED, self._handle_audio_chunk_event
        )
        self.event_bus.subscribe(
            VoiceAssistantEvent.WAKE_WORD_DETECTED, self._handle_wake_word_event
        )
        self.event_bus.subscribe(
            VoiceAssistantEvent.IDLE_TRANSITION, self._handle_idle_transition_event
        )
        self.event_bus.subscribe(
            VoiceAssistantEvent.ERROR_OCCURRED, self._handle_error_event
        )
        self.event_bus.subscribe(
            VoiceAssistantEvent.USER_STARTED_SPEAKING,
            self._handle_user_started_speaking,
        )

    def _handle_audio_chunk_event(
        self,
        event: VoiceAssistantEvent,
        response_output_audio_delta: ResponseOutputAudioDelta,
    ) -> None:
        """Handle AUDIO_CHUNK_RECEIVED events by adding the audio to the playback queue"""
        if event == VoiceAssistantEvent.AUDIO_CHUNK_RECEIVED:
            self.logger.debug("Received audio chunk via EventBus")
            self.sound_player.add_audio_chunk(response_output_audio_delta.delta)

    def _handle_wake_word_event(self, event: VoiceAssistantEvent) -> None:
        """Handle WAKE_WORD_DETECTED events by playing the wake word sound"""
        if event == VoiceAssistantEvent.WAKE_WORD_DETECTED:
            self.logger.debug("Playing wake word sound via EventBus")
            self.sound_player.play_wake_word_sound()

    def _handle_idle_transition_event(self, event: VoiceAssistantEvent) -> None:
        """Handle IDLE_TRANSITION events by playing the return to idle sound"""
        if event == VoiceAssistantEvent.IDLE_TRANSITION:
            self.logger.debug("Playing return to idle sound via EventBus")
            self.sound_player.play_return_to_idle_sound()

    def _handle_error_event(self, event: VoiceAssistantEvent) -> None:
        """Handle ERROR_OCCURRED events by playing the error sound"""
        if event == VoiceAssistantEvent.ERROR_OCCURRED:
            self.logger.debug("Playing error sound via EventBus")
            self.sound_player.play_error_sound()

    def _handle_user_started_speaking(self, event: VoiceAssistantEvent) -> None:
        """Handle USER_STARTED_SPEAKING events by clearing the audio queue"""
        if event == VoiceAssistantEvent.USER_STARTED_SPEAKING:
            if self.sound_player.is_currently_playing_chunks():
                self.event_bus.publish_sync(
                    VoiceAssistantEvent.ASSISTANT_SPEECH_INTERRUPTED
                )

            self.logger.debug("User started speaking, clearing audio queue")
            self.sound_player.clear_queue_and_stop_chunks()

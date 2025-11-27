from agent.events import EventBus
from agent.realtime.views import ResponseOutputAudioDelta
from agent.sound.player import AudioPlayer
from agent.state.base import VoiceAssistantEvent
from shared.logging_mixin import LoggingMixin


class SoundEventHandler(LoggingMixin):
    def __init__(self, audio_player: AudioPlayer, event_bus: EventBus):
        self._audio_manager = audio_player
        self.event_bus = event_bus

        self._subscribe_to_events()
        self.logger.info("SoundEventHandler initialized and subscribed to events")

    def _subscribe_to_events(self):
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
        if event == VoiceAssistantEvent.AUDIO_CHUNK_RECEIVED:
            self.logger.debug("Received audio chunk via EventBus")
            self._audio_manager.strategy.add_audio_chunk(
                response_output_audio_delta.delta
            )

    def _handle_wake_word_event(self, event: VoiceAssistantEvent) -> None:
        if event == VoiceAssistantEvent.WAKE_WORD_DETECTED:
            self.logger.debug("Playing wake word sound via EventBus")
            self._audio_manager.strategy.play_wake_word_sound()

    def _handle_idle_transition_event(self, event: VoiceAssistantEvent) -> None:
        if event == VoiceAssistantEvent.IDLE_TRANSITION:
            self.logger.debug("Playing return to idle sound via EventBus")
            self._audio_manager.strategy.play_return_to_idle_sound()

    def _handle_error_event(self, event: VoiceAssistantEvent) -> None:
        if event == VoiceAssistantEvent.ERROR_OCCURRED:
            self.logger.debug("Playing error sound via EventBus")
            self._audio_manager.strategy.play_error_sound()

    def _handle_user_started_speaking(self, event: VoiceAssistantEvent) -> None:
        if event == VoiceAssistantEvent.USER_STARTED_SPEAKING:
            if self._audio_manager.strategy.is_currently_playing_chunks():
                self.event_bus.publish_sync(
                    VoiceAssistantEvent.ASSISTANT_SPEECH_INTERRUPTED
                )

            self.logger.debug("User started speaking, clearing audio queue")
            self._audio_manager.strategy.clear_queue_and_stop_chunks()

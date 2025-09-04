from shared.logging_mixin import LoggingMixin


# TODO: This one could be responsible for sending messages like the first one to the Realtime API (also with much better typing)
class RealtimeMessageManager(LoggingMixin):

    def __init__(self):
        self.logger.info("RealtimeMessageManager initialized")

from typing import List, Literal, Optional, Union

from pydantic import BaseModel
from agent.realtime.event_types import RealtimeClientEvent


class MessageContent(BaseModel):
    """Base class for message content types."""
    type: str


class OutputTextContent(MessageContent):
    """Text content for output messages."""
    type: Literal["output_text"] = "output_text"
    text: str


class ConversationItem(BaseModel):
    """Represents a conversation item in the OpenAI Realtime API."""
    type: str
    role: str
    content: List[Union[OutputTextContent, MessageContent]]


class ConversationItemCreateEvent(BaseModel):
    """Event for creating a new conversation item."""
    type: Literal[RealtimeClientEvent.CONVERSATION_ITEM_CREATE] = (
        RealtimeClientEvent.CONVERSATION_ITEM_CREATE
    )
    item: ConversationItem


class ConversationItemFactory:
    """Factory for creating conversation items."""

    @staticmethod
    def create_assistant_message(text: str) -> ConversationItemCreateEvent:
        """
        Create a simple assistant message with text content.

        Args:
            text: The text content for the assistant message

        Returns:
            ConversationItemCreateEvent: The event to send to the API
        """
        content = [OutputTextContent(text=text)]

        item = ConversationItem(
            type="message",
            role="assistant",
            content=content
        )

        return ConversationItemCreateEvent(item=item)

    @staticmethod
    def create_user_message(text: str) -> ConversationItemCreateEvent:
        """
        Create a simple user message with text content.

        Args:
            text: The text content for the user message

        Returns:
            ConversationItemCreateEvent: The event to send to the API
        """
        content = [OutputTextContent(text=text)]

        item = ConversationItem(
            type="message",
            role="user",
            content=content
        )

        return ConversationItemCreateEvent(item=item)

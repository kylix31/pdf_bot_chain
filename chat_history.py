"""
This module provides a ChatHistory class to manage chat conversation history.
It allows adding messages with different roles (user, assistant, system) and retrieving the history in a structured format.
"""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ChatMessage:
    role: str  # Role can be 'user', 'assistant', or 'system'
    content: str


class ChatHistory:
    """
    ChatHistory class manages a chat conversation history.
    """

    def __init__(self) -> None:
        self._history: List[ChatMessage] = []

    def add_message(self, role: str, content: str) -> None:
        """
        Add a new chat message to the history.

        Parameters:
        - role: A string indicating the role, e.g., 'user', 'assistant', or 'system'.
        - content: The message content.
        """
        self._history.append(ChatMessage(role=role, content=content))

    def add_user_message(self, content: str) -> None:
        """Add a message from the user."""
        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        """Add a message from the assistant."""
        self.add_message("assistant", content)

    def add_system_message(self, content: str) -> None:
        """Add a system message."""
        self.add_message("system", content)

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Return the chat history as a list of dictionaries with 'role' and 'content' keys.
        """
        return [{"role": msg.role, "content": msg.content} for msg in self._history]

    def clear(self) -> None:
        """Clear the chat history."""
        self._history = []

    def __repr__(self) -> str:
        return "\n".join(f"{msg.role}: {msg.content}" for msg in self._history)

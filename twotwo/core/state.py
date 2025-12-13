"""TwoTwo Application State Management

Defines the various states the application and avatar can be in.
"""

from enum import Enum, auto


class AvatarState(Enum):
    """States for the avatar animation."""
    IDLE = auto()       # Default state, subtle breathing animation
    LISTENING = auto()  # Push-to-talk active, recording audio
    THINKING = auto()   # Waiting for LLM response
    SPEAKING = auto()   # TTS is playing audio


class AppState(Enum):
    """Overall application states."""
    STARTING = auto()   # Application is initializing
    READY = auto()      # Ready for interaction
    PROCESSING = auto() # Processing a request
    ERROR = auto()      # An error has occurred
    SHUTTING_DOWN = auto()  # Application is closing


class OverlayVisibility(Enum):
    """Overlay window visibility states."""
    VISIBLE = auto()
    HIDDEN = auto()
    MINIMIZED = auto()


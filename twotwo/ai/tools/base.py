"""TwoTwo Tool System - Base Tool Class

Abstract base class for all AI tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from config import get_config


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    content: str
    error: Optional[str] = None


class BaseTool(ABC):
    """Abstract base class for AI tools.
    
    Each tool must implement:
    - name: Unique identifier for the tool
    - description: Human-readable description for the AI
    - tag: XML-style tag the AI uses to invoke the tool
    - detect_intent(): Check if user input should trigger this tool
    - execute(): Run the tool and return results
    """
    
    def __init__(self):
        self._config = get_config()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this tool."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description shown to the AI in the system prompt."""
        pass
    
    @property
    @abstractmethod
    def tag(self) -> str:
        """The XML tag used to invoke this tool (e.g., 'search')."""
        pass
    
    @property
    def config_key(self) -> str:
        """Config key for enabling/disabling this tool."""
        return f"enable_{self.name}"
    
    def is_enabled(self) -> bool:
        """Check if this tool is enabled in settings."""
        return self._config.get("ai", self.config_key, default=True)
    
    def set_enabled(self, enabled: bool):
        """Enable or disable this tool."""
        self._config.set("ai", self.config_key, enabled)
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this tool is available (has required config/API keys)."""
        pass
    
    @abstractmethod
    def detect_intent(self, user_input: str) -> Optional[str]:
        """Detect if user input should trigger this tool.
        
        Args:
            user_input: The user's message
            
        Returns:
            Query to execute if this tool should be triggered, None otherwise
        """
        pass
    
    @abstractmethod
    def execute(self, query: str) -> ToolResult:
        """Execute the tool with the given query.
        
        Args:
            query: The query/input extracted from the tool tag
            
        Returns:
            ToolResult with the output
        """
        pass
    
    def get_prompt_section(self) -> str:
        """Generate the system prompt section for this tool."""
        return f"- **{self.name}**: {self.description}\n  Usage: <{self.tag}>your query</{self.tag}>"

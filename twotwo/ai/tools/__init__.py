"""TwoTwo Tool Manager

Manages AI tools - discovery, system prompt generation, and execution.
"""

import re
from typing import Optional

from ai.tools.base import BaseTool, ToolResult
from ai.tools.search_tool import SearchTool


class ToolManager:
    """Manages all AI tools.
    
    Handles tool discovery, enabling/disabling, system prompt generation,
    and routing tool calls to the appropriate handler.
    """
    
    def __init__(self):
        # Register all available tools
        self._tools: dict[str, BaseTool] = {}
        self._register_tools()
    
    def _register_tools(self):
        """Register all available tools."""
        # Add new tools here as they're created
        tools = [
            SearchTool(),
        ]
        
        for tool in tools:
            self._tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_all_tools(self) -> list[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def get_enabled_tools(self) -> list[BaseTool]:
        """Get all tools that are enabled and available."""
        return [
            tool for tool in self._tools.values()
            if tool.is_enabled() and tool.is_available()
        ]
    
    def detect_tool_intent(self, user_input: str) -> Optional[tuple[str, str]]:
        """Detect if any tool should be triggered for the user input.
        
        Iterates through all enabled tools and asks each one if it wants
        to handle this input.
        
        Args:
            user_input: The user's message
            
        Returns:
            Tuple of (tool_name, query) if a tool should be triggered, None otherwise
        """
        for tool in self.get_enabled_tools():
            query = tool.detect_intent(user_input)
            if query:
                return (tool.name, query)
        return None
    
    def get_system_prompt_addition(self) -> str:
        """Generate the tools section for the system prompt."""
        enabled = self.get_enabled_tools()
        
        if not enabled:
            return ""
        
        # Balance between clarity and brevity
        return """

You can search the web: <search>your query</search>
Use search for: weather, news, current events, things you don't know.
Include location and time context in your search queries when relevant."""
    
    def extract_tool_call(self, text: str) -> Optional[tuple[str, str]]:
        """Extract a tool call from LLM response.
        
        Args:
            text: The LLM response text
            
        Returns:
            Tuple of (tool_name, query) if found, None otherwise
        """
        for tool in self.get_enabled_tools():
            # Match <tag>content</tag> pattern
            pattern = rf'<{tool.tag}>(.+?)</{tool.tag}>'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return (tool.name, match.group(1).strip())
        
        return None
    
    def execute_tool(self, name: str, query: str) -> ToolResult:
        """Execute a tool by name.
        
        Args:
            name: Tool name
            query: Query to pass to the tool
            
        Returns:
            ToolResult from the tool
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(
                success=False,
                content="",
                error=f"Unknown tool: {name}"
            )
        
        if not tool.is_enabled():
            return ToolResult(
                success=False,
                content="",
                error=f"Tool '{name}' is disabled"
            )
        
        return tool.execute(query)
    
    def clean_tool_tags(self, text: str) -> str:
        """Remove all tool tags and fake tool syntax from text.
        
        Args:
            text: Text potentially containing tool tags
            
        Returns:
            Cleaned text with tool tags removed
        """
        # Remove fake tool syntax that models sometimes invent
        # Matches ```tool_code ... ``` or similar code blocks
        text = re.sub(r'```[\w_]*\s*[\s\S]*?```', '', text, flags=re.IGNORECASE)
        
        # Remove tool_name: and parameters: lines
        text = re.sub(r'tool_name:\s*\w+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'parameters:\s*\{[^}]*\}', '', text, flags=re.IGNORECASE)
        
        for tool in self._tools.values():
            # Remove complete tags
            text = re.sub(
                rf'<{tool.tag}>.+?</{tool.tag}>',
                '',
                text,
                flags=re.IGNORECASE | re.DOTALL
            )
            # Remove incomplete opening tags
            text = re.sub(
                rf'<{tool.tag}>.*$',
                '',
                text,
                flags=re.IGNORECASE | re.DOTALL
            )
            # Remove orphaned closing tags
            text = re.sub(
                rf'</{tool.tag}>',
                '',
                text,
                flags=re.IGNORECASE
            )
        
        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()


# Global instance
_manager: Optional[ToolManager] = None


def get_tool_manager() -> ToolManager:
    """Get the global tool manager instance."""
    global _manager
    if _manager is None:
        _manager = ToolManager()
    return _manager

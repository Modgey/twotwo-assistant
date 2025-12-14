"""TwoTwo Tool - Web Search

Brave Search API integration for real-time web information.
"""

import requests
from dataclasses import dataclass
from typing import Optional

from config import get_config
from ai.tools.base import BaseTool, ToolResult


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    description: str


class SearchTool(BaseTool):
    """Web search tool using Brave Search API."""
    
    API_URL = "https://api.search.brave.com/res/v1/web/search"
    
    # Keywords that trigger automatic search
    TRIGGER_KEYWORDS = {
        # Weather
        'weather', 'temperature', 'forecast', 'rain', 'sunny', 'cloudy',
        # Time
        'time', 'clock', "what's the time", 'current time',
        # News
        'news', 'headline', 'breaking',
        # Sports
        'score', 'game', 'match', 'standings',
        # Stocks
        'stock', 'price', 'market', 'trading',
        # General current info
        'today', 'right now', 'currently', 'latest',
    }
    
    def __init__(self):
        super().__init__()
        self._api_key = self._config.get("ai", "brave_api_key", default="")
    
    @property
    def name(self) -> str:
        return "search"
    
    @property
    def description(self) -> str:
        return "Search the web for current information like weather, news, time, sports scores, or facts you don't know"
    
    @property
    def tag(self) -> str:
        return "search"
    
    def is_available(self) -> bool:
        """Check if Brave API key is configured."""
        return bool(self._api_key)
    
    def set_api_key(self, key: str):
        """Set the Brave API key."""
        self._api_key = key
        self._config.set("ai", "brave_api_key", key)
    
    def detect_intent(self, user_input: str) -> Optional[str]:
        """Detect if user input should trigger a web search.
        
        Args:
            user_input: The user's message
            
        Returns:
            The search query if search should be triggered, None otherwise
        """
        if not self.is_enabled() or not self.is_available():
            return None
        
        input_lower = user_input.lower()
        
        # Check for trigger keywords
        for keyword in self.TRIGGER_KEYWORDS:
            if keyword in input_lower:
                # Use the full user input as the search query
                return user_input
        
        return None
    
    def execute(self, query: str) -> ToolResult:
        """Search the web and return formatted results."""
        if not self._api_key:
            return ToolResult(
                success=False,
                content="",
                error="No Brave API key configured"
            )
        
        try:
            response = requests.get(
                self.API_URL,
                headers={"X-Subscription-Token": self._api_key},
                params={"q": query, "count": 5},
                timeout=10,
            )
            
            if response.status_code != 200:
                return ToolResult(
                    success=False,
                    content="",
                    error=f"Search API returned {response.status_code}"
                )
            
            data = response.json()
            results = []
            
            for item in data.get("web", {}).get("results", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    description=item.get("description", ""),
                ))
            
            # Format results for the AI
            formatted = self._format_results(results, query)
            return ToolResult(success=True, content=formatted)
            
        except requests.RequestException as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Search request failed: {e}"
            )
    
    def _format_results(self, results: list[SearchResult], query: str) -> str:
        """Format search results as context for the AI."""
        if not results:
            return f"No results found for '{query}'."
        
        lines = [f"Search results for '{query}':"]
        lines.append("")
        
        for r in results:
            lines.append(f"• {r.title}")
            if r.description:
                lines.append(f"  {r.description}")
            lines.append("")
        
        return "\n".join(lines)

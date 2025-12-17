"""TwoTwo Tool - Web Search

Brave Search API integration for real-time web information.
"""

import requests
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

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
        self._lock = threading.Lock()
        self._active_searches: Dict[str, threading.Event] = {}
        self._search_results: Dict[str, ToolResult] = {}
        self._last_query: Optional[str] = None
        self._last_result: Optional[ToolResult] = None
        self._last_time: float = 0

    
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
        """Search the web and return formatted results.
        
        Includes de-duplication to prevent hitting Rate Limits (429) 
        when parallel systems trigger the same search.
        """
        if not self._api_key:
            return ToolResult(
                success=False,
                content="",
                error="No Brave API key configured"
            )
        
        # Normalize query for better de-duplication
        query_norm = query.strip().lower()
        
        # 1. Check recent cache (last 10 seconds)
        with self._lock:
            if query_norm == self._last_query and (time.time() - self._last_time) < 10:
                print(f"[Search] Using recent cache for: {query_norm[:30]}...")
                return self._last_result
            
            # 2. Check if this exact search is currently running in another thread
            if query_norm in self._active_searches:
                print(f"[Search] Waiting for in-progress search: {query_norm[:30]}...")
                event = self._active_searches[query_norm]
            else:
                event = threading.Event()
                self._active_searches[query_norm] = event

        # If we didn't create the event, it's already running. Wait for it.
        if query_norm in self._active_searches and self._active_searches[query_norm] != event:
            event.wait(timeout=15)
            with self._lock:
                return self._search_results.get(query_norm, ToolResult(False, "", "Search timed out"))

        # We are the primary search runner
        try:
            response = requests.get(
                self.API_URL,
                headers={"X-Subscription-Token": self._api_key},
                params={"q": query, "count": 5},
                timeout=10,
            )
            
            if response.status_code != 200:
                result = ToolResult(
                    success=False,
                    content="",
                    error=f"Search API returned {response.status_code}"
                )
            else:
                data = response.json()
                results = []
                
                for item in data.get("web", {}).get("results", []):
                    results.append(SearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        description=item.get("description", ""),
                    ))
                
                formatted = self._format_results(results, query)
                result = ToolResult(success=True, content=formatted)
                
            # Update cache and notify waiting threads
            with self._lock:
                self._search_results[query_norm] = result
                self._last_query = query_norm
                self._last_result = result
                self._last_time = time.time()
                if query_norm in self._active_searches:
                    self._active_searches[query_norm].set()
                    del self._active_searches[query_norm]
            
            return result
            
        except requests.RequestException as e:
            result = ToolResult(
                success=False,
                content="",
                error=f"Search request failed: {e}"
            )
            with self._lock:
                if query_norm in self._active_searches:
                    self._active_searches[query_norm].set()
                    del self._active_searches[query_norm]
            return result
    
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

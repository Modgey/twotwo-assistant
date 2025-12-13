"""TwoTwo Web Search

Brave Search API integration for web search capabilities.
"""

import re
from dataclasses import dataclass
from typing import Optional
import requests

from config import get_config


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    description: str


class BraveSearch:
    """Brave Search API client."""
    
    API_URL = "https://api.search.brave.com/res/v1/web/search"
    
    def __init__(self, api_key: Optional[str] = None):
        self._config = get_config()
        self.api_key = api_key or self._config.get("ai", "brave_api_key", default="")
    
    def is_available(self) -> bool:
        """Check if search is available (has API key)."""
        return bool(self.api_key)
    
    def set_api_key(self, key: str):
        """Set the API key."""
        self.api_key = key
        self._config.set("ai", "brave_api_key", key)
    
    def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        """Perform a web search.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        if not self.api_key:
            return []
        
        try:
            response = requests.get(
                self.API_URL,
                headers={"X-Subscription-Token": self.api_key},
                params={"q": query, "count": num_results},
                timeout=10,
            )
            
            if response.status_code != 200:
                print(f"Search error: {response.status_code}")
                return []
            
            data = response.json()
            results = []
            
            for item in data.get("web", {}).get("results", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    description=item.get("description", ""),
                ))
            
            return results
            
        except requests.RequestException as e:
            print(f"Search error: {e}")
            return []
    
    def format_results(self, results: list[SearchResult]) -> str:
        """Format search results as text for LLM context.
        
        Args:
            results: List of SearchResult objects
            
        Returns:
            Formatted string
        """
        if not results:
            return "No search results found."
        
        lines = ["Search Results:"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n{i}. {r.title}")
            lines.append(f"   URL: {r.url}")
            lines.append(f"   {r.description}")
        
        return "\n".join(lines)


class SearchHandler:
    """Handles autonomous search decisions in LLM responses."""
    
    # Pattern to detect search requests from LLM
    SEARCH_PATTERN = re.compile(r'\[SEARCH:\s*(.+?)\]', re.IGNORECASE)
    
    def __init__(self, search_client: Optional[BraveSearch] = None):
        self._config = get_config()
        self.search = search_client or BraveSearch()
        self.enabled = self._config.get("ai", "enable_search", default=True)
    
    def is_enabled(self) -> bool:
        """Check if search is enabled and available."""
        return self.enabled and self.search.is_available()
    
    def set_enabled(self, enabled: bool):
        """Enable or disable search."""
        self.enabled = enabled
        self._config.set("ai", "enable_search", enabled)
    
    def get_search_prompt(self) -> str:
        """Get the system prompt addition for search capability."""
        if not self.is_enabled():
            return ""
        
        return """

You have access to web search for current information. When you need up-to-date information or facts you're uncertain about, respond with:
[SEARCH: your search query]

Wait for the search results, then incorporate them into your response. Use search sparingly - only when truly needed for current events, recent data, or facts you're unsure about."""
    
    def extract_search_query(self, text: str) -> Optional[str]:
        """Extract a search query from LLM response if present.
        
        Args:
            text: LLM response text
            
        Returns:
            Search query if found, None otherwise
        """
        match = self.SEARCH_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        return None
    
    def handle_search(self, query: str) -> str:
        """Perform search and return formatted results.
        
        Args:
            query: Search query
            
        Returns:
            Formatted results string
        """
        results = self.search.search(query)
        return self.search.format_results(results)
    
    def process_response_with_search(
        self,
        response: str,
        llm_callback: callable,
    ) -> str:
        """Process an LLM response, handling any search requests.
        
        This is for non-streaming responses where we can detect search
        requests and make additional LLM calls.
        
        Args:
            response: Initial LLM response
            llm_callback: Function to call LLM with follow-up context
            
        Returns:
            Final response after any search augmentation
        """
        if not self.is_enabled():
            return response
        
        # Check for search request
        query = self.extract_search_query(response)
        if not query:
            return response
        
        # Perform search
        print(f"Performing search: {query}")
        search_results = self.handle_search(query)
        
        # Get follow-up response with search results
        follow_up = f"Here are the search results for '{query}':\n\n{search_results}\n\nNow please provide your response based on this information."
        
        return llm_callback(follow_up)


# Global instances
_search: BraveSearch | None = None
_handler: SearchHandler | None = None


def get_search() -> BraveSearch:
    """Get the global search client."""
    global _search
    if _search is None:
        _search = BraveSearch()
    return _search


def get_search_handler() -> SearchHandler:
    """Get the global search handler."""
    global _handler
    if _handler is None:
        _handler = SearchHandler()
    return _handler


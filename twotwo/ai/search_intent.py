"""Search Intent Detector

Uses a tiny, fast LLM to classify whether a query needs real-time search.
This avoids hardcoded keyword matching and provides semantic understanding.
"""

import requests
from typing import Optional


class SearchIntentDetector:
    """Detects if a query needs real-time web search using a small classifier model."""
    
    # Small, fast models for classification (in order of preference)
    CLASSIFIER_MODELS = [
        "qwen2:0.5b",
        "qwen2:1.5b",  
        "tinyllama",
        "gemma2:2b",
    ]
    
    CLASSIFICATION_PROMPT = """You are a search intent classifier. Your ONLY job is to decide if a query needs a real-time web search.

Respond with ONLY "SEARCH" or "NO_SEARCH" - nothing else.

SEARCH if the query asks about:
- Current/live sports scores, games, or schedules
- Today's weather or forecasts
- Breaking news or recent events
- Stock prices or market data  
- What time it is in a location
- Current status of anything (is X open, is X happening)
- Any fact the user expects to be up-to-date

NO_SEARCH if the query is:
- General knowledge (history, science, definitions)
- Casual conversation (hello, how are you, thanks)
- Personal questions (what do you think, your opinion)
- Requests that don't need current data

Query: "{query}"

Response (SEARCH or NO_SEARCH):"""

    def __init__(self, host: str = "http://127.0.0.1:11434"):
        self.host = host
        self._classifier_model: Optional[str] = None
        self._session = requests.Session()
        self._find_classifier_model()
    
    def _find_classifier_model(self):
        """Find an available tiny model for classification."""
        try:
            response = self._session.get(f"{self.host}/api/tags", timeout=2)
            if response.status_code != 200:
                return
            
            available = [m["name"] for m in response.json().get("models", [])]
            
            # Find first available classifier model
            for model in self.CLASSIFIER_MODELS:
                if model in available:
                    self._classifier_model = model
                    print(f"Search intent detector using: {model}")
                    return
            
            # No tiny model found - will fall back to main model or skip
            print("No small classifier model found. Search intent detection disabled.")
            print(f"Install one with: ollama pull qwen2:0.5b")
            
        except requests.RequestException:
            pass
    
    def needs_search(self, query: str, timeout: float = 1.5) -> bool:
        """Determine if the query needs a real-time web search.
        
        Args:
            query: The user's query
            timeout: Max time to wait for classification (default 1.5s)
            
        Returns:
            True if search is recommended, False otherwise
        """
        if not self._classifier_model:
            # No classifier available - return False (don't search by default)
            return False
        
        try:
            prompt = self.CLASSIFICATION_PROMPT.format(query=query)
            
            response = self._session.post(
                f"{self.host}/api/generate",
                json={
                    "model": self._classifier_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 10,  # We only need one word
                        "temperature": 0,   # Deterministic
                    }
                },
                timeout=timeout,
            )
            
            if response.status_code != 200:
                return False
            
            result = response.json().get("response", "").strip().upper()
            
            # Check if response indicates search needed
            needs = "SEARCH" in result and "NO_SEARCH" not in result
            
            if needs:
                print(f"[Intent] Query needs search: '{query[:50]}...'")
            
            return needs
            
        except requests.RequestException:
            # Timeout or error - don't block, just skip search
            return False
    
    def is_available(self) -> bool:
        """Check if the intent detector is available."""
        return self._classifier_model is not None


# Global instance
_detector: Optional[SearchIntentDetector] = None


def get_search_intent_detector() -> SearchIntentDetector:
    """Get the global search intent detector instance."""
    global _detector
    if _detector is None:
        _detector = SearchIntentDetector()
    return _detector

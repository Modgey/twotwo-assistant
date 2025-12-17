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
    
    # Cloud classifier model (fast & cheap)
    CLOUD_CLASSIFIER = "mistralai/voxtral-small-24b-2507"

    
    CLASSIFICATION_PROMPT = """
    <task>
    Classify if query needs REAL-TIME WEB SEARCH.
    Output ONLY "SEARCH" or "NO_SEARCH".
    </task>

    <rules>
    1. SEARCH ONLY if query asks for:
       - Current events/news (politics, sports, entertainment)
       - Live data (weather, stocks, time)
       - Dynamic specific facts (release dates, scores)

    2. NO_SEARCH if query is:
       - Conversational ("What's up", "How are you", "Hello")
       - General knowledge/Facts (History, Science, Math)
       - Opinions/Creative ("Tell me a joke", "Write a poem")
       - Clarifications/Follow-ups ("What do you mean?", "Explain that")

    3. CRITICAL: "What's up" and "What's going on" are GREETINGS, not search queries.
    </rules>

    Query: "{query}"
    Classification:
    """

    def __init__(self, backend: str = "ollama", host: str = "http://127.0.0.1:11434", api_key: str = ""):
        self.backend = backend
        self.host = host
        self.api_key = api_key
        self._classifier_model: Optional[str] = None
        self._session = requests.Session()
        
        if self.backend == "ollama":
            self._find_classifier_model()
        else:
            self._classifier_model = self.CLOUD_CLASSIFIER
            self._session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })
            print(f"Search intent detector using cloud backend: {self._classifier_model}")

    
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
            
            if self.backend == "ollama":
                url = f"{self.host}/api/generate"
                payload = {
                    "model": self._classifier_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 10,
                        "temperature": 0,
                    },
                    "context": [],
                }
            else:
                url = "https://openrouter.ai/api/v1/chat/completions"
                payload = {
                    "model": self._classifier_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 10,
                    "temperature": 0,
                }
            
            response = self._session.post(
                url,
                json=payload,
                timeout=timeout,
            )
            
            if response.status_code != 200:
                print(f"[Intent] Backend error: {response.status_code}")
                return False
            
            data = response.json()
            if self.backend == "ollama":
                result = data.get("response", "").strip().upper()
            else:
                result = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
            
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


def get_search_intent_detector(backend: str = "ollama", **kwargs) -> SearchIntentDetector:
    """Get or create the search intent detector instance."""
    global _detector
    
    # If backend changed, we must re-initialize
    if _detector is not None and _detector.backend != backend:
        _detector = None
        
    if _detector is None:
        _detector = SearchIntentDetector(
            backend=backend,
            host=kwargs.get("host", "http://127.0.0.1:11434"),
            api_key=kwargs.get("api_key", "")
        )
    return _detector

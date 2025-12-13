"""TwoTwo LLM Interface

Ollama integration with streaming support for local AI inference.
"""

import json
import threading
from typing import Callable, Generator, Optional
import requests

from config import get_config


class OllamaLLM:
    """Interface to Ollama for local LLM inference."""
    
    DEFAULT_HOST = "http://localhost:11434"
    
    def __init__(
        self,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        host: Optional[str] = None,
    ):
        self._config = get_config()
        
        # Get settings from config or use defaults
        self.model = model or self._config.get("ai", "model", default="llama3.2:3b")
        self.system_prompt = system_prompt or self._config.get(
            "ai", "personality",
            default="You are TwoTwo, a helpful AI assistant. Be concise and direct."
        )
        self.host = host or self.DEFAULT_HOST
        
        # Conversation history
        self._messages: list[dict] = []
        
        # Connection status
        self._available = False
        self._check_connection()
    
    def _check_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=2)
            self._available = response.status_code == 200
        except requests.RequestException:
            self._available = False
        return self._available
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        return self._available
    
    def warm_up(self):
        """Pre-warm the model to reduce first-response latency.
        
        Sends a minimal request to load the model into memory.
        Call this during startup or periodically to keep model hot.
        """
        if not self._available:
            return
        
        def _warm():
            try:
                # Send minimal request to load model
                requests.post(
                    f"{self.host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": "hi",
                        "stream": False,
                        "options": {"num_predict": 1}  # Generate just 1 token
                    },
                    timeout=30,
                )
                print(f"LLM warmed up: {self.model}")
            except Exception as e:
                print(f"LLM warm-up failed: {e}")
        
        # Run in background to not block
        threading.Thread(target=_warm, daemon=True).start()
    
    def set_model(self, model: str):
        """Change the active model."""
        self.model = model
        self._config.set("ai", "model", model)
    
    def set_system_prompt(self, prompt: str):
        """Update the system prompt."""
        self.system_prompt = prompt
        self._config.set("ai", "personality", prompt)
    
    def clear_history(self):
        """Clear conversation history."""
        self._messages = []
    
    def chat(self, user_message: str) -> str:
        """Send a message and get a complete response (blocking).
        
        Args:
            user_message: The user's message
            
        Returns:
            Complete response text
        """
        if not self._available:
            if not self._check_connection():
                return "Error: Ollama is not running. Please start Ollama first."
        
        # Build messages with system prompt
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self._messages)
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                },
                timeout=60,
            )
            
            if response.status_code != 200:
                return f"Error: Ollama returned status {response.status_code}"
            
            data = response.json()
            assistant_message = data.get("message", {}).get("content", "")
            
            # Store in history
            self._messages.append({"role": "user", "content": user_message})
            self._messages.append({"role": "assistant", "content": assistant_message})
            
            # Trim history if too long (keep last 20 messages)
            if len(self._messages) > 20:
                self._messages = self._messages[-20:]
            
            return assistant_message
            
        except requests.Timeout:
            return "Error: Request timed out. The model may be loading."
        except requests.RequestException as e:
            self._available = False
            return f"Error: Connection failed - {e}"
    
    def chat_stream(
        self,
        user_message: str,
        on_token: Callable[[str], None],
        on_complete: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> threading.Thread:
        """Send a message and stream the response.
        
        Args:
            user_message: The user's message
            on_token: Called with each token as it arrives
            on_complete: Called with full response when done
            on_error: Called if an error occurs
            
        Returns:
            Thread running the request
        """
        def worker():
            if not self._available:
                if not self._check_connection():
                    error_msg = "Ollama is not running. Please start Ollama first."
                    if on_error:
                        on_error(error_msg)
                    return
            
            # Build messages with system prompt
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self._messages)
            messages.append({"role": "user", "content": user_message})
            
            full_response = ""
            
            try:
                response = requests.post(
                    f"{self.host}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": True,
                    },
                    stream=True,
                    timeout=120,
                )
                
                if response.status_code != 200:
                    if on_error:
                        on_error(f"Ollama returned status {response.status_code}")
                    return
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            full_response += content
                            on_token(content)
                        
                        # Check if done
                        if data.get("done", False):
                            break
                            
                    except json.JSONDecodeError:
                        continue
                
                # Store in history
                self._messages.append({"role": "user", "content": user_message})
                self._messages.append({"role": "assistant", "content": full_response})
                
                # Trim history
                if len(self._messages) > 20:
                    self._messages = self._messages[-20:]
                
                if on_complete:
                    on_complete(full_response)
                    
            except requests.Timeout:
                if on_error:
                    on_error("Request timed out. The model may be loading.")
            except requests.RequestException as e:
                self._available = False
                if on_error:
                    on_error(f"Connection failed - {e}")
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return thread
    
    def generate_stream(self, user_message: str) -> Generator[str, None, None]:
        """Generator that yields tokens as they arrive.
        
        Args:
            user_message: The user's message
            
        Yields:
            Tokens as they arrive
        """
        if not self._available:
            if not self._check_connection():
                yield "Error: Ollama is not running."
                return
        
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self._messages)
        messages.append({"role": "user", "content": user_message})
        
        full_response = ""
        
        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                },
                stream=True,
                timeout=120,
            )
            
            if response.status_code != 200:
                yield f"Error: Status {response.status_code}"
                return
            
            for line in response.iter_lines():
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        full_response += content
                        yield content
                    
                    if data.get("done", False):
                        break
                        
                except json.JSONDecodeError:
                    continue
            
            # Store in history
            self._messages.append({"role": "user", "content": user_message})
            self._messages.append({"role": "assistant", "content": full_response})
            
            if len(self._messages) > 20:
                self._messages = self._messages[-20:]
                
        except requests.RequestException as e:
            self._available = False
            yield f"Error: {e}"


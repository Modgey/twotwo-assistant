"""TwoTwo LLM Interface

Ollama integration with streaming support for local AI inference.
"""

import json
import threading
import codecs
from datetime import datetime
from typing import Callable, Generator, Optional
import requests

from config import get_config


class OllamaLLM:
    """Interface to Ollama for local LLM inference."""
    
    # Use 127.0.0.1 instead of localhost to avoid Windows DNS delay!
    DEFAULT_HOST = "http://127.0.0.1:11434"
    
    def __init__(
        self,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        host: Optional[str] = None,
    ):
        self._config = get_config()
        
        # Get settings from config or use defaults
        self.model = model or self._config.get("ai", "model", default="gemma3:4b")
        self.system_prompt = system_prompt or self._config.get(
            "ai", "personality",
            default="You are TwoTwo, a helpful AI assistant. Be concise and direct."
        )
        self.host = host or self.DEFAULT_HOST
        
        # Use a session for connection pooling (reuse TCP connections)
        self._session = requests.Session()
        
        # Conversation history
        self._messages: list[dict] = []
        
        # Connection status
        self._available = False
        self._check_connection()
        
        # Keep-alive thread to prevent model unloading
        self._keep_alive_running = False
        self._keep_alive_thread: Optional[threading.Thread] = None
        self._start_keep_alive()
    
    def _check_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = self._session.get(f"{self.host}/api/tags", timeout=2)
            self._available = response.status_code == 200
        except requests.RequestException:
            self._available = False
        return self._available
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        return self._available
    
    def _start_keep_alive(self):
        """Start background thread to keep model loaded."""
        if not self._available:
            return
        
        self._keep_alive_running = True
        
        def _keep_alive_loop():
            import time
            while self._keep_alive_running:
                try:
                    # Send minimal request every 20 seconds to keep model hot
                    self._session.post(
                        f"{self.host}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": ".",
                            "stream": False,
                            "keep_alive": "5m",  # Keep loaded for 5 minutes
                            "options": {"num_predict": 1}
                        },
                        timeout=10,
                    )
                except Exception:
                    pass  # Silently fail - model might not be available
                
                # Sleep 20 seconds before next ping
                for _ in range(20):
                    if not self._keep_alive_running:
                        break
                    time.sleep(1)
        
        self._keep_alive_thread = threading.Thread(target=_keep_alive_loop, daemon=True)
        self._keep_alive_thread.start()
        print(f"Keep-alive thread started for {self.model}")
    
    def warm_up(self):
        """Pre-warm the model to reduce first-response latency.
        
        Sends a minimal request to load the model into memory.
        Call this during startup or periodically to keep model hot.
        """
        if not self._available:
            return
        
        def _warm():
            try:
                # Send minimal request to load model and keep it loaded
                self._session.post(
                    f"{self.host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": "hi",
                        "stream": False,
                        "keep_alive": "30m",  # Keep model loaded for 30 minutes
                        "options": {"num_predict": 1}  # Generate just 1 token
                    },
                    timeout=30,
                )
                print(f"LLM warmed up: {self.model} (keep_alive: 30m)")
            except Exception as e:
                print(f"LLM warm-up failed: {e}")
        
        # Run in background to not block
        threading.Thread(target=_warm, daemon=True).start()
    
    def set_model(self, model: str):
        """Change the active model."""
        # Stop old keep-alive thread
        self._keep_alive_running = False
        
        self.model = model
        self._config.set("ai", "model", model)
        
        # Restart keep-alive with new model
        self._start_keep_alive()
    
    def set_system_prompt(self, prompt: str):
        """Update the system prompt."""
        self.system_prompt = prompt
        self._config.set("ai", "personality", prompt)
    
    def clear_history(self):
        """Clear conversation history."""
        self._messages = []
    
    def get_full_system_prompt(self) -> str:
        """Load and fill the system prompt template.
        
        Edit ai/system_prompt.txt to change the prompt structure.
        Placeholders: {{location}}, {{date}}, {{time}}, {{personality}}, {{tools}}
        """
        from pathlib import Path
        
        # Load template
        template_path = Path(__file__).parent / "system_prompt.txt"
        try:
            template = template_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            # Fallback if template missing
            template = "{{personality}}\n\n{{tools}}"
        
        # Get current date/time
        now = datetime.now()
        
        # Get location (cached)
        if not hasattr(self, '_cached_location'):
            self._cached_location = self._get_location()
        
        # Get tools section (will be filled in by controller)
        tools_section = getattr(self, '_tools_prompt', '')
        
        # Replace placeholders
        prompt = template.replace("{{location}}", self._cached_location or "Unknown")
        prompt = prompt.replace("{{date}}", now.strftime("%A, %B %d, %Y"))
        prompt = prompt.replace("{{time}}", now.strftime("%I:%M %p"))
        prompt = prompt.replace("{{personality}}", self.system_prompt)
        prompt = prompt.replace("{{tools}}", tools_section)
        
        return prompt
    
    def _get_location(self) -> str:
        """Get location from config or infer from timezone (cached)."""
        import time as time_module
        
        location = self._config.get("user", "location", default="")
        if location:
            return location
        
        try:
            tz_name = time_module.tzname[0] if time_module.tzname else ""
            if "Standard Time" in tz_name or "Daylight Time" in tz_name:
                return tz_name.replace(" Standard Time", "").replace(" Daylight Time", "")
            tz_map = {"IST": "Israel", "EST": "Eastern US", "PST": "Pacific US", "GMT": "UK"}
            return tz_map.get(tz_name, "")
        except:
            return ""
    
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
        
        # Build messages with system prompt (includes current date/time)
        messages = [{"role": "system", "content": self.get_full_system_prompt()}]
        messages.extend(self._messages)
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self._session.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "keep_alive": "30m",
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
            
            # Build messages with system prompt (includes current date/time)
            # For speed, limit history to last 4 messages (2 exchanges)
            recent_history = self._messages[-4:] if len(self._messages) > 4 else self._messages
            
            messages = [{"role": "system", "content": self.get_full_system_prompt()}]
            messages.extend(recent_history)
            messages.append({"role": "user", "content": user_message})
            
            full_response = ""
            
            try:
                response = self._session.post(
                    f"{self.host}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": True,
                        "keep_alive": "30m",
                    },
                    stream=True,
                    timeout=120,
                )
                
                if response.status_code != 200:
                    if on_error:
                        on_error(f"Ollama returned status {response.status_code}")
                    return
                
                # Manual line buffering for instant streaming (no extensive buffering)
                buffer = ""
                decoder = codecs.getincrementaldecoder("utf-8")(errors='replace')
                
                for chunk in response.iter_content(chunk_size=None):
                    if not chunk:
                        continue
                        
                    # Decode chunk safely (handling split multi-byte chars)
                    text = decoder.decode(chunk, final=False)
                    buffer += text
                    
                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
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
            response = self._session.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "keep_alive": "30m",
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
                
            if len(self._messages) > 20:
                self._messages = self._messages[-20:]
                
        except requests.RequestException as e:
            self._available = False
            yield f"Error: {e}"


class OpenRouterLLM(OllamaLLM):
    """Interface to OpenRouter for cloud LLM inference."""
    
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    def __init__(
        self,
        model: str,
        api_key: str,
        system_prompt: Optional[str] = None
    ):
        # Initialize without calling super().__init__ completely since we struggle with host/checking
        self._config = get_config()
        self.model = model
        self.api_key = api_key
        self.system_prompt = system_prompt or self._config.get(
            "ai", "personality",
            default="You are TwoTwo, a helpful AI assistant."
        )
        
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/modgey/twotwo-assistant",
            "X-Title": "TwoTwo Assistant",
            "Content-Type": "application/json"
        })
        
        self._messages: list[dict] = []
        self._available = True  # Assume available if key provided
        
        # No keep-alive needed for cloud
        self._keep_alive_running = False
        self._keep_alive_thread = None

    def _check_connection(self) -> bool:
        """Check if API key works."""
        return True  # Skip actual check to avoid latency, handle errors in requests
        
    def _start_keep_alive(self):
        pass  # Not needed
        
    def warm_up(self):
        pass  # Not needed
    
    def set_model(self, model: str):
        """Change the active model."""
        self.model = model
        self._config.set("ai", "openrouter_model", model)
        
    def chat_stream(
        self,
        user_message: str,
        on_token: Callable[[str], None],
        on_complete: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> threading.Thread:
        """Stream response from OpenRouter."""
        def worker():
            if not self.api_key:
                if on_error: on_error("Error: OpenRouter API key not configured")
                return
            
            # Prepare messages
            recent_history = self._messages[-4:] if len(self._messages) > 4 else self._messages
            messages = [{"role": "system", "content": self.get_full_system_prompt()}]
            messages.extend(recent_history)
            messages.append({"role": "user", "content": user_message})
            
            full_response = ""
            
            try:
                response = self._session.post(
                    self.API_URL,
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": True,
                    },
                    stream=True,
                    timeout=120
                )
                
                if response.status_code != 200:
                    try:
                        err_msg = response.json().get('error', {}).get('message', str(response.status_code))
                    except:
                        err_msg = str(response.status_code)
                    if on_error: on_error(f"OpenRouter Error: {err_msg}")
                    return
                
                # Manual line buffering for instant streaming
                buffer = ""
                decoder = codecs.getincrementaldecoder("utf-8")(errors='replace')
                
                for chunk in response.iter_content(chunk_size=None):
                    if not chunk: continue
                    
                    # Decode chunk safely
                    text = decoder.decode(chunk, final=False)
                    buffer += text
                    
                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        
                        if not line: continue
                        
                        if line.startswith('data: '):
                            line = line[6:]
                            
                        if line == '[DONE]':
                            break
                            
                        try:
                            data = json.loads(line)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            
                            if content:
                                full_response += content
                                on_token(content)
                        except json.JSONDecodeError:
                            continue
                
                # Update history
                self._messages.append({"role": "user", "content": user_message})
                self._messages.append({"role": "assistant", "content": full_response})
                if len(self._messages) > 20:
                    self._messages = self._messages[-20:]
                
                if on_complete:
                    on_complete(full_response)
                    
            except Exception as e:
                if on_error: on_error(f"Network Error: {str(e)}")
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return thread


def get_llm(backend: str = "ollama", **kwargs):
    """Factory to get the appropriate LLM backend."""
    if backend == "openrouter":
        return OpenRouterLLM(
            model=kwargs.get("model", "google/gemini-2.0-flash-exp:free"),
            api_key=kwargs.get("api_key", ""),
            system_prompt=kwargs.get("system_prompt")
        )
    else:
        return OllamaLLM(
            model=kwargs.get("model"),
            system_prompt=kwargs.get("system_prompt"),
            host=kwargs.get("host")
        )



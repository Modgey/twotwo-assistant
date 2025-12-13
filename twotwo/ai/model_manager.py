"""TwoTwo Model Manager

Auto-detection and management of Ollama models.
Supports auto-starting Ollama if not running.
"""

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import requests


@dataclass
class ModelInfo:
    """Information about an available model."""
    name: str
    size: str  # Human-readable size
    modified: str  # Last modified date
    family: str  # Model family (llama, gemma, etc.)
    parameter_size: str  # e.g., "3B", "7B"
    quantization: str  # e.g., "Q4_0", "Q8_0"


class ModelManager:
    """Manages discovery and selection of LLM models."""
    
    OLLAMA_HOST = "http://localhost:11434"
    
    # Recommended models for TwoTwo (small, fast, good quality)
    RECOMMENDED_MODELS = [
        "gemma3:4b",
        "gemma3:12b", 
        "llama3.2:3b",
        "llama3.2:1b",
        "phi3:mini",
        "qwen2:1.5b",
    ]
    
    def __init__(self, host: Optional[str] = None):
        self.host = host or self.OLLAMA_HOST
        self._models_cache: list[ModelInfo] = []
        self._last_check = 0
        self._ollama_path: Optional[Path] = None
        self._ollama_process = None
    
    def _find_ollama(self) -> Optional[Path]:
        """Find the Ollama executable."""
        if self._ollama_path and self._ollama_path.exists():
            return self._ollama_path
        
        # Check common Windows locations
        possible_paths = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
            Path(os.environ.get("PROGRAMFILES", "")) / "Ollama" / "ollama.exe",
            Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Ollama" / "ollama.exe",
            Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
        ]
        
        for path in possible_paths:
            if path.exists():
                self._ollama_path = path
                return path
        
        # Try to find in PATH
        ollama_in_path = shutil.which("ollama")
        if ollama_in_path:
            self._ollama_path = Path(ollama_in_path)
            return self._ollama_path
        
        return None
    
    def is_ollama_installed(self) -> bool:
        """Check if Ollama is installed on this system."""
        return self._find_ollama() is not None
    
    def is_ollama_running(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def start_ollama(self, wait_timeout: float = 10.0) -> bool:
        """Start the Ollama server if not running.
        
        Args:
            wait_timeout: Seconds to wait for server to start
            
        Returns:
            True if server is running (started or already running)
        """
        # Already running?
        if self.is_ollama_running():
            print("Ollama already running")
            return True
        
        # Find executable
        ollama_path = self._find_ollama()
        if not ollama_path:
            print("Ollama not found. Please install from https://ollama.ai")
            return False
        
        try:
            # Start Ollama serve in background
            print(f"Starting Ollama from {ollama_path}...")
            
            # On Windows, use CREATE_NO_WINDOW to hide console
            startupinfo = None
            creationflags = 0
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                creationflags = subprocess.CREATE_NO_WINDOW
            
            self._ollama_process = subprocess.Popen(
                [str(ollama_path), "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                startupinfo=startupinfo,
                creationflags=creationflags,
            )
            
            # Wait for server to be ready
            start_time = time.time()
            while time.time() - start_time < wait_timeout:
                if self.is_ollama_running():
                    print("Ollama started successfully")
                    return True
                time.sleep(0.5)
            
            print("Ollama start timeout - server not responding")
            return False
            
        except Exception as e:
            print(f"Failed to start Ollama: {e}")
            return False
    
    def ensure_running(self, auto_start: bool = True) -> bool:
        """Ensure Ollama is running, optionally auto-starting it.
        
        Args:
            auto_start: If True, start Ollama if not running
            
        Returns:
            True if Ollama is running
        """
        if self.is_ollama_running():
            return True
        
        if auto_start:
            return self.start_ollama()
        
        return False
    
    def get_available_models(self, refresh: bool = False) -> list[ModelInfo]:
        """Get list of models available in Ollama.
        
        Args:
            refresh: Force refresh from server
            
        Returns:
            List of ModelInfo objects
        """
        if self._models_cache and not refresh:
            return self._models_cache
        
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code != 200:
                return []
            
            data = response.json()
            models = []
            
            for model_data in data.get("models", []):
                name = model_data.get("name", "")
                size_bytes = model_data.get("size", 0)
                modified = model_data.get("modified_at", "")
                
                # Parse model details from name and metadata
                details = model_data.get("details", {})
                family = details.get("family", self._infer_family(name))
                param_size = details.get("parameter_size", self._infer_param_size(name))
                quantization = details.get("quantization_level", "")
                
                # Format size
                if size_bytes > 1e9:
                    size_str = f"{size_bytes / 1e9:.1f} GB"
                elif size_bytes > 1e6:
                    size_str = f"{size_bytes / 1e6:.1f} MB"
                else:
                    size_str = f"{size_bytes} bytes"
                
                models.append(ModelInfo(
                    name=name,
                    size=size_str,
                    modified=modified[:10] if modified else "",
                    family=family,
                    parameter_size=param_size,
                    quantization=quantization,
                ))
            
            # Sort: recommended first, then by name
            def sort_key(m):
                is_recommended = m.name in self.RECOMMENDED_MODELS
                return (not is_recommended, m.name.lower())
            
            models.sort(key=sort_key)
            self._models_cache = models
            return models
            
        except requests.RequestException:
            return []
    
    def get_model_names(self, refresh: bool = False) -> list[str]:
        """Get just the model names (for dropdowns)."""
        return [m.name for m in self.get_available_models(refresh)]
    
    def get_best_model(self, preferred: Optional[str] = None) -> Optional[str]:
        """Get the best available model.
        
        Args:
            preferred: Preferred model name (used if available)
            
        Returns:
            Model name, or None if no models available
        """
        models = self.get_model_names(refresh=True)
        
        if not models:
            return None
        
        # Use preferred if available
        if preferred and preferred in models:
            return preferred
        
        # Try recommended models in order
        for rec in self.RECOMMENDED_MODELS:
            if rec in models:
                return rec
        
        # Just use the first available
        return models[0]
    
    def _infer_family(self, name: str) -> str:
        """Infer model family from name."""
        name_lower = name.lower()
        if "llama" in name_lower:
            return "llama"
        elif "gemma" in name_lower:
            return "gemma"
        elif "phi" in name_lower:
            return "phi"
        elif "mistral" in name_lower:
            return "mistral"
        elif "qwen" in name_lower:
            return "qwen"
        elif "deepseek" in name_lower:
            return "deepseek"
        return "unknown"
    
    def _infer_param_size(self, name: str) -> str:
        """Infer parameter size from model name."""
        import re
        # Look for patterns like :3b, :7b, :12b, etc.
        match = re.search(r':(\d+\.?\d*[bB])', name)
        if match:
            return match.group(1).upper()
        # Look for patterns in the name itself
        match = re.search(r'(\d+\.?\d*)[bB]', name)
        if match:
            return f"{match.group(1)}B"
        return ""
    
    def pull_model(
        self,
        model_name: str,
        on_progress: Optional[callable] = None,
    ) -> bool:
        """Pull a model from Ollama registry.
        
        Args:
            model_name: Name of model to pull
            on_progress: Callback for progress updates (0-100)
            
        Returns:
            True if successful
        """
        try:
            response = requests.post(
                f"{self.host}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=600,  # 10 minutes for large models
            )
            
            if response.status_code != 200:
                return False
            
            for line in response.iter_lines():
                if line and on_progress:
                    import json
                    try:
                        data = json.loads(line)
                        total = data.get("total", 0)
                        completed = data.get("completed", 0)
                        if total > 0:
                            progress = int((completed / total) * 100)
                            on_progress(progress)
                    except json.JSONDecodeError:
                        pass
            
            # Refresh cache
            self.get_available_models(refresh=True)
            return True
            
        except requests.RequestException:
            return False
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a model from Ollama.
        
        Args:
            model_name: Name of model to delete
            
        Returns:
            True if successful
        """
        try:
            response = requests.delete(
                f"{self.host}/api/delete",
                json={"name": model_name},
                timeout=30,
            )
            
            if response.status_code == 200:
                self.get_available_models(refresh=True)
                return True
            return False
            
        except requests.RequestException:
            return False


# Global instance
_manager: ModelManager | None = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager


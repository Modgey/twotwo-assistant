"""TwoTwo Configuration Management

Handles loading and saving configuration from %APPDATA%/TwoTwo/config.json
"""

import json
import os
from pathlib import Path
from typing import Any


def get_config_dir() -> Path:
    """Get the configuration directory path."""
    appdata = os.environ.get("APPDATA")
    if appdata:
        config_dir = Path(appdata) / "TwoTwo"
    else:
        # Fallback for non-Windows systems
        config_dir = Path.home() / ".twotwo"
    
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_path() -> Path:
    """Get the full path to the config file."""
    return get_config_dir() / "config.json"


DEFAULT_CONFIG = {
    "version": "1.0",
    
    "ui": {
        "avatar_position": {"x": 100, "y": 100},
        "avatar_size": "medium",
        "opacity": 0.85,
        "text_display_duration": 5.0
    },
    
    "voice": {
        "hotkey": "x",
        "stt_model": "tiny",
        "tts_voice": "en_US-lessac-medium",
        "tts_speed": 1.1,
        "voice_style": "claptrap",
        "input_device": None,
        "output_device": None
    },
    
    "ai": {
        "backend": "ollama",
        "model": "llama3.2:3b",
        "personality": "You are TwoTwo, a helpful AI assistant with a calm and professional demeanor. You provide concise, accurate responses.",
        "brave_api_key": "",
        "enable_search": True
    }
}


class Config:
    """Configuration manager with auto-save functionality."""
    
    def __init__(self):
        self._data: dict = {}
        self._config_path = get_config_path()
        self.load()
    
    def load(self) -> None:
        """Load configuration from file, creating defaults if needed."""
        if self._config_path.exists():
            try:
                with open(self._config_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                # Merge with defaults to ensure all keys exist
                self._data = self._merge_defaults(DEFAULT_CONFIG, self._data)
            except (json.JSONDecodeError, IOError):
                self._data = DEFAULT_CONFIG.copy()
                self.save()
        else:
            self._data = DEFAULT_CONFIG.copy()
            self.save()
    
    def _merge_defaults(self, defaults: dict, current: dict) -> dict:
        """Recursively merge defaults with current config."""
        result = defaults.copy()
        for key, value in current.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_defaults(result[key], value)
            else:
                result[key] = value
        return result
    
    def save(self) -> None:
        """Save current configuration to file."""
        try:
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save config: {e}")
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """Get a nested config value using dot notation.
        
        Example: config.get("ui", "avatar_position", "x")
        """
        value = self._data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, *keys_and_value) -> None:
        """Set a nested config value and auto-save.
        
        Example: config.set("ui", "avatar_position", "x", 200)
        The last argument is the value, all preceding are keys.
        """
        if len(keys_and_value) < 2:
            raise ValueError("Need at least one key and a value")
        
        keys = keys_and_value[:-1]
        value = keys_and_value[-1]
        
        # Navigate to the parent dict
        current = self._data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the value
        current[keys[-1]] = value
        self.save()
    
    @property
    def data(self) -> dict:
        """Get the raw config data."""
        return self._data


# Global config instance
_config: Config | None = None


def get_config() -> Config:
    """Get the global config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config



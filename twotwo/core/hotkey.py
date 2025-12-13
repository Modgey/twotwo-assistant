"""TwoTwo Global Hotkey Handler

Handles global keyboard shortcuts for push-to-talk and other actions.
Uses pynput for cross-platform global hotkey support.
"""

import threading
from typing import Callable, Optional, Set
from enum import Enum, auto

from PySide6.QtCore import QObject, Signal


class HotkeyAction(Enum):
    """Available hotkey actions."""
    PUSH_TO_TALK = auto()
    CANCEL = auto()
    TOGGLE_VISIBILITY = auto()


class HotkeyHandler(QObject):
    """Global hotkey handler with Qt signal integration."""
    
    # Signals for hotkey events
    ptt_pressed = Signal()      # Push-to-talk key pressed
    ptt_released = Signal()     # Push-to-talk key released
    cancel_pressed = Signal()   # Cancel key pressed
    
    def __init__(self, ptt_key: str = "x"):
        super().__init__()
        
        self._ptt_key = ptt_key.lower()
        self._listener = None
        self._running = False
        self._pressed_keys: Set[str] = set()
        self._ptt_active = False
        
    def start(self):
        """Start listening for global hotkeys."""
        if self._running:
            return
        
        try:
            from pynput import keyboard
            
            self._running = True
            self._listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release,
            )
            self._listener.start()
            print(f"Hotkey listener started. Push-to-talk: {self._ptt_key.upper()}")
            
        except ImportError:
            print("pynput not installed. Hotkeys disabled.")
            print("Install with: pip install pynput")
        except Exception as e:
            print(f"Failed to start hotkey listener: {e}")
    
    def stop(self):
        """Stop listening for global hotkeys."""
        self._running = False
        if self._listener:
            self._listener.stop()
            self._listener = None
        print("Hotkey listener stopped")
    
    def _normalize_key(self, key) -> Optional[str]:
        """Normalize a pynput key to a string."""
        try:
            from pynput import keyboard
            
            if hasattr(key, 'char') and key.char:
                return key.char.lower()
            elif hasattr(key, 'name'):
                return key.name.lower()
            elif key == keyboard.Key.space:
                return 'space'
            elif key == keyboard.Key.esc:
                return 'escape'
            else:
                return str(key).lower()
        except Exception:
            return None
    
    def _on_key_press(self, key):
        """Handle key press event."""
        if not self._running:
            return
        
        key_name = self._normalize_key(key)
        if not key_name:
            return
        
        # Track pressed keys
        self._pressed_keys.add(key_name)
        
        # Check for push-to-talk
        if key_name == self._ptt_key and not self._ptt_active:
            self._ptt_active = True
            self.ptt_pressed.emit()
        
        # Check for cancel (Escape)
        elif key_name in ('escape', 'esc'):
            self.cancel_pressed.emit()
    
    def _on_key_release(self, key):
        """Handle key release event."""
        if not self._running:
            return
        
        key_name = self._normalize_key(key)
        if not key_name:
            return
        
        # Track released keys
        self._pressed_keys.discard(key_name)
        
        # Check for push-to-talk release
        if key_name == self._ptt_key and self._ptt_active:
            self._ptt_active = False
            self.ptt_released.emit()
    
    def set_ptt_key(self, key: str):
        """Change the push-to-talk key."""
        self._ptt_key = key.lower()
        print(f"Push-to-talk key changed to: {self._ptt_key.upper()}")
    
    @property
    def ptt_key(self) -> str:
        """Get the current push-to-talk key."""
        return self._ptt_key
    
    @property
    def is_ptt_active(self) -> bool:
        """Check if push-to-talk is currently active."""
        return self._ptt_active
    
    def is_running(self) -> bool:
        """Check if the hotkey listener is running."""
        return self._running


class KeyCapture(QObject):
    """Utility for capturing a key press (for settings)."""
    
    key_captured = Signal(str)  # Emitted when a key is captured
    
    def __init__(self):
        super().__init__()
        self._listener = None
        self._capturing = False
    
    def start_capture(self):
        """Start capturing the next key press."""
        if self._capturing:
            return
        
        try:
            from pynput import keyboard
            
            self._capturing = True
            
            def on_press(key):
                if not self._capturing:
                    return False
                
                # Normalize key
                if hasattr(key, 'char') and key.char:
                    key_name = key.char.lower()
                elif hasattr(key, 'name'):
                    key_name = key.name.lower()
                else:
                    key_name = str(key).lower()
                
                self._capturing = False
                self.key_captured.emit(key_name)
                return False  # Stop listener
            
            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.start()
            
        except ImportError:
            print("pynput not installed")
            self._capturing = False
        except Exception as e:
            print(f"Key capture error: {e}")
            self._capturing = False
    
    def cancel_capture(self):
        """Cancel key capture."""
        self._capturing = False
        if self._listener:
            self._listener.stop()
            self._listener = None
    
    @property
    def is_capturing(self) -> bool:
        """Check if currently capturing."""
        return self._capturing


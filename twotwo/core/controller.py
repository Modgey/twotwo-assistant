"""TwoTwo Core Controller

Main application controller that coordinates all components.
"""

from PySide6.QtCore import QObject, Slot
from PySide6.QtWidgets import QApplication

from config import get_config
from core.state import AppState, AvatarState
from ui.overlay_window import OverlayWindow
from ui.tray_icon import TrayIcon


class Controller(QObject):
    """Main application controller."""
    
    def __init__(self, app: QApplication):
        super().__init__()
        
        self._app = app
        self._config = get_config()
        
        # Application state
        self._app_state = AppState.STARTING
        self._avatar_state = AvatarState.IDLE
        
        # UI components
        self._overlay: OverlayWindow | None = None
        self._tray: TrayIcon | None = None
        
        self._setup_components()
        self._connect_signals()
        
        # Mark as ready
        self._app_state = AppState.READY
    
    def _setup_components(self):
        """Initialize all UI components."""
        # Create overlay window
        self._overlay = OverlayWindow()
        
        # Create tray icon
        self._tray = TrayIcon(self)
    
    def _connect_signals(self):
        """Connect signals between components."""
        # Tray icon signals
        self._tray.quit_requested.connect(self._on_quit_requested)
        self._tray.settings_requested.connect(self._on_settings_requested)
        self._tray.toggle_visibility_requested.connect(self._on_toggle_visibility)
        self._tray.demo_state_requested.connect(self._on_demo_state_requested)
        
        # Overlay signals
        self._overlay.visibility_changed.connect(self._on_overlay_visibility_changed)
        self._overlay.avatar_state_changed.connect(self._on_avatar_state_changed)
    
    @Slot()
    def _on_quit_requested(self):
        """Handle quit request from tray menu."""
        self._app_state = AppState.SHUTTING_DOWN
        self._cleanup()
        self._app.quit()
    
    @Slot()
    def _on_settings_requested(self):
        """Handle settings request from tray menu."""
        # Placeholder - will be implemented in Phase 5
        self._tray.show_message(
            "Settings",
            "Settings panel will be available in a future update.",
        )
    
    @Slot()
    def _on_toggle_visibility(self):
        """Handle toggle visibility request."""
        if self._overlay:
            self._overlay.toggle_visibility()
    
    @Slot(bool)
    def _on_overlay_visibility_changed(self, is_visible: bool):
        """Handle overlay visibility change."""
        if self._tray:
            self._tray.update_visibility_state(is_visible)
    
    @Slot(AvatarState)
    def _on_avatar_state_changed(self, state: AvatarState):
        """Handle avatar state change from overlay."""
        self._avatar_state = state
    
    @Slot(AvatarState)
    def _on_demo_state_requested(self, state: AvatarState):
        """Handle demo state request from tray menu."""
        self.set_avatar_state(state)
    
    def _cleanup(self):
        """Clean up resources before shutdown."""
        if self._tray:
            self._tray.hide()
        if self._overlay:
            self._overlay.close()
    
    def start(self):
        """Start the application - show UI components."""
        if self._overlay:
            self._overlay.show()
        
        if self._tray:
            self._tray.show()
    
    @property
    def app_state(self) -> AppState:
        """Get current application state."""
        return self._app_state
    
    @property
    def avatar_state(self) -> AvatarState:
        """Get current avatar state."""
        return self._avatar_state
    
    def set_avatar_state(self, state: AvatarState):
        """Set the avatar animation state."""
        self._avatar_state = state
        if self._overlay:
            self._overlay.set_avatar_state(state)
    
    def set_audio_amplitude(self, amplitude: float):
        """Set audio amplitude for speaking animation (0-1)."""
        if self._overlay:
            self._overlay.set_audio_amplitude(amplitude)

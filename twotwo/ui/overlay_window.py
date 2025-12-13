"""TwoTwo Overlay Window

Transparent, always-on-top overlay window that hosts the avatar.
"""

from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtWidgets import QMainWindow

from config import get_config
from core.state import AvatarState
from avatar.renderer import AvatarRenderer


class OverlayWindow(QMainWindow):
    """Transparent overlay window for TwoTwo avatar."""
    
    # Signals
    position_changed = Signal(int, int)
    visibility_changed = Signal(bool)
    avatar_state_changed = Signal(AvatarState)
    
    def __init__(self):
        super().__init__()
        
        self.config = get_config()
        self._dragging = False
        self._drag_offset = QPoint()
        self._alt_pressed = False
        
        self._setup_window()
        self._setup_avatar()
        self._restore_position()
    
    def _setup_window(self):
        """Configure window flags and attributes for transparent overlay."""
        self.setWindowTitle("TwoTwo")
        
        # Window flags for transparent, always-on-top overlay
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool  # Prevents taskbar entry
        )
        
        # Transparency attributes
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        
        # Get avatar size from config
        size_map = {"small": 150, "medium": 200, "large": 250}
        size_name = self.config.get("ui", "avatar_size", default="medium")
        self.avatar_size = size_map.get(size_name, 200)
        
        # Set window size to match avatar with padding for glow
        self.setFixedSize(self.avatar_size + 40, self.avatar_size + 40)
    
    def _setup_avatar(self):
        """Set up the animated avatar renderer."""
        self.avatar = AvatarRenderer(self.avatar_size, self)
        self.avatar.move(20, 20)  # Center with padding for glow
        self.avatar.state_changed.connect(self.avatar_state_changed.emit)
        self.setCentralWidget(None)  # Don't use central widget layout
    
    def _restore_position(self):
        """Restore window position from config."""
        x = self.config.get("ui", "avatar_position", "x", default=100)
        y = self.config.get("ui", "avatar_position", "y", default=100)
        self.move(x, y)
    
    def _save_position(self):
        """Save current window position to config."""
        pos = self.pos()
        self.config.set("ui", "avatar_position", "x", pos.x())
        self.config.set("ui", "avatar_position", "y", pos.y())
        self.position_changed.emit(pos.x(), pos.y())
    
    def set_avatar_state(self, state: AvatarState):
        """Set the avatar animation state."""
        self.avatar.set_state(state)
    
    def set_audio_amplitude(self, amplitude: float):
        """Set audio amplitude for speaking animation."""
        self.avatar.set_audio_amplitude(amplitude)
    
    def keyPressEvent(self, event):
        """Track Alt key for drag mode."""
        if event.key() == Qt.Key.Key_Alt:
            self._alt_pressed = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event):
        """Track Alt key release."""
        if event.key() == Qt.Key.Key_Alt:
            self._alt_pressed = False
            if not self._dragging:
                self.setCursor(Qt.CursorShape.ArrowCursor)
        super().keyReleaseEvent(event)
    
    def mousePressEvent(self, event):
        """Start dragging if Alt is held."""
        if event.button() == Qt.MouseButton.LeftButton and self._alt_pressed:
            self._dragging = True
            self._drag_offset = event.globalPosition().toPoint() - self.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle window dragging."""
        if self._dragging:
            new_pos = event.globalPosition().toPoint() - self._drag_offset
            self.move(new_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """End dragging and save position."""
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self._save_position()
            if self._alt_pressed:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)
    
    def toggle_visibility(self):
        """Toggle window visibility."""
        if self.isVisible():
            self.hide()
            self.visibility_changed.emit(False)
        else:
            self.show()
            self.visibility_changed.emit(True)
    
    def showEvent(self, event):
        """Emit signal when shown."""
        super().showEvent(event)
        self.visibility_changed.emit(True)
    
    def hideEvent(self, event):
        """Emit signal when hidden."""
        super().hideEvent(event)
        self.visibility_changed.emit(False)
    
    def closeEvent(self, event):
        """Clean up avatar renderer on close."""
        self.avatar.cleanup()
        super().closeEvent(event)

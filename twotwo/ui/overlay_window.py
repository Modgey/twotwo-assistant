"""TwoTwo Overlay Window

Transparent, always-on-top overlay window that hosts the avatar and text display.
"""

from PySide6.QtCore import Qt, Signal, QPoint, QTimer
from PySide6.QtWidgets import QMainWindow

from config import get_config
from core.state import AvatarState
from avatar.renderer import AvatarRenderer
from ui.text_display import TextDisplay


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
        self._setup_text_display()
        self._restore_position()
        
        # Auto-hide timer for text
        self._text_hide_timer = QTimer(self)
        self._text_hide_timer.setSingleShot(True)
        self._text_hide_timer.timeout.connect(self._hide_text)
    
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
        
        # Set window size to accommodate text on multiple sides of avatar
        self._text_area_h = 280  # Horizontal text area (left or right)
        self._text_area_v = 100  # Vertical text area (above or below)
        self._padding = 10       # General padding
        
        # Avatar position (with space for text on left and above)
        self._avatar_x = self._text_area_h + self._padding
        self._avatar_y = self._text_area_v + self._padding
        
        # Window size
        window_width = self._text_area_h + self.avatar_size + self._text_area_h + self._padding * 2
        window_height = self._text_area_v + self.avatar_size + self._text_area_v + self._padding * 2
        self.setFixedSize(window_width, window_height)
    
    def _setup_avatar(self):
        """Set up the animated avatar renderer."""
        self.avatar = AvatarRenderer(self.avatar_size, self)
        self.avatar.move(self._avatar_x, self._avatar_y)
        self.avatar.state_changed.connect(self.avatar_state_changed.emit)
        self.setCentralWidget(None)  # Don't use central widget layout
    
    def _setup_text_display(self):
        """Set up text display for AI response."""
        self.text_display = TextDisplay(self)
        # Store position info for dynamic centering
        self._response_x = self._avatar_x + self.avatar_size
        self._avatar_center_y = self._avatar_y + (self.avatar_size // 2)
        self.text_display.move(self._response_x, self._avatar_center_y)
        self.text_display.hide()
    
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
    
    def set_opacity(self, opacity: float):
        """Set avatar opacity (0.0 to 1.0)."""
        self.avatar.set_opacity(opacity)

    def set_theme(self, theme_name: str):
        """Update color theme for avatar and text."""
        self.avatar.set_theme(theme_name)
        self.text_display.set_theme(theme_name)
    
    def show_text(self, text: str, duration: float = 0):
        """Show AI response text with smart positioning based on screen location.
        
        Args:
            text: Text to display
            duration: Auto-hide after this many seconds (0 = manual hide)
        """
        # Determine alignment FIRST based on screen position
        alignment = self._get_smart_alignment()
        self.text_display.set_alignment(alignment)
        
        # Now show text (will render with correct alignment)
        self.text_display.show_text(text)
        
        # Smart position after text renders and we know its size
        from PySide6.QtCore import QTimer
        QTimer.singleShot(10, self._smart_position_text)
        
        if duration > 0:
            self._text_hide_timer.start(int(duration * 1000))
    
    def _get_smart_alignment(self) -> str:
        """Determine text alignment based on avatar's screen position."""
        from PySide6.QtWidgets import QApplication
        
        screen = QApplication.primaryScreen().geometry()
        screen_width = screen.width()
        
        avatar_global_pos = self.mapToGlobal(self.avatar.pos())
        avatar_center_x = avatar_global_pos.x() + (self.avatar_size // 2)
        
        left_zone = screen_width / 3
        right_zone = screen_width * 2 / 3
        
        if avatar_center_x < left_zone:
            return "left"
        elif avatar_center_x > right_zone:
            return "right"
        else:
            return "center"
    
    def _smart_position_text(self):
        """Position text smartly based on avatar's screen location."""
        from PySide6.QtWidgets import QApplication
        
        # Get screen dimensions
        screen = QApplication.primaryScreen().geometry()
        screen_width = screen.width()
        screen_height = screen.height()
        
        # Get avatar's absolute position on screen
        avatar_global_pos = self.mapToGlobal(self.avatar.pos())
        avatar_center_x = avatar_global_pos.x() + (self.avatar_size // 2)
        avatar_center_y = avatar_global_pos.y() + (self.avatar_size // 2)
        
        # Get text dimensions
        text_width = self.text_display.width()
        text_height = self.text_display.height()
        
        # Small offset to bring text closer to avatar (but not overlapping)
        avatar_padding = int(self.avatar_size * 0.05)  # ~10px for 200px avatar
        
        # Determine horizontal zone (left third, center third, right third)
        left_zone = screen_width / 3
        right_zone = screen_width * 2 / 3
        
        # Determine vertical zone (top half, bottom half)
        vertical_mid = screen_height / 2
        
        # Calculate best position (alignment already set in show_text)
        if avatar_center_x < left_zone:
            # Avatar on LEFT → text to the RIGHT (tuck into avatar padding)
            text_x = self._avatar_x + self.avatar_size - avatar_padding
            text_y = self._avatar_y + (self.avatar_size // 2) - (text_height // 2)
            
        elif avatar_center_x > right_zone:
            # Avatar on RIGHT → text to the LEFT (tuck into avatar padding)
            text_x = self._avatar_x + avatar_padding - text_width
            text_y = self._avatar_y + (self.avatar_size // 2) - (text_height // 2)
            
        else:
            # Avatar in CENTER horizontally
            if avatar_center_y < vertical_mid:
                # Top center → text BELOW (tuck into avatar padding)
                text_x = self._avatar_x + (self.avatar_size // 2) - (text_width // 2)
                text_y = self._avatar_y + self.avatar_size - avatar_padding
            else:
                # Bottom center → text ABOVE (tuck into avatar padding)
                text_x = self._avatar_x + (self.avatar_size // 2) - (text_width // 2)
                text_y = self._avatar_y + avatar_padding - text_height
        
        # Clamp to stay within window bounds
        text_x = max(0, text_x)
        text_y = max(0, text_y)
        
        self.text_display.move(int(text_x), int(text_y))
    
    def _hide_text(self):
        """Hide the AI response text display."""
        self.text_display.hide_text()
    
    def clear_text(self):
        """Immediately clear and hide text."""
        self._text_hide_timer.stop()
        self.text_display.clear()
    
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
        self.text_display.cleanup()
        super().closeEvent(event)

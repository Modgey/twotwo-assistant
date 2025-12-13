"""TwoTwo System Tray Icon

System tray icon with context menu for settings and quit.
"""

from PySide6.QtCore import Signal, QObject
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QRadialGradient
from PySide6.QtWidgets import QSystemTrayIcon, QMenu

from core.state import AvatarState


def create_tray_icon_pixmap(size: int = 64) -> QPixmap:
    """Create a simple tray icon pixmap (aperture eye design)."""
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(0, 0, 0, 0))
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    
    center = size // 2
    outer_radius = size // 2 - 4
    inner_radius = outer_radius // 2
    pupil_radius = inner_radius // 2
    
    # Outer ring
    ring_gradient = QRadialGradient(center, center, outer_radius)
    ring_gradient.setColorAt(0.5, QColor(60, 70, 80))
    ring_gradient.setColorAt(0.7, QColor(180, 190, 200))
    ring_gradient.setColorAt(0.9, QColor(220, 225, 230))
    ring_gradient.setColorAt(1, QColor(150, 160, 170))
    
    painter.setBrush(ring_gradient)
    painter.setPen(QColor(100, 110, 120))
    painter.drawEllipse(center - outer_radius, center - outer_radius,
                       outer_radius * 2, outer_radius * 2)
    
    # Inner dark area
    inner_gradient = QRadialGradient(center, center, inner_radius)
    inner_gradient.setColorAt(0, QColor(25, 30, 40))
    inner_gradient.setColorAt(1, QColor(45, 50, 60))
    
    painter.setBrush(inner_gradient)
    painter.setPen(QColor(50, 55, 65))
    painter.drawEllipse(center - inner_radius, center - inner_radius,
                       inner_radius * 2, inner_radius * 2)
    
    # Pupil
    pupil_gradient = QRadialGradient(center - 1, center - 1, pupil_radius)
    pupil_gradient.setColorAt(0, QColor(220, 230, 240))
    pupil_gradient.setColorAt(1, QColor(180, 190, 200))
    
    painter.setBrush(pupil_gradient)
    painter.setPen(QColor(200, 210, 220))
    painter.drawEllipse(center - pupil_radius, center - pupil_radius,
                       pupil_radius * 2, pupil_radius * 2)
    
    painter.end()
    return pixmap


class TrayIcon(QObject):
    """System tray icon with context menu."""
    
    # Signals
    settings_requested = Signal()
    quit_requested = Signal()
    toggle_visibility_requested = Signal()
    demo_state_requested = Signal(AvatarState)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._tray = QSystemTrayIcon(parent)
        self._setup_icon()
        self._setup_menu()
        self._connect_signals()
    
    def _setup_icon(self):
        """Create and set the tray icon."""
        pixmap = create_tray_icon_pixmap(64)
        icon = QIcon(pixmap)
        self._tray.setIcon(icon)
        self._tray.setToolTip("TwoTwo - AI Assistant")
    
    def _setup_menu(self):
        """Create the context menu."""
        self._menu = QMenu()
        
        # Show/Hide action
        self._toggle_action = self._menu.addAction("Hide")
        self._toggle_action.triggered.connect(self._on_toggle_visibility)
        
        self._menu.addSeparator()
        
        # Demo submenu for testing avatar states
        demo_menu = self._menu.addMenu("Demo States")
        
        idle_action = demo_menu.addAction("Idle")
        idle_action.triggered.connect(lambda: self.demo_state_requested.emit(AvatarState.IDLE))
        
        listening_action = demo_menu.addAction("Listening")
        listening_action.triggered.connect(lambda: self.demo_state_requested.emit(AvatarState.LISTENING))
        
        thinking_action = demo_menu.addAction("Thinking")
        thinking_action.triggered.connect(lambda: self.demo_state_requested.emit(AvatarState.THINKING))
        
        speaking_action = demo_menu.addAction("Speaking")
        speaking_action.triggered.connect(lambda: self.demo_state_requested.emit(AvatarState.SPEAKING))
        
        self._menu.addSeparator()
        
        # Settings action (placeholder for now)
        self._settings_action = self._menu.addAction("Settings...")
        self._settings_action.triggered.connect(self.settings_requested.emit)
        
        self._menu.addSeparator()
        
        # Quit action
        self._quit_action = self._menu.addAction("Quit")
        self._quit_action.triggered.connect(self.quit_requested.emit)
        
        self._tray.setContextMenu(self._menu)
    
    def _connect_signals(self):
        """Connect tray icon signals."""
        self._tray.activated.connect(self._on_activated)
    
    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason):
        """Handle tray icon activation (double-click)."""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.toggle_visibility_requested.emit()
    
    def _on_toggle_visibility(self):
        """Handle toggle visibility menu action."""
        self.toggle_visibility_requested.emit()
    
    def update_visibility_state(self, is_visible: bool):
        """Update the toggle action text based on overlay visibility."""
        self._toggle_action.setText("Hide" if is_visible else "Show")
    
    def show(self):
        """Show the tray icon."""
        self._tray.show()
    
    def hide(self):
        """Hide the tray icon."""
        self._tray.hide()
    
    def show_message(self, title: str, message: str, 
                     icon: QSystemTrayIcon.MessageIcon = QSystemTrayIcon.MessageIcon.Information,
                     duration_ms: int = 3000):
        """Show a notification message."""
        self._tray.showMessage(title, message, icon, duration_ms)

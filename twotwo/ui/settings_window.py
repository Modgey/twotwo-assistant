"""TwoTwo Settings Panel

Minimal, premium black and white settings interface.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QSlider, QPushButton, QFrame,
    QScrollArea, QLineEdit, QApplication
)

from config import get_config
from ui.theme import AMBER
from voice.stt import WhisperSTT


class SettingsSection(QFrame):
    """A collapsible section in the settings panel."""
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("section")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 16)
        layout.setSpacing(12)
        
        # Section title
        title_label = QLabel(title.upper())
        title_label.setObjectName("sectionTitle")
        layout.addWidget(title_label)
        
        # Content container
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)
        layout.addWidget(self.content)
    
    def add_row(self, label_text: str, widget: QWidget) -> QWidget:
        """Add a labeled row to the section."""
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(12)
        
        label = QLabel(label_text)
        label.setObjectName("rowLabel")
        label.setFixedWidth(100)
        row_layout.addWidget(label)
        
        row_layout.addWidget(widget, 1)
        
        self.content_layout.addWidget(row)
        return widget


class SettingsWindow(QWidget):
    """Minimal premium settings panel."""
    
    # Signals
    closed = Signal()
    settings_changed = Signal(str, object)  # key, value
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.config = get_config()
        
        self._setup_window()
        self._setup_style()
        self._setup_ui()
        self._load_settings()
    
    def _setup_window(self):
        """Configure window properties."""
        self.setWindowTitle("Settings")
        self.setFixedWidth(320)
        self.setMinimumHeight(400)
        self.setMaximumHeight(600)
        
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    
    def _setup_style(self):
        """Set up the black and white minimal style with amber accent."""
        # Use amber accent color from theme
        amber_rgb = f"rgb({AMBER[0]}, {AMBER[1]}, {AMBER[2]})"
        amber_dim = f"rgba({AMBER[0]}, {AMBER[1]}, {AMBER[2]}, 0.3)"
        
        style = """
            QWidget {
                background-color: transparent;
                color: #ffffff;
                font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
                font-size: 11px;
            }
            
            #mainFrame {
                background-color: rgba(10, 10, 10, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
            }
            
            #titleBar {
                background-color: transparent;
                border-bottom: 1px solid rgba(255, 255, 255, 0.08);
                padding: 12px 16px;
            }
            
            #titleLabel {
                font-size: 13px;
                font-weight: 600;
                letter-spacing: 1px;
                color: #ffffff;
            }
            
            #closeButton {
                background-color: transparent;
                border: none;
                color: rgba(255, 255, 255, 0.5);
                font-size: 16px;
                padding: 4px 8px;
            }
            
            #closeButton:hover {
                color: #ffffff;
            }
            
            #sectionTitle {
                font-size: 10px;
                font-weight: 600;
                letter-spacing: 2px;
                color: rgba(255, 255, 255, 0.4);
                padding-bottom: 8px;
            }
            
            #rowLabel {
                color: rgba(255, 255, 255, 0.7);
            }
            
            QComboBox {
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 4px;
                padding: 6px 10px;
                min-height: 24px;
            }
            
            QComboBox:hover {
                border-color: $AMBER_DIM;
            }
            
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid rgba(255, 255, 255, 0.5);
                margin-right: 8px;
            }
            
            QComboBox QAbstractItemView {
                background-color: rgb(20, 20, 20);
                border: 1px solid rgba(255, 255, 255, 0.1);
                selection-background-color: rgba(255, 255, 255, 0.1);
                outline: none;
            }
            
            QSlider {
                height: 24px;
            }
            
            QSlider::groove:horizontal {
                background: rgba(255, 255, 255, 0.1);
                height: 2px;
                border-radius: 1px;
            }
            
            QSlider::handle:horizontal {
                background: $AMBER_RGB;
                width: 12px;
                height: 12px;
                margin: -5px 0;
                border-radius: 6px;
            }
            
            QSlider::handle:horizontal:hover {
                background: $AMBER_RGB;
                width: 14px;
                height: 14px;
                margin: -6px 0;
                border-radius: 7px;
            }
            
            QLineEdit {
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 4px;
                padding: 6px 10px;
                color: #ffffff;
            }
            
            QLineEdit:focus {
                border-color: $AMBER_DIM;
            }
            
            QPushButton {
                background-color: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: 500;
            }
            
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.12);
                border-color: rgba(255, 255, 255, 0.25);
            }
            
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.05);
            }
            
            QScrollArea {
                border: none;
                background: transparent;
            }
            
            QScrollBar:vertical {
                background: transparent;
                width: 6px;
                margin: 0;
            }
            
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 0.2);
                border-radius: 3px;
                min-height: 30px;
            }
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """
        
        # Replace color placeholders
        style = style.replace("$AMBER_RGB", amber_rgb)
        style = style.replace("$AMBER_DIM", amber_dim)
        
        self.setStyleSheet(style)
    
    def _setup_ui(self):
        """Build the UI structure."""
        # Main layout (for transparency)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        
        # Main frame with background
        main_frame = QFrame()
        main_frame.setObjectName("mainFrame")
        outer_layout.addWidget(main_frame)
        
        main_layout = QVBoxLayout(main_frame)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Title bar
        title_bar = QWidget()
        title_bar.setObjectName("titleBar")
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(16, 12, 12, 12)
        
        title_label = QLabel("SETTINGS")
        title_label.setObjectName("titleLabel")
        title_layout.addWidget(title_label)
        
        title_layout.addStretch()
        
        close_btn = QPushButton("×")
        close_btn.setObjectName("closeButton")
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.close)
        title_layout.addWidget(close_btn)
        
        main_layout.addWidget(title_bar)
        
        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.setSpacing(8)
        
        # Voice section
        voice_section = SettingsSection("Voice")
        
        self._voice_style = QComboBox()
        self._voice_style.addItems(["Normal", "Robot", "Claptrap", "GLaDOS", "Metallic"])
        self._voice_style.currentTextChanged.connect(self._on_voice_style_changed)
        voice_section.add_row("Style", self._voice_style)
        
        self._tts_speed = QSlider(Qt.Orientation.Horizontal)
        self._tts_speed.setRange(50, 150)
        self._tts_speed.setValue(100)
        self._tts_speed.valueChanged.connect(self._on_tts_speed_changed)
        voice_section.add_row("Speed", self._tts_speed)
        
        self._stt_model = QComboBox()
        available_models = WhisperSTT.list_available_models()
        if available_models:
            self._stt_model.addItems([m.capitalize() for m in available_models])
        else:
            self._stt_model.addItem("No models found")
            self._stt_model.setEnabled(False)
        self._stt_model.currentTextChanged.connect(self._on_stt_model_changed)
        voice_section.add_row("STT Model", self._stt_model)
        
        self._ptt_key = QLineEdit()
        self._ptt_key.setReadOnly(True)
        self._ptt_key.setPlaceholderText("Press a key...")
        self._ptt_key.mousePressEvent = self._start_key_capture
        voice_section.add_row("Push-to-Talk", self._ptt_key)
        
        content_layout.addWidget(voice_section)
        
        # Avatar section
        avatar_section = SettingsSection("Avatar")
        
        self._avatar_size = QComboBox()
        self._avatar_size.addItems(["Small", "Medium", "Large"])
        self._avatar_size.currentTextChanged.connect(self._on_avatar_size_changed)
        avatar_section.add_row("Size", self._avatar_size)
        
        self._opacity = QSlider(Qt.Orientation.Horizontal)
        self._opacity.setRange(50, 100)
        self._opacity.setValue(85)
        self._opacity.valueChanged.connect(self._on_opacity_changed)
        avatar_section.add_row("Opacity", self._opacity)
        
        content_layout.addWidget(avatar_section)
        
        # About section
        about_section = SettingsSection("About")
        
        version_label = QLabel("TwoTwo v1.0.0")
        version_label.setStyleSheet("color: rgba(255, 255, 255, 0.5);")
        about_section.content_layout.addWidget(version_label)
        
        content_layout.addWidget(about_section)
        
        content_layout.addStretch()
        
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        
        # Enable dragging
        self._drag_pos = None
        title_bar.mousePressEvent = self._on_title_mouse_press
        title_bar.mouseMoveEvent = self._on_title_mouse_move
    
    def _load_settings(self):
        """Load current settings into UI."""
        # Voice style
        style = self.config.get("voice", "voice_style", default="robot")
        style_map = {"normal": 0, "robot": 1, "claptrap": 2, "glados": 3, "metallic": 4}
        self._voice_style.setCurrentIndex(style_map.get(style.lower(), 1))
        
        # TTS speed
        speed = self.config.get("voice", "tts_speed", default=1.0)
        self._tts_speed.setValue(int(speed * 100))
        
        # STT model - find in available models
        model = self.config.get("voice", "stt_model", default="tiny")
        idx = self._stt_model.findText(model.capitalize())
        if idx >= 0:
            self._stt_model.setCurrentIndex(idx)
        
        # PTT key
        key = self.config.get("voice", "hotkey", default="x")
        self._ptt_key.setText(key.upper())
        
        # Avatar size
        size = self.config.get("ui", "avatar_size", default="medium")
        size_map = {"small": 0, "medium": 1, "large": 2}
        self._avatar_size.setCurrentIndex(size_map.get(size.lower(), 1))
        
        # Opacity
        opacity = self.config.get("ui", "opacity", default=0.85)
        self._opacity.setValue(int(opacity * 100))
    
    def _on_voice_style_changed(self, text: str):
        """Handle voice style change."""
        style = text.lower()
        self.config.set("voice", "voice_style", style)
        self.settings_changed.emit("voice_style", style)
    
    def _on_tts_speed_changed(self, value: int):
        """Handle TTS speed change."""
        speed = value / 100.0
        self.config.set("voice", "tts_speed", speed)
        self.settings_changed.emit("tts_speed", speed)
    
    def _on_stt_model_changed(self, text: str):
        """Handle STT model change."""
        model = text.lower()
        self.config.set("voice", "stt_model", model)
        self.settings_changed.emit("stt_model", model)
    
    def _on_avatar_size_changed(self, text: str):
        """Handle avatar size change."""
        size = text.lower()
        self.config.set("ui", "avatar_size", size)
        self.settings_changed.emit("avatar_size", size)
    
    def _on_opacity_changed(self, value: int):
        """Handle opacity change."""
        opacity = value / 100.0
        self.config.set("ui", "opacity", opacity)
        self.settings_changed.emit("opacity", opacity)
    
    def _start_key_capture(self, event):
        """Start capturing a key for PTT."""
        self._ptt_key.setText("Press a key...")
        self._ptt_key.setFocus()
    
    def keyPressEvent(self, event):
        """Capture key for PTT setting."""
        if self._ptt_key.hasFocus():
            key = event.text().lower() if event.text() else event.key()
            if isinstance(key, str) and key:
                self._ptt_key.setText(key.upper())
                self.config.set("voice", "hotkey", key)
                self.settings_changed.emit("hotkey", key)
                self._ptt_key.clearFocus()
        else:
            super().keyPressEvent(event)
    
    def _on_title_mouse_press(self, event):
        """Start dragging the window."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.pos()
    
    def _on_title_mouse_move(self, event):
        """Handle window dragging."""
        if self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
    
    def mouseReleaseEvent(self, event):
        """End dragging."""
        self._drag_pos = None
        super().mouseReleaseEvent(event)
    
    def closeEvent(self, event):
        """Emit closed signal."""
        self.closed.emit()
        super().closeEvent(event)


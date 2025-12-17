"""TwoTwo Settings Panel

Tabbed, premium black settings interface with amber accents.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QSlider, QPushButton, QFrame,
    QLineEdit, QTextEdit, QCheckBox, QTabWidget,
    QGridLayout, QSpacerItem, QSizePolicy
)

from config import get_config
from ui.theme import AMBER, THEMES
from voice.stt import WhisperSTT
from ai.model_manager import get_model_manager


class SettingsWindow(QWidget):
    """Tabbed premium settings panel."""
    
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
        self.setFixedWidth(420)
        self.setMinimumHeight(480)
        
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    
    def _setup_style(self):
        """Set up the black and white minimal style."""
        accent_rgb = "rgb(255, 255, 255)"
        accent_dim = "rgba(255, 255, 255, 0.4)"
        accent_faint = "rgba(255, 255, 255, 0.15)"
        
        style = """
            QWidget {
                background-color: transparent;
                color: #ffffff;
                font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
                font-size: 12px;
            }
            
            #mainFrame {
                background-color: rgba(12, 12, 12, 0.97);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 10px;
            }
            
            #titleBar {
                background-color: transparent;
                border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            }
            
            #titleLabel {
                font-size: 14px;
                font-weight: 600;
                letter-spacing: 1px;
                color: #ffffff;
            }
            
            #closeButton {
                background-color: transparent;
                border: none;
                color: rgba(255, 255, 255, 0.4);
                font-size: 20px;
                font-weight: 300;
                padding: 0;
                min-width: 32px;
                min-height: 32px;
            }
            
            #closeButton:hover {
                color: #ffffff;
                background-color: rgba(255, 80, 80, 0.3);
                border-radius: 6px;
            }
            
            /* Tab Widget */
            QTabWidget::pane {
                border: none;
                background: transparent;
                margin-top: -1px;
            }
            
            QTabBar {
                background: transparent;
            }
            
            QTabBar::tab {
                background: transparent;
                color: rgba(255, 255, 255, 0.5);
                padding: 10px 20px;
                margin-right: 4px;
                border: none;
                border-bottom: 2px solid transparent;
                font-weight: 500;
                font-size: 12px;
            }
            
            QTabBar::tab:hover {
                color: rgba(255, 255, 255, 0.8);
                background: rgba(255, 255, 255, 0.03);
            }
            
            QTabBar::tab:selected {
                color: $ACCENT_RGB;
                border-bottom: 2px solid $ACCENT_RGB;
            }
            
            /* Labels */
            #fieldLabel {
                color: rgba(255, 255, 255, 0.6);
                font-size: 11px;
                font-weight: 500;
                letter-spacing: 0.5px;
            }
            
            #sectionLabel {
                color: rgba(255, 255, 255, 0.35);
                font-size: 10px;
                font-weight: 600;
                letter-spacing: 1.5px;
                padding-top: 8px;
            }
            
            /* Form Controls */
            QComboBox {
                background-color: rgba(255, 255, 255, 0.06);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                padding: 8px 12px;
                min-height: 20px;
            }
            
            QComboBox:hover {
                border-color: $ACCENT_DIM;
                background-color: rgba(255, 255, 255, 0.08);
            }
            
            QComboBox:focus {
                border-color: $ACCENT_RGB;
            }
            
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid rgba(255, 255, 255, 0.4);
                margin-right: 10px;
            }
            
            QComboBox QAbstractItemView {
                background-color: rgb(20, 20, 20);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 6px;
                selection-background-color: $ACCENT_FAINT;
                padding: 4px;
                outline: none;
            }
            
            QSlider {
                height: 28px;
            }
            
            QSlider::groove:horizontal {
                background: rgba(255, 255, 255, 0.1);
                height: 4px;
                border-radius: 2px;
            }
            
            QSlider::sub-page:horizontal {
                background: $ACCENT_DIM;
                height: 4px;
                border-radius: 2px;
            }
            
            QSlider::handle:horizontal {
                background: $ACCENT_RGB;
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            
            QSlider::handle:horizontal:hover {
                background: #ffffff;
            }
            
            QLineEdit {
                background-color: rgba(255, 255, 255, 0.06);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                padding: 8px 12px;
                color: #ffffff;
                selection-background-color: $ACCENT_DIM;
                min-height: 20px;
            }
            
            QLineEdit:hover {
                border-color: $ACCENT_DIM;
            }
            
            QLineEdit:focus {
                border-color: $ACCENT_RGB;
                background-color: rgba(255, 255, 255, 0.08);
            }
            
            QTextEdit {
                background-color: rgba(255, 255, 255, 0.06);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                padding: 8px;
                color: #ffffff;
                selection-background-color: $ACCENT_DIM;
            }
            
            QTextEdit:hover {
                border-color: $ACCENT_DIM;
            }
            
            QTextEdit:focus {
                border-color: $ACCENT_RGB;
            }
            
            QCheckBox {
                spacing: 8px;
                color: rgba(255, 255, 255, 0.8);
            }
            
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                background: rgba(255, 255, 255, 0.05);
            }
            
            QCheckBox::indicator:hover {
                border-color: $ACCENT_DIM;
            }
            
            QCheckBox::indicator:checked {
                background: $ACCENT_RGB;
                border-color: $ACCENT_RGB;
            }
            
            /* Toggle Switch Style */
            #toolToggle::indicator {
                width: 32px;
                height: 16px;
                border-radius: 8px;
            }
            
            #toolToggle::indicator:unchecked {
                background-color: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            #toolToggle::indicator:checked {
                background-color: $ACCENT_RGB;
                border: 1px solid $ACCENT_RGB;
            }
            
            /* Tool Item */
            #toolItem {
                background-color: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 8px;
                margin-bottom: 8px;
            }
            
            #toolHeader {
                padding: 12px 16px;
                background: transparent;
            }
            
            #toolName {
                font-weight: 600;
                font-size: 13px;
                color: #ffffff;
            }
            
            #toolDesc {
                font-size: 11px;
                color: rgba(255, 255, 255, 0.4);
            }
            
            #toolSettingsBtn {
                background: transparent;
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 4px;
                color: rgba(255, 255, 255, 0.5);
                font-size: 9px;
                font-weight: 700;
                letter-spacing: 1px;
                padding: 4px 8px;
                min-width: 60px;
                min-height: 20px;
            }
            
            #toolSettingsBtn:hover {
                color: #ffffff;
                border-color: rgba(255, 255, 255, 0.3);
                background: rgba(255, 255, 255, 0.05);
            }
            
            #toolSettingsArea {
                background-color: rgba(0, 0, 0, 0.2);
                border-top: 1px solid rgba(255, 255, 255, 0.05);
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
                padding: 12px 16px;
            }
            
            QPushButton {
                background-color: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
                min-height: 20px;
            }
            
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.12);
                border-color: rgba(255, 255, 255, 0.2);
            }
            
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.06);
            }
            
            #accentButton {
                background-color: $ACCENT_FAINT;
                border-color: $ACCENT_DIM;
            }
            
            #accentButton:hover {
                background-color: $ACCENT_DIM;
            }
        """
        
        style = style.replace("$ACCENT_RGB", accent_rgb)
        style = style.replace("$ACCENT_DIM", accent_dim)
        style = style.replace("$ACCENT_FAINT", accent_faint)
        
        self.setStyleSheet(style)
    
    def _setup_ui(self):
        """Build the tabbed UI structure."""
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        
        # Main frame
        main_frame = QFrame()
        main_frame.setObjectName("mainFrame")
        outer_layout.addWidget(main_frame)
        
        main_layout = QVBoxLayout(main_frame)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Title bar
        title_bar = QWidget()
        title_bar.setObjectName("titleBar")
        title_bar.setFixedHeight(48)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(20, 0, 12, 0)
        
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
        
        # Tab widget
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        
        # Create tabs
        self._tabs.addTab(self._create_voice_tab(), "Voice")
        self._tabs.addTab(self._create_ai_tab(), "AI")
        self._tabs.addTab(self._create_tools_tab(), "Tools")
        self._tabs.addTab(self._create_display_tab(), "Display")
        
        main_layout.addWidget(self._tabs)
        
        # Enable dragging
        self._drag_pos = None
        title_bar.mousePressEvent = self._on_title_mouse_press
        title_bar.mouseMoveEvent = self._on_title_mouse_move
    
    def _create_field(self, label_text: str, widget: QWidget, parent_layout: QVBoxLayout):
        """Create a labeled field row."""
        label = QLabel(label_text)
        label.setObjectName("fieldLabel")
        parent_layout.addWidget(label)
        parent_layout.addWidget(widget)
        parent_layout.addSpacing(12)
    
    def _create_section(self, text: str, parent_layout: QVBoxLayout):
        """Create a section header."""
        label = QLabel(text.upper())
        label.setObjectName("sectionLabel")
        parent_layout.addWidget(label)
        parent_layout.addSpacing(4)
    
    def _create_voice_tab(self) -> QWidget:
        """Create the Voice settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 16, 20, 20)
        layout.setSpacing(4)
        
        # TTS Section
        self._create_section("Text-to-Speech", layout)
        
        self._voice_style = QComboBox()
        self._voice_style.addItems(["Normal", "Robot", "Claptrap", "GLaDOS", "Metallic"])
        self._voice_style.currentTextChanged.connect(self._on_voice_style_changed)
        self._create_field("Voice Style", self._voice_style, layout)
        
        # Speed with value display
        speed_row = QWidget()
        speed_layout = QHBoxLayout(speed_row)
        speed_layout.setContentsMargins(0, 0, 0, 0)
        speed_layout.setSpacing(12)
        
        self._tts_speed = QSlider(Qt.Orientation.Horizontal)
        self._tts_speed.setRange(50, 150)
        self._tts_speed.setValue(100)
        self._tts_speed.valueChanged.connect(self._on_tts_speed_changed)
        speed_layout.addWidget(self._tts_speed, 1)
        
        self._speed_label = QLabel("100%")
        self._speed_label.setFixedWidth(45)
        self._speed_label.setStyleSheet("color: rgba(255,255,255,0.5);")
        speed_layout.addWidget(self._speed_label)
        
        self._create_field("Speed", speed_row, layout)
        
        # STT Section
        self._create_section("Speech Recognition", layout)
        
        self._stt_model = QComboBox()
        available_models = WhisperSTT.list_available_models()
        if available_models:
            self._stt_model.addItems([m.capitalize() for m in available_models])
        else:
            self._stt_model.addItem("No models found")
            self._stt_model.setEnabled(False)
        self._stt_model.currentTextChanged.connect(self._on_stt_model_changed)
        self._create_field("Whisper Model", self._stt_model, layout)
        
        # Hotkey Section
        self._create_section("Controls", layout)
        
        self._ptt_key = QLineEdit()
        self._ptt_key.setReadOnly(True)
        self._ptt_key.setPlaceholderText("Click to set key...")
        self._ptt_key.mousePressEvent = self._start_key_capture
        self._create_field("Push-to-Talk Key", self._ptt_key, layout)
        
        layout.addStretch()
        return tab
    
    def _create_ai_tab(self) -> QWidget:
        """Create the AI settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 16, 20, 20)
        layout.setSpacing(4)
        
        # Backend Selection
        self._create_section("AI Source", layout)
        self._ai_backend = QComboBox()
        self._ai_backend.addItems(["Ollama (Local)", "OpenRouter (Cloud)"])
        self._ai_backend.currentIndexChanged.connect(self._on_backend_changed)
        self._create_field("Backend", self._ai_backend, layout)

        # Ollama Section (visible only when Ollama is selected)
        self._ollama_container = QWidget()
        ollama_layout = QVBoxLayout(self._ollama_container)
        ollama_layout.setContentsMargins(0, 0, 0, 0)
        ollama_layout.setSpacing(4)
        
        # Redundant header removed
        
        model_row = QWidget()
        model_layout = QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(8)
        
        self._ai_model = QComboBox()
        self._ai_model.currentTextChanged.connect(self._on_ai_model_changed)
        model_layout.addWidget(self._ai_model, 1)
        
        refresh_btn = QPushButton("↻")
        refresh_btn.setObjectName("accentButton")
        refresh_btn.setFixedWidth(36)
        refresh_btn.clicked.connect(self._refresh_ai_models)
        model_layout.addWidget(refresh_btn)
        
        self._create_field("Local Model", model_row, ollama_layout)
        layout.addWidget(self._ollama_container)

        # OpenRouter Section (visible only when OpenRouter is selected)
        self._openrouter_container = QWidget()
        or_layout = QVBoxLayout(self._openrouter_container)
        or_layout.setContentsMargins(0, 0, 0, 0)
        or_layout.setSpacing(4)

        # Redundant header removed
        
        self._openrouter_api_key = QLineEdit()
        self._openrouter_api_key.setPlaceholderText("Enter OpenRouter API key...")
        self._openrouter_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._openrouter_api_key.textChanged.connect(self._on_openrouter_key_changed)
        self._create_field("API Key", self._openrouter_api_key, or_layout)

        self._openrouter_model = QLineEdit()
        self._openrouter_model.setPlaceholderText("e.g., google/gemini-2.0-flash-exp:free")
        self._openrouter_model.textChanged.connect(self._on_openrouter_model_changed)
        self._create_field("Cloud Model ID", self._openrouter_model, or_layout)
        
        layout.addWidget(self._openrouter_container)
        
        # Personality
        self._personality = QTextEdit()
        self._personality.setFixedHeight(100)
        self._personality.setPlaceholderText("Enter AI personality/system prompt...")
        self._personality.textChanged.connect(self._on_personality_changed)
        self._create_field("Personality Prompt", self._personality, layout)
        
        # Web Search Section removed - moved to Tools tab
        
        layout.addStretch()
        
        # Load models
        self._refresh_ai_models()
        
        return tab

    def _create_tools_tab(self) -> QWidget:
        """Create the Tools settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 16, 20, 20)
        layout.setSpacing(4)
        
        self._create_section("Available Tools", layout)
        
        # Search Tool Item
        search_item = QFrame()
        search_item.setObjectName("toolItem")
        search_layout = QVBoxLayout(search_item)
        search_layout.setContentsMargins(0, 0, 0, 0)
        search_layout.setSpacing(0)
        
        # Header (Name, Toggle, Settings)
        header = QWidget()
        header.setObjectName("toolHeader")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(16, 12, 12, 12)
        
        info_v = QVBoxLayout()
        info_v.setSpacing(2)
        name_label = QLabel("Web Search")
        name_label.setObjectName("toolName")
        desc_label = QLabel("Real-time info via Brave Search")
        desc_label.setObjectName("toolDesc")
        info_v.addWidget(name_label)
        info_v.addWidget(desc_label)
        h_layout.addLayout(info_v, 1)
        
        self._enable_search = QCheckBox()
        self._enable_search.setObjectName("toolToggle")
        self._enable_search.setCursor(Qt.CursorShape.PointingHandCursor)
        self._enable_search.stateChanged.connect(self._on_search_enabled_changed)
        h_layout.addWidget(self._enable_search)
        
        self._search_settings_btn = QPushButton("SETTINGS")
        self._search_settings_btn.setObjectName("toolSettingsBtn")
        self._search_settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._search_settings_btn.clicked.connect(self._toggle_search_settings)
        h_layout.addWidget(self._search_settings_btn)
        
        search_layout.addWidget(header)
        
        # Expandable Settings Area
        self._search_settings_area = QWidget()
        self._search_settings_area.setObjectName("toolSettingsArea")
        self._search_settings_area.setVisible(False)
        area_layout = QVBoxLayout(self._search_settings_area)
        area_layout.setContentsMargins(16, 12, 16, 12)
        
        self._brave_api_key = QLineEdit()
        self._brave_api_key.setPlaceholderText("Enter Brave Search API key...")
        self._brave_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._brave_api_key.textChanged.connect(self._on_brave_key_changed)
        self._create_field("Brave API Key", self._brave_api_key, area_layout)
        
        search_layout.addWidget(self._search_settings_area)
        layout.addWidget(search_item)
        
        layout.addStretch()
        return tab
    
    def _create_display_tab(self) -> QWidget:
        """Create the Display settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 16, 20, 20)
        layout.setSpacing(4)
        
        # Avatar Section
        self._create_section("Avatar", layout)
        
        self._avatar_size = QComboBox()
        self._avatar_size.addItems(["Small (150px)", "Medium (200px)", "Large (250px)"])
        self._avatar_size.currentIndexChanged.connect(self._on_avatar_size_changed)
        self._create_field("Size (requires restart)", self._avatar_size, layout)
        
        self._display_color = QComboBox()
        self._display_color.addItems([t.capitalize() for t in THEMES.keys()])
        self._display_color.currentTextChanged.connect(self._on_display_color_changed)
        self._create_field("Color Theme", self._display_color, layout)
        
        # Opacity with value display
        opacity_row = QWidget()
        opacity_layout = QHBoxLayout(opacity_row)
        opacity_layout.setContentsMargins(0, 0, 0, 0)
        opacity_layout.setSpacing(12)
        
        self._opacity = QSlider(Qt.Orientation.Horizontal)
        self._opacity.setRange(50, 100)
        self._opacity.setValue(85)
        self._opacity.valueChanged.connect(self._on_opacity_changed)
        opacity_layout.addWidget(self._opacity, 1)
        
        self._opacity_label = QLabel("85%")
        self._opacity_label.setFixedWidth(45)
        self._opacity_label.setStyleSheet("color: rgba(255,255,255,0.5);")
        opacity_layout.addWidget(self._opacity_label)
        
        self._create_field("Opacity", opacity_row, layout)
        
        # About Section
        layout.addSpacing(24)
        self._create_section("About", layout)
        
        version_label = QLabel("TwoTwo v1.0.0")
        version_label.setStyleSheet("color: rgba(255, 255, 255, 0.4); font-size: 11px;")
        layout.addWidget(version_label)
        
        desc_label = QLabel("Local AI Assistant with Voice")
        desc_label.setStyleSheet("color: rgba(255, 255, 255, 0.3); font-size: 11px;")
        layout.addWidget(desc_label)
        
        layout.addStretch()
        return tab
    
    def _load_settings(self):
        """Load current settings into UI."""
        # Voice style
        style = self.config.get("voice", "voice_style", default="robot")
        style_map = {"normal": 0, "robot": 1, "claptrap": 2, "glados": 3, "metallic": 4}
        self._voice_style.setCurrentIndex(style_map.get(style.lower(), 1))
        
        # TTS speed
        speed = self.config.get("voice", "tts_speed", default=1.0)
        speed_val = int(speed * 100)
        self._tts_speed.setValue(speed_val)
        self._speed_label.setText(f"{speed_val}%")
        
        # STT model
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
        
        # Display color
        color = self.config.get("ui", "display_color", default="amber")
        idx = self._display_color.findText(color.capitalize())
        if idx >= 0:
            self._display_color.setCurrentIndex(idx)
        
        # Opacity
        opacity = self.config.get("ui", "opacity", default=0.85)
        opacity_val = int(opacity * 100)
        self._opacity.setValue(opacity_val)
        self._opacity_label.setText(f"{opacity_val}%")
        
        # AI Settings
        backend = self.config.get("ai", "backend", default="ollama")
        self._ai_backend.setCurrentIndex(1 if backend == "openrouter" else 0)
        self._update_backend_visibility(backend)

        ai_model = self.config.get("ai", "model", default="llama3.2:3b")
        idx = self._ai_model.findText(ai_model)
        if idx >= 0:
            self._ai_model.setCurrentIndex(idx)

        or_key = self.config.get("ai", "openrouter_api_key", default="")
        self._openrouter_api_key.setText(or_key)

        or_model = self.config.get("ai", "openrouter_model", default="google/gemini-2.0-flash-exp:free")
        self._openrouter_model.setText(or_model)
        
        # Personality
        personality = self.config.get("ai", "personality", default="")
        self._personality.setPlainText(personality)
        
        # Search settings
        enable_search = self.config.get("ai", "enable_search", default=True)
        self._enable_search.setChecked(enable_search)
        
        brave_key = self.config.get("ai", "brave_api_key", default="")
        self._brave_api_key.setText(brave_key)
        
    # Event handlers
    
    def _toggle_search_settings(self):
        """Toggle visibility of search tool settings."""
        visible = self._search_settings_area.isVisible()
        self._search_settings_area.setVisible(not visible)
        # Update button appearance
        if not visible:
            self._search_settings_btn.setStyleSheet("color: white; background: rgba(255, 255, 255, 0.1);")
        else:
            self._search_settings_btn.setStyleSheet("")
    
    def _on_voice_style_changed(self, text: str):
        style = text.lower()
        self.config.set("voice", "voice_style", style)
        self.settings_changed.emit("voice_style", style)
    
    def _on_tts_speed_changed(self, value: int):
        self._speed_label.setText(f"{value}%")
        speed = value / 100.0
        self.config.set("voice", "tts_speed", speed)
        self.settings_changed.emit("tts_speed", speed)
    
    def _on_stt_model_changed(self, text: str):
        model = text.lower()
        self.config.set("voice", "stt_model", model)
        self.settings_changed.emit("stt_model", model)
    
    def _on_avatar_size_changed(self, index: int):
        sizes = ["small", "medium", "large"]
        if 0 <= index < len(sizes):
            size = sizes[index]
            self.config.set("ui", "avatar_size", size)
            self.settings_changed.emit("avatar_size", size)
            
    def _on_display_color_changed(self, text: str):
        color = text.lower()
        self.config.set("ui", "display_color", color)
        self.settings_changed.emit("display_color", color)
    
    def _on_opacity_changed(self, value: int):
        self._opacity_label.setText(f"{value}%")
        opacity = value / 100.0
        self.config.set("ui", "opacity", opacity)
        self.settings_changed.emit("opacity", opacity)
    
    def _refresh_ai_models(self):
        manager = get_model_manager()
        
        if not manager.is_ollama_running():
            if manager.is_ollama_installed():
                print("Starting Ollama...")
                manager.start_ollama()
            else:
                self._ai_model.clear()
                self._ai_model.addItem("Ollama not installed")
                self._ai_model.setEnabled(False)
                return
        
        models = manager.get_model_names(refresh=True)
        current = self._ai_model.currentText()
        self._ai_model.clear()
        
        if models:
            self._ai_model.addItems(models)
            idx = self._ai_model.findText(current)
            if idx >= 0:
                self._ai_model.setCurrentIndex(idx)
            self._ai_model.setEnabled(True)
        else:
            self._ai_model.addItem("No models found")
            self._ai_model.setEnabled(False)
    
    def _on_ai_model_changed(self, text: str):
        if text and not text.startswith("No models") and not text.startswith("Ollama"):
            self.config.set("ai", "model", text)
            self.settings_changed.emit("ai_model", text)

    def _on_backend_changed(self, index: int):
        backend = "openrouter" if index == 1 else "ollama"
        self._update_backend_visibility(backend)
        self.config.set("ai", "backend", backend)
        self.settings_changed.emit("ai_backend", backend)

    def _update_backend_visibility(self, backend: str):
        is_or = backend == "openrouter"
        self._ollama_container.setVisible(not is_or)
        self._openrouter_container.setVisible(is_or)

    def _on_openrouter_key_changed(self, text: str):
        self.config.set("ai", "openrouter_api_key", text)
        self.settings_changed.emit("openrouter_api_key", text)

    def _on_openrouter_model_changed(self, text: str):
        self.config.set("ai", "openrouter_model", text)
        self.settings_changed.emit("ai_model", text)
    
    def _on_personality_changed(self):
        text = self._personality.toPlainText()
        self.config.set("ai", "personality", text)
        self.settings_changed.emit("personality", text)
    
    def _on_search_enabled_changed(self, state: int):
        enabled = state == Qt.CheckState.Checked.value
        self.config.set("ai", "enable_search", enabled)
        self.settings_changed.emit("enable_search", enabled)
    
    def _on_brave_key_changed(self, text: str):
        self.config.set("ai", "brave_api_key", text)
        try:
            from ai.search import get_search
            get_search().set_api_key(text)
        except Exception:
            pass
    
    def _start_key_capture(self, event):
        self._ptt_key.setText("Press a key...")
        self._ptt_key.setFocus()
    
    def keyPressEvent(self, event):
        if self._ptt_key.hasFocus():
            key = event.text().lower() if event.text() else None
            if key:
                self._ptt_key.setText(key.upper())
                self.config.set("voice", "hotkey", key)
                self.settings_changed.emit("hotkey", key)
                self._ptt_key.clearFocus()
        else:
            super().keyPressEvent(event)
    
    def _on_title_mouse_press(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.pos()
    
    def _on_title_mouse_move(self, event):
        if self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
    
    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        super().mouseReleaseEvent(event)
    
    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

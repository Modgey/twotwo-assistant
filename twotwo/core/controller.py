"""TwoTwo Core Controller

Main application controller that coordinates all components.
"""

import threading
from PySide6.QtCore import QObject, Slot, Signal, QTimer
from PySide6.QtWidgets import QApplication

from config import get_config
from core.state import AppState, AvatarState
from core.hotkey import HotkeyHandler
from ui.overlay_window import OverlayWindow
from ui.tray_icon import TrayIcon
from ui.settings_window import SettingsWindow


class Controller(QObject):
    """Main application controller."""
    
    # Signals for voice processing
    transcription_complete = Signal(str)
    
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
        self._settings: SettingsWindow | None = None
        
        # Voice components (lazy loaded)
        self._hotkey: HotkeyHandler | None = None
        self._recorder = None
        self._stt = None
        self._tts = None
        self._tts_queue = None
        self._player = None
        
        # Voice state
        self._voice_enabled = False
        self._processing_lock = threading.Lock()
        
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
        
        # Create settings window (hidden initially)
        self._settings = SettingsWindow()
        self._settings.settings_changed.connect(self._on_setting_changed)
        self._settings.hide()
        
        # Setup voice components
        self._setup_voice()
    
    def _setup_voice(self):
        """Initialize voice components."""
        try:
            from voice.audio import AudioRecorder, AudioPlayer
            from voice.stt import WhisperSTT
            from voice.tts import PiperTTS, TTSQueue
            
            # Get config
            ptt_key = self._config.get("voice", "hotkey", default="x")
            stt_model = self._config.get("voice", "stt_model", default="base")
            tts_voice = self._config.get("voice", "tts_voice", default="en_US-lessac-medium")
            tts_speed = self._config.get("voice", "tts_speed", default=1.0)
            
            # Initialize hotkey handler
            self._hotkey = HotkeyHandler(ptt_key=ptt_key)
            self._hotkey.ptt_pressed.connect(self._on_ptt_pressed)
            self._hotkey.ptt_released.connect(self._on_ptt_released)
            self._hotkey.cancel_pressed.connect(self._on_cancel_pressed)
            
            # Initialize audio recorder with amplitude callback
            self._recorder = AudioRecorder(
                amplitude_callback=self._on_recording_amplitude
            )
            
            # Initialize audio player with amplitude callback
            self._player = AudioPlayer(
                amplitude_callback=self._on_playback_amplitude
            )
            
            # Initialize STT
            self._stt = WhisperSTT(model_name=stt_model)
            if not self._stt.is_available():
                print("Warning: Whisper STT not available")
                print(f"Download models to: {self._stt.model_dir}")
            
            # Initialize TTS
            self._tts = PiperTTS(voice=tts_voice, speed=tts_speed)
            if not self._tts.is_available():
                print("Warning: Piper TTS not available")
                print("Download Piper from: https://github.com/rhasspy/piper/releases")
            
            # Initialize TTS queue with robot voice
            voice_style = self._config.get("voice", "voice_style", default="claptrap")
            self._tts_queue = TTSQueue(
                tts=self._tts,
                amplitude_callback=self._on_playback_amplitude,
                on_speaking_start=self._on_speaking_start,
                on_speaking_end=self._on_speaking_end,
                voice_style=voice_style,
            )
            
            self._voice_enabled = True
            print("Voice components initialized")
            
        except ImportError as e:
            print(f"Voice components not available: {e}")
            print("Install with: pip install sounddevice pywhispercpp pynput")
            self._voice_enabled = False
        except Exception as e:
            print(f"Voice initialization error: {e}")
            self._voice_enabled = False
    
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
        
        # Internal signals
        self.transcription_complete.connect(self._on_transcription_complete)
    
    @Slot()
    def _on_quit_requested(self):
        """Handle quit request from tray menu."""
        self._app_state = AppState.SHUTTING_DOWN
        self._cleanup()
        self._app.quit()
    
    @Slot()
    def _on_settings_requested(self):
        """Handle settings request from tray menu."""
        if self._settings:
            if self._settings.isVisible():
                self._settings.hide()
            else:
                # Position near the overlay
                if self._overlay:
                    pos = self._overlay.pos()
                    self._settings.move(pos.x() + self._overlay.width() + 10, pos.y())
                self._settings.show()
                self._settings.raise_()
    
    @Slot(str, object)
    def _on_setting_changed(self, key: str, value):
        """Handle setting changes from settings panel."""
        print(f"Setting changed: {key} = {value}")
        
        if key == "voice_style" and self._tts_queue:
            self._tts_queue.set_voice_style(value)
        elif key == "tts_speed" and self._tts:
            self._tts.speed = value
        elif key == "hotkey" and self._hotkey:
            self._hotkey.set_ptt_key(value)
    
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
    
    # Voice handling methods
    
    @Slot()
    def _on_ptt_pressed(self):
        """Handle push-to-talk pressed."""
        if not self._voice_enabled or not self._recorder:
            return
        
        with self._processing_lock:
            # Don't allow recording if already recording
            if self._recorder.is_recording():
                return
            
            # If we're in a non-idle state, cancel it first
            if self._avatar_state != AvatarState.IDLE and self._avatar_state != AvatarState.LISTENING:
                # Cancel current operation
                if self._tts_queue:
                    self._tts_queue.clear()
                if self._player:
                    self._player.stop()
            
            # Clear previous text
            if self._overlay:
                self._overlay.clear_text()
            
            # Start recording
            self.set_avatar_state(AvatarState.LISTENING)
            self._recorder.start()
            print("Recording started...")
    
    @Slot()
    def _on_ptt_released(self):
        """Handle push-to-talk released."""
        if not self._voice_enabled or not self._recorder:
            return
        
        if not self._recorder.is_recording():
            return
        
        # Stop recording and get audio
        audio = self._recorder.stop()
        print(f"Recording stopped. Got {len(audio)} samples")
        
        # Check minimum duration (0.3 seconds at 16kHz = 4800 samples)
        min_samples = int(0.3 * 16000)
        if len(audio) < min_samples:
            print(f"Recording too short ({len(audio)} < {min_samples}), ignoring")
            self.set_avatar_state(AvatarState.IDLE)
            return
        
        # Check if audio is mostly silence (very low threshold)
        import numpy as np
        rms = np.sqrt(np.mean(audio ** 2))
        silence_threshold = 0.001  # Very low - only blocks completely silent recordings
        
        print(f"Audio RMS: {rms:.6f} (threshold: {silence_threshold})")
        
        if rms < silence_threshold:
            print(f"Recording is completely silent (RMS: {rms:.6f}), ignoring")
            self.set_avatar_state(AvatarState.IDLE)
            return
        
        # Valid audio - start transcription
        print(f"Valid audio detected, transcribing...")
        self.set_avatar_state(AvatarState.THINKING)
        self._transcribe_audio(audio)
    
    @Slot()
    def _on_cancel_pressed(self):
        """Handle cancel (Escape) pressed."""
        if not self._voice_enabled:
            return
        
        # Stop any ongoing operation
        if self._recorder and self._recorder.is_recording():
            self._recorder.stop()
        
        if self._tts_queue:
            self._tts_queue.clear()
        
        if self._player:
            self._player.stop()
        
        self.set_avatar_state(AvatarState.IDLE)
        print("Operation cancelled")
    
    def _transcribe_audio(self, audio):
        """Transcribe audio in background thread."""
        if not self._stt:
            self.set_avatar_state(AvatarState.IDLE)
            return
        
        def on_complete(text: str):
            # Emit signal to handle on main thread
            self.transcription_complete.emit(text)
        
        self._stt.transcribe_async(audio, callback=on_complete)
    
    @Slot(str)
    def _on_transcription_complete(self, text: str):
        """Handle completed transcription."""
        try:
            # Check for empty, blank, or [blank audio] responses
            if not text.strip() or text.strip().lower() in ["[blank_audio]", "[blank audio]", "[blank]"]:
                print(f"No speech detected (got: '{text}')")
                self.set_avatar_state(AvatarState.IDLE)
                if self._overlay:
                    self._overlay.clear_text()
                return
            
            print(f"Transcribed: {text}")
            
            # For now, just echo back the text via TTS
            # In Phase 4, this will go to the LLM
            if self._tts_queue and self._tts and self._tts.is_available():
                response = f"You said: {text}"
                
                # Show AI response text next to avatar
                if self._overlay:
                    self._overlay.show_text(response)
                
                self._tts_queue.speak(response)
            else:
                print("TTS not available, returning to idle")
                self.set_avatar_state(AvatarState.IDLE)
        except Exception as e:
            print(f"Error handling transcription: {e}")
            self.set_avatar_state(AvatarState.IDLE)
    
    def _on_recording_amplitude(self, amplitude: float):
        """Handle recording amplitude update."""
        # Could be used for visual feedback during recording
        pass
    
    def _on_playback_amplitude(self, amplitude: float):
        """Handle playback amplitude for avatar sync."""
        self.set_audio_amplitude(amplitude)
    
    def _on_speaking_start(self):
        """Handle TTS speaking start."""
        self.set_avatar_state(AvatarState.SPEAKING)
    
    def _on_speaking_end(self):
        """Handle TTS speaking end."""
        self.set_avatar_state(AvatarState.IDLE)
        
        # Hide text after a delay
        if self._overlay:
            duration = self._config.get("ui", "text_display_duration", default=5.0)
            from PySide6.QtCore import QTimer
            # Hide both user text and response text after delay
            QTimer.singleShot(int(duration * 1000), self._hide_all_text)
    
    def _hide_all_text(self):
        """Hide response text."""
        if self._overlay:
            self._overlay._hide_text()
    
    def _cleanup(self):
        """Clean up resources before shutdown."""
        # Stop voice components
        if self._hotkey:
            self._hotkey.stop()
        
        if self._tts_queue:
            self._tts_queue.stop()
        
        if self._recorder and self._recorder.is_recording():
            self._recorder.stop()
        
        if self._player:
            self._player.stop()
        
        # Stop UI
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
        
        # Start voice components
        if self._voice_enabled:
            if self._hotkey:
                self._hotkey.start()
            if self._tts_queue:
                self._tts_queue.start()
            
            ptt_key = self._config.get("voice", "hotkey", default="x").upper()
            print(f"Voice enabled. Hold '{ptt_key}' to talk.")
    
    @property
    def app_state(self) -> AppState:
        """Get current application state."""
        return self._app_state
    
    @property
    def avatar_state(self) -> AvatarState:
        """Get current avatar state."""
        return self._avatar_state
    
    @property
    def voice_enabled(self) -> bool:
        """Check if voice is enabled."""
        return self._voice_enabled
    
    def set_avatar_state(self, state: AvatarState):
        """Set the avatar animation state."""
        self._avatar_state = state
        if self._overlay:
            self._overlay.set_avatar_state(state)
    
    def set_audio_amplitude(self, amplitude: float):
        """Set audio amplitude for speaking animation (0-1)."""
        if self._overlay:
            self._overlay.set_audio_amplitude(amplitude)
    
    def speak(self, text: str):
        """Speak text using TTS."""
        if self._tts_queue and self._voice_enabled:
            self._tts_queue.speak(text)

"""TwoTwo Core Controller

Main application controller that coordinates all components.
"""

import threading
import time
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
    
    # Signals for voice processing (for thread safety)
    transcription_complete = Signal(str)
    speaking_finished = Signal()
    llm_response_ready = Signal(str)  # Full response from LLM
    llm_token_received = Signal(str)  # Streaming token from LLM
    llm_error = Signal(str)  # LLM error
    
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
        
        # AI components
        self._llm = None
        self._tool_manager = None
        self._ai_enabled = False
        
        # Voice state
        self._voice_enabled = False
        self._processing_lock = threading.Lock()
        self._current_response = ""  # Accumulates streaming response for display
        self._sentence_buffer = ""   # Accumulates tokens until sentence boundary
        self._speaking_started = False  # Track if TTS has started for this response
        
        # Performance timing
        self._timing = {
            "ptt_released": 0.0,
            "stt_start": 0.0,
            "stt_end": 0.0,
            "llm_start": 0.0,
            "llm_first_token": 0.0,
            "llm_end": 0.0,
            "tts_first_audio": 0.0,
            "tts_end": 0.0,
        }
        
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
        
        # Setup AI components
        self._setup_ai()
    
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
    
    def _setup_ai(self):
        """Initialize AI components (LLM and tools)."""
        try:
            from ai.llm import OllamaLLM
            from ai.model_manager import get_model_manager
            from ai.tools import get_tool_manager
            
            # Get model manager and ensure Ollama is running
            manager = get_model_manager()
            
            if not manager.is_ollama_installed():
                print("Ollama not installed. Download from https://ollama.ai")
                self._ai_enabled = False
                return
            
            # Auto-start Ollama if not running
            if not manager.ensure_running(auto_start=True):
                print("Could not start Ollama")
                self._ai_enabled = False
                return
            
            # Get configured model - just use what's in the config (persisted from last use)
            configured_model = self._config.get("ai", "model", default="llama3.2:3b")
            
            # Check if configured model is available
            available_models = manager.get_model_names(refresh=True)
            
            if not available_models:
                print("No Ollama models found. Pull one with: ollama pull llama3.2:3b")
                self._ai_enabled = False
                return
            
            if configured_model not in available_models:
                print(f"Warning: Configured model '{configured_model}' not found.")
                print(f"Available models: {', '.join(available_models[:5])}")
                print(f"Using first available: {available_models[0]}")
                configured_model = available_models[0]
                # Note: We don't save this to config - user's preference is preserved
            
            # Initialize LLM with the configured model
            self._llm = OllamaLLM(model=configured_model)
            
            if self._llm.is_available():
                print(f"AI initialized with model: {self._llm.model}")
                self._ai_enabled = True
                # Pre-warm the model to reduce first-response latency
                self._llm.warm_up()
            else:
                print("Warning: Ollama connection failed")
                self._ai_enabled = False
                return
            
            # Initialize tool manager
            self._tool_manager = get_tool_manager()
            
            # Add tools to system prompt template
            tools_prompt = self._tool_manager.get_system_prompt_addition()
            if tools_prompt:
                self._llm._tools_prompt = tools_prompt.strip()
                enabled_tools = [t.name for t in self._tool_manager.get_enabled_tools()]
                print(f"Tools enabled: {', '.join(enabled_tools)}")
            
            # Initialize search intent detector (uses tiny model for classification)
            try:
                from ai.search_intent import get_search_intent_detector
                detector = get_search_intent_detector()
                if detector.is_available():
                    print("Smart search intent detection enabled")
            except ImportError:
                pass
            
        except ImportError as e:
            print(f"AI components not available: {e}")
            self._ai_enabled = False
        except Exception as e:
            print(f"AI initialization error: {e}")
            self._ai_enabled = False
    
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
        self.speaking_finished.connect(self._on_speaking_finished)
        self.llm_response_ready.connect(self._on_llm_response_ready)
        self.llm_token_received.connect(self._on_llm_token_received)
        self.llm_error.connect(self._on_llm_error)
    
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
                # Position near the overlay, clamped to screen
                if self._overlay:
                    from PySide6.QtWidgets import QApplication
                    screen = QApplication.primaryScreen().availableGeometry()
                    
                    # Try to position to the right of overlay
                    x = self._overlay.pos().x() + self._overlay.width() + 10
                    y = self._overlay.pos().y()
                    
                    # Clamp to screen bounds
                    x = min(x, screen.right() - self._settings.width())
                    x = max(x, screen.left())
                    y = min(y, screen.bottom() - self._settings.height())
                    y = max(y, screen.top())
                    
                    self._settings.move(x, y)
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
        elif key == "stt_model":
            from voice.stt import WhisperSTT
            self._stt = WhisperSTT(model_name=value)
            print(f"Switched to STT model: {value}")
        elif key == "opacity" and self._overlay:
            self._overlay.set_opacity(value)
        elif key == "avatar_size":
            print(f"Avatar size change requires restart to take effect")
        elif key == "ai_model" and self._llm:
            self._llm.set_model(value)
            self._config.set("ai", "model", value)  # Ensure it's saved to config
            self._llm.warm_up()  # Pre-warm the new model
            print(f"Switched to AI model: {value} (saved to config)")
        elif key == "personality" and self._llm:
            self._llm.set_system_prompt(value)
            print("Updated AI personality")
        elif key.startswith("enable_") and self._tool_manager:
            # Handle tool enable/disable (e.g., enable_search)
            tool_name = key.replace("enable_", "")
            tool = self._tool_manager.get_tool(tool_name)
            if tool:
                tool.set_enabled(value)
                print(f"Tool '{tool_name}' {'enabled' if value else 'disabled'}")
    
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
        
        # Mark timing: user stopped speaking
        self._timing["ptt_released"] = time.time()
        
        # Stop recording and get audio
        audio = self._recorder.stop()
        print(f"Recording stopped. Got {len(audio)} samples")
        
        # Check minimum duration (0.15 seconds at 16kHz = 2400 samples)
        min_samples = int(0.15 * 16000)
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
        
        # Mark timing: STT start
        self._timing["stt_start"] = time.time()
        
        def on_complete(text: str):
            # Mark timing: STT end
            self._timing["stt_end"] = time.time()
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
            
            # Log STT timing
            stt_duration = self._timing["stt_end"] - self._timing["stt_start"]
            print(f"[TIMING] STT took {stt_duration:.3f}s")
            print(f"Transcribed: {text}")
            
            # Send to LLM if available, otherwise echo
            if self._ai_enabled and self._llm:
                self._send_to_llm(text)
            elif self._tts_queue and self._tts and self._tts.is_available():
                # Fallback: echo back if no LLM
                response = f"You said: {text}"
                if self._overlay:
                    self._overlay.show_text(response)
                self._tts_queue.speak(response)
            else:
                print("Neither AI nor TTS available, returning to idle")
                self.set_avatar_state(AvatarState.IDLE)
        except Exception as e:
            print(f"Error handling transcription: {e}")
            self.set_avatar_state(AvatarState.IDLE)
    
    def _send_to_llm(self, user_message: str):
        """Send message to LLM with streaming.
        
        Runs search intent detection in PARALLEL - no added latency.
        If search results come back, they're used in a follow-up if needed.
        """
        self._current_response = ""
        self._sentence_buffer = ""
        self._speaking_started = False
        self._pending_search_results = None  # For parallel search
        self._search_detected = False  # Reset search detection flag
        
        # Mark timing: LLM start
        self._timing["llm_start"] = time.time()
        self._timing["llm_first_token"] = 0.0
        
        # Start parallel search intent detection (non-blocking)
        if self._tool_manager:
            self._start_parallel_search(user_message)
        
        def on_token(token: str):
            self.llm_token_received.emit(token)
        
        def on_complete(full_response: str):
            self.llm_response_ready.emit(full_response)
        
        def on_error(error: str):
            self.llm_error.emit(error)
        
        # Start main LLM immediately (no delay from search)
        print(f"Sending to LLM: {user_message[:50]}...")
        self._llm.chat_stream(
            user_message,
            on_token=on_token,
            on_complete=on_complete,
            on_error=on_error,
        )
    
    def _start_parallel_search(self, query: str):
        """Run search intent detection and search in background thread."""
        import threading
        
        def search_worker():
            try:
                from ai.search_intent import get_search_intent_detector
                detector = get_search_intent_detector()
                
                if not detector.is_available():
                    return
                
                if detector.needs_search(query):
                    print(f"[Parallel] Search needed for: {query[:40]}...")
                    search_tool = self._tool_manager.get_tool("search")
                    if search_tool and search_tool.is_enabled():
                        result = search_tool.execute(query)
                        if result.success:
                            self._pending_search_results = result.content
                            print(f"[Parallel] Search results ready")
            except Exception as e:
                print(f"[Parallel] Search error: {e}")
        
        thread = threading.Thread(target=search_worker, daemon=True)
        thread.start()
    
    @Slot(str)
    def _on_llm_token_received(self, token: str):
        """Handle streaming token from LLM with sentence-level TTS."""
        # Mark timing: first token received
        if self._timing["llm_first_token"] == 0.0:
            self._timing["llm_first_token"] = time.time()
            time_to_first_token = self._timing["llm_first_token"] - self._timing["llm_start"]
            print(f"[TIMING] LLM first token took {time_to_first_token:.3f}s")
        
        self._current_response += token
        self._sentence_buffer += token
        
        # Detect search tag early - stop TTS if we see it
        if '<search>' in self._current_response and not getattr(self, '_search_detected', False):
            self._search_detected = True
            print("[Stream] Search tag detected - clearing TTS, searching...")
            # Clear the sentence buffer so we don't speak the content after <search>
            self._sentence_buffer = ""
            # Clear pending TTS and speak acknowledgement
            if self._tts_queue:
                self._tts_queue.clear_queue()
            # Speak "Checking..." - this will also update the UI
            self._speak_sentence("Checking...")
            return
        
        # If search was detected, just accumulate but don't speak
        if getattr(self, '_search_detected', False):
            return
        
        # Check for sentence boundary to start speaking (UI updates in _speak_sentence)
        sentence = self._extract_complete_sentence()
        if sentence:
            self._speak_sentence(sentence)
    
    def _extract_complete_sentence(self) -> str | None:
        """Extract a speakable chunk from the buffer.
        
        Uses 'fast start' strategy:
        - FIRST chunk: Aggressive small thresholds to start speaking ASAP
        - SUBSEQUENT chunks: Larger thresholds for smooth, natural flow
        """
        import re
        
        buffer = self._sentence_buffer
        if not buffer:
            return None
        
        # === FAST START: Use smaller thresholds for first chunk ===
        # This minimizes time-to-first-audio while parallel synthesis
        # handles smooth flow for subsequent chunks
        if not self._speaking_started:
            MIN_LEN = 12  # Very short - speak almost anything
            
            # Sentence end - immediate
            sentence_end = re.search(r'[.!?](?:\s|$)', buffer)
            if sentence_end:
                end_pos = sentence_end.start() + 1
                sentence = buffer[:end_pos].strip()
                if len(sentence) >= MIN_LEN:
                    self._sentence_buffer = buffer[end_pos:].lstrip()
                    return sentence
            
            # Newline - immediate
            newline_match = re.search(r'\n', buffer)
            if newline_match and newline_match.start() >= MIN_LEN:
                end_pos = newline_match.start()
                sentence = buffer[:end_pos].strip()
                if sentence:
                    self._sentence_buffer = buffer[end_pos:].lstrip()
                    return sentence
            
            # Comma/pause - after just 20 chars
            if len(buffer) > 20:
                pause_match = re.search(r'[,;:\-–—](?:\s|$)', buffer)
                if pause_match and pause_match.start() >= 12:
                    end_pos = pause_match.start() + 1
                    sentence = buffer[:end_pos].strip()
                    self._sentence_buffer = buffer[end_pos:].lstrip()
                    return sentence
            
            # Word boundary - after 35 chars
            if len(buffer) > 35:
                space_match = re.search(r'\s', buffer[20:])
                if space_match:
                    end_pos = 20 + space_match.start()
                    sentence = buffer[:end_pos].strip()
                    self._sentence_buffer = buffer[end_pos:].lstrip()
                    return sentence
            
            # Hard cutoff at 45 chars
            if len(buffer) > 45:
                sentence = buffer[:35].strip()
                self._sentence_buffer = buffer[35:].lstrip()
                return sentence
            
            return None
        
        # === SMOOTH FLOW: Larger thresholds for subsequent chunks ===
        # Parallel synthesis is already pre-buffering, so we can wait for
        # more natural break points
        MIN_SPEAK_LEN = 15
        
        # Sentence-ending punctuation (highest priority)
        sentence_end = re.search(r'[.!?](?:\s|$)', buffer)
        if sentence_end:
            end_pos = sentence_end.start() + 1
            sentence = buffer[:end_pos].strip()
            if len(sentence) >= MIN_SPEAK_LEN:
                self._sentence_buffer = buffer[end_pos:].lstrip()
                return sentence
        
        # Newlines
        newline_match = re.search(r'\n', buffer)
        if newline_match and newline_match.start() >= MIN_SPEAK_LEN:
            end_pos = newline_match.start()
            sentence = buffer[:end_pos].strip()
            if sentence:
                self._sentence_buffer = buffer[end_pos:].lstrip()
                return sentence
        
        # Pause punctuation - wait for 45+ chars
        if len(buffer) > 45:
            pause_match = re.search(r'[,;:\-–—](?:\s|$)', buffer)
            if pause_match and pause_match.start() >= 25:
                end_pos = pause_match.start() + 1
                sentence = buffer[:end_pos].strip()
                self._sentence_buffer = buffer[end_pos:].lstrip()
                return sentence
        
        # Word boundary fallback - after 70 chars
        if len(buffer) > 70:
            space_match = re.search(r'\s', buffer[45:])
            if space_match:
                end_pos = 45 + space_match.start()
                sentence = buffer[:end_pos].strip()
                self._sentence_buffer = buffer[end_pos:].lstrip()
                return sentence
        
        # Hard cutoff at 90 chars
        if len(buffer) > 90:
            sentence = buffer[:70].strip()
            self._sentence_buffer = buffer[70:].lstrip()
            return sentence
        
        return None
    
    def _clean_for_tts(self, text: str) -> str:
        """Clean text for TTS - remove markdown, emojis, and formatting."""
        import re
        
        # Remove markdown bold/italic (***text***, **text**, *text*, __text__, _text_)
        text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
        text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)
        
        # Remove inline code backticks
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        
        # Remove headers (# ## ### etc)
        text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
        
        # Remove bullet points and numbered lists
        text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Remove links [text](url) -> text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove emojis (common unicode ranges)
        text = re.sub(r'[\U0001F600-\U0001F64F]', '', text)  # emoticons
        text = re.sub(r'[\U0001F300-\U0001F5FF]', '', text)  # symbols & pictographs
        text = re.sub(r'[\U0001F680-\U0001F6FF]', '', text)  # transport & map
        text = re.sub(r'[\U0001F1E0-\U0001F1FF]', '', text)  # flags
        text = re.sub(r'[\U00002702-\U000027B0]', '', text)  # dingbats
        text = re.sub(r'[\U0001F900-\U0001F9FF]', '', text)  # supplemental symbols
        
        # Remove other special characters that TTS might struggle with
        text = text.replace('…', '...')
        text = text.replace('—', ', ')
        text = text.replace('–', ', ')
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _speak_sentence(self, sentence: str):
        """Send a sentence to TTS and update UI.
        
        This is the single source of truth: UI shows what we speak.
        """
        # Filter out any tool tags before speaking
        if self._tool_manager:
            sentence = self._tool_manager.clean_tool_tags(sentence)
        
        # Clean markdown and formatting for TTS
        sentence = self._clean_for_tts(sentence)
        
        if not sentence:
            return
        
        if not self._tts_queue or not self._tts or not self._tts.is_available():
            return
        
        # Cancel any pending hide timers by incrementing counter
        self._hide_text_counter = getattr(self, '_hide_text_counter', 0) + 1
        
        # Update UI with what we're about to speak
        if self._overlay:
            self._overlay.show_text(sentence)
        
        # Mark timing: first audio sent to TTS
        if not self._speaking_started:
            self._speaking_started = True
            self._timing["tts_first_audio"] = time.time()
            time_to_first_audio = self._timing["tts_first_audio"] - self._timing["ptt_released"]
            print(f"[TIMING] Time to first audio: {time_to_first_audio:.3f}s (PTT release → TTS start)")
        
        print(f"TTS (streaming): '{sentence[:40]}...'")
        self._tts_queue.speak(sentence)
    
    @Slot(str)
    def _on_llm_response_ready(self, response: str):
        """Handle complete LLM response."""
        # Mark timing: LLM end
        self._timing["llm_end"] = time.time()
        llm_total_duration = self._timing["llm_end"] - self._timing["llm_start"]
        print(f"[TIMING] LLM total time: {llm_total_duration:.3f}s")
        print(f"LLM response complete: {response[:100]}...")
        
        # Check for tool calls and process
        if self._tool_manager:
            tool_call = self._tool_manager.extract_tool_call(response)
            if tool_call:
                tool_name, query = tool_call
                print(f"Tool requested: {tool_name} - {query}")
                
                # Note: Acknowledgement already spoken when search tag was detected during streaming
                
                # Execute the tool
                result = self._tool_manager.execute_tool(tool_name, query)
                
                if result.success:
                    print(f"Search executed successfully. Results received.")
                    
                    # Send tool results back to LLM - tell it NOT to repeat acknowledgement
                    follow_up_prompt = f"""Here are the search results:

{result.content}

Answer naturally with this information. Start directly with the answer - do NOT say things like 'Let me check' or 'Here's what I found'."""
                    
                    # Make a follow-up LLM call with the tool results
                    self._send_followup_with_results(follow_up_prompt)
                    return  # Wait for follow-up response
                else:
                    print(f"Tool error: {result.error}")
        
        # Check if parallel search found results that the LLM didn't use
        if hasattr(self, '_pending_search_results') and self._pending_search_results:
            # LLM responded without search data, but we have results ready
            print(f"[Parallel] Injecting search results into follow-up...")
            follow_up_prompt = f"""The user asked: {self._current_response}

Here is current information from the web:
{self._pending_search_results}

Please answer the user's original question using this real-time data. Be conversational and direct."""
            
            self._pending_search_results = None  # Clear it
            self._send_followup_with_results(follow_up_prompt)
            return  # Wait for follow-up
        
        # Clean any tool tags from response
        if self._tool_manager:
            response = self._tool_manager.clean_tool_tags(response)
        
        # Update display with cleaned response
        if self._overlay and response:
            self._overlay.show_text(response)
        
        # Speak any remaining text in the sentence buffer
        remaining = self._sentence_buffer.strip()
        if remaining:
            self._speak_sentence(remaining)
            self._sentence_buffer = ""
        
        # If nothing was spoken (very short response), speak the whole thing
        if not self._speaking_started and response.strip():
            if self._tts_queue and self._tts and self._tts.is_available():
                self._tts_queue.speak(response)
        
        # If TTS not available, go to idle
        if not self._tts_queue or not self._tts or not self._tts.is_available():
            print("TTS not available")
            self.set_avatar_state(AvatarState.IDLE)
    
    def _send_followup_with_results(self, follow_up_prompt: str):
        """Send tool results to LLM for a follow-up response."""
        if not self._llm:
            return
        
        # Reset for new response
        self._current_response = ""
        self._sentence_buffer = ""
        self._speaking_started = False
        self._search_detected = False  # Reset so follow-up tokens go to TTS
        
        def on_token(token: str):
            self.llm_token_received.emit(token)
        
        def on_complete(full_response: str):
            self.llm_response_ready.emit(full_response)
        
        def on_error(error: str):
            self.llm_error.emit(error)
        
        print(f"Sending follow-up to LLM with tool results...")
        self._llm.chat_stream(
            follow_up_prompt,
            on_token=on_token,
            on_complete=on_complete,
            on_error=on_error,
        )
    
    @Slot(str)
    def _on_llm_error(self, error: str):
        """Handle LLM error."""
        print(f"LLM error: {error}")
        error_msg = f"Sorry, I encountered an error: {error}"
        
        if self._overlay:
            self._overlay.show_text(error_msg)
        
        if self._tts_queue and self._tts and self._tts.is_available():
            self._tts_queue.speak("Sorry, I encountered an error.")
        else:
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
        """Handle TTS speaking end (called from background thread)."""
        # Emit signal to handle on main Qt thread
        self.speaking_finished.emit()
    
    @Slot()
    def _on_speaking_finished(self):
        """Handle speaking finished on main thread."""
        # Mark timing: TTS end
        self._timing["tts_end"] = time.time()
        
        # Print complete timing summary
        self._print_timing_summary()
        
        self.set_avatar_state(AvatarState.IDLE)
        
        # Hide text after a delay (counter ensures new speech cancels old hide timer)
        if self._overlay:
            duration = self._config.get("ui", "text_display_duration", default=5.0)
            self._hide_text_counter = getattr(self, '_hide_text_counter', 0) + 1
            expected = self._hide_text_counter
            QTimer.singleShot(int(duration * 1000), 
                lambda: self._hide_all_text() if self._hide_text_counter == expected else None)
    
    def _hide_all_text(self):
        """Hide response text."""
        if self._overlay:
            self._overlay._hide_text()
    
    def _print_timing_summary(self):
        """Print detailed timing breakdown of the interaction."""
        t = self._timing
        
        # Calculate stage durations
        stt_duration = t["stt_end"] - t["stt_start"] if t["stt_start"] > 0 else 0
        llm_first_token = t["llm_first_token"] - t["llm_start"] if t["llm_first_token"] > 0 else 0
        llm_total = t["llm_end"] - t["llm_start"] if t["llm_end"] > 0 else 0
        time_to_first_audio = t["tts_first_audio"] - t["ptt_released"] if t["tts_first_audio"] > 0 else 0
        tts_duration = t["tts_end"] - t["tts_first_audio"] if t["tts_end"] > 0 and t["tts_first_audio"] > 0 else 0
        total_duration = t["tts_end"] - t["ptt_released"] if t["tts_end"] > 0 else 0
        
        print("\n" + "="*60)
        print("TIMING SUMMARY")
        print("="*60)
        print(f"STT Processing:           {stt_duration:7.3f}s")
        print(f"LLM First Token:          {llm_first_token:7.3f}s")
        print(f"LLM Total (streaming):    {llm_total:7.3f}s")
        print(f"Time to First Audio:      {time_to_first_audio:7.3f}s ⚡ (KEY METRIC)")
        print(f"TTS Playback Duration:    {tts_duration:7.3f}s")
        print("-"*60)
        print(f"TOTAL (PTT → Speech End): {total_duration:7.3f}s")
        print("="*60 + "\n")
    
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
        
        # Print AI status
        if self._ai_enabled and self._llm:
            print(f"AI ready with model: {self._llm.model}")
        else:
            print("AI not available - start Ollama with: ollama serve")
    
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

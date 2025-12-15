"""TwoTwo Text-to-Speech

Piper TTS integration for local speech synthesis.
Supports streaming output for low latency.
"""

import subprocess
import threading
import queue
from pathlib import Path
from typing import Callable, Optional, Generator
import numpy as np
import struct

from config import get_config, get_config_dir


class PiperTTS:
    """Text-to-speech using Piper."""
    
    # Common voice models
    VOICES = {
        "en_US-lessac-medium": "en_US-lessac-medium.onnx",
        "en_US-lessac-high": "en_US-lessac-high.onnx",
        "en_US-amy-medium": "en_US-amy-medium.onnx",
        "en_US-ryan-medium": "en_US-ryan-medium.onnx",
        "en_GB-alan-medium": "en_GB-alan-medium.onnx",
    }
    
    def __init__(
        self,
        voice: str = "en_US-lessac-medium",
        voice_dir: Optional[Path] = None,
        speed: float = 1.0,
    ):
        self.voice = voice
        self.speed = speed
        
        # Voice directory
        if voice_dir:
            self.voice_dir = voice_dir
        else:
            self.voice_dir = get_config_dir() / "piper-voices"
        
        self._piper_path: Optional[Path] = None
        self._sample_rate = 22050  # Piper default
        
        self._find_piper()
    
    def _find_piper(self):
        """Find Piper executable."""
        config_dir = get_config_dir()
        
        # Check common locations
        possible_paths = [
            config_dir / "piper" / "piper" / "piper.exe",  # Extracted zip structure
            config_dir / "piper" / "piper.exe",
            config_dir / "piper" / "piper",
            Path("piper") / "piper.exe",
            Path("piper") / "piper",
            Path.home() / ".local" / "bin" / "piper",
        ]
        
        for path in possible_paths:
            if path.exists():
                self._piper_path = path
                print(f"Found Piper at {path}")
                return
        
        # Try to find in PATH
        try:
            import shutil
            piper_path = shutil.which("piper")
            if piper_path:
                self._piper_path = Path(piper_path)
                print(f"Found Piper in PATH: {piper_path}")
                return
        except Exception:
            pass
        
        print("Piper not found. Please install Piper TTS.")
    
    def _get_model_path(self) -> Optional[Path]:
        """Get the path to the voice model."""
        # Try exact voice name
        model_file = self.VOICES.get(self.voice, f"{self.voice}.onnx")
        model_path = self.voice_dir / model_file
        
        if model_path.exists():
            return model_path
        
        # Try without extension
        model_path = self.voice_dir / self.voice
        if model_path.exists():
            return model_path
        
        # Search for any matching model
        for path in self.voice_dir.glob(f"*{self.voice}*"):
            if path.suffix == ".onnx":
                return path
        
        print(f"Voice model not found: {self.voice}")
        print(f"Please download models to: {self.voice_dir}")
        return None
    
    def synthesize(self, text: str) -> Optional[np.ndarray]:
        """Synthesize text to audio.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data as numpy array (float32), or None on error
        """
        if not self._piper_path:
            return None
        
        model_path = self._get_model_path()
        if not model_path:
            return None
        
        try:
            # Run Piper with raw output
            process = subprocess.Popen(
                [
                    str(self._piper_path),
                    "--model", str(model_path),
                    "--output-raw",
                    "--length-scale", str(1.0 / self.speed),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            # Send text and get audio
            stdout, stderr = process.communicate(input=text.encode("utf-8"), timeout=30)
            
            if process.returncode != 0:
                print(f"Piper error: {stderr.decode()}")
                return None
            
            # Convert raw bytes to numpy array (16-bit PCM)
            audio = np.frombuffer(stdout, dtype=np.int16)
            audio = audio.astype(np.float32) / 32767.0
            
            return audio
            
        except Exception as e:
            print(f"TTS synthesis error: {e}")
            return None
    
    def synthesize_streaming(
        self,
        text: str,
        chunk_callback: Callable[[np.ndarray], None],
        on_complete: Optional[Callable[[], None]] = None,
    ) -> threading.Thread:
        """Synthesize text with streaming output.
        
        Args:
            text: Text to synthesize
            chunk_callback: Called with each audio chunk
            on_complete: Called when synthesis is complete
            
        Returns:
            Thread running the synthesis
        """
        def worker():
            if not self._piper_path:
                if on_complete:
                    on_complete()
                return
            
            model_path = self._get_model_path()
            if not model_path:
                if on_complete:
                    on_complete()
                return
            
            try:
                process = subprocess.Popen(
                    [
                        str(self._piper_path),
                        "--model", str(model_path),
                        "--output-raw",
                        "--length-scale", str(1.0 / self.speed),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                
                # Send text
                process.stdin.write(text.encode("utf-8"))
                process.stdin.close()
                
                # Read audio in chunks
                chunk_size = 4096  # bytes
                
                while True:
                    data = process.stdout.read(chunk_size)
                    if not data:
                        break
                    
                    # Convert to numpy
                    audio = np.frombuffer(data, dtype=np.int16)
                    audio = audio.astype(np.float32) / 32767.0
                    chunk_callback(audio)
                
                process.wait()
                
            except Exception as e:
                print(f"Streaming TTS error: {e}")
            
            if on_complete:
                on_complete()
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return thread
    
    def synthesize_to_file(self, text: str, output_path: Path) -> bool:
        """Synthesize text and save to WAV file.
        
        Args:
            text: Text to synthesize
            output_path: Path to save WAV file
            
        Returns:
            True on success, False on error
        """
        audio = self.synthesize(text)
        if audio is None:
            return False
        
        try:
            import wave
            
            audio_int16 = (audio * 32767).astype(np.int16)
            
            with wave.open(str(output_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            return True
            
        except Exception as e:
            print(f"Error saving WAV: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if Piper is available and ready."""
        return self._piper_path is not None and self._get_model_path() is not None
    
    @property
    def sample_rate(self) -> int:
        """Get the output sample rate."""
        return self._sample_rate
    
    @classmethod
    def list_available_voices(cls) -> list[str]:
        """List voice models that are downloaded."""
        config_dir = get_config_dir()
        voice_dir = config_dir / "piper-voices"
        
        if not voice_dir.exists():
            return []
        
        voices = []
        for path in voice_dir.glob("*.onnx"):
            # Extract voice name from filename
            name = path.stem
            voices.append(name)
        
        return voices


class TTSQueue:
    """Queue for managing TTS synthesis and playback with continuous streaming.
    
    Supports seamless playback of multiple sentences without gaps.
    """
    
    def __init__(
        self,
        tts: PiperTTS,
        amplitude_callback: Optional[Callable[[float], None]] = None,
        on_speaking_start: Optional[Callable[[], None]] = None,
        on_speaking_end: Optional[Callable[[], None]] = None,
        streaming: bool = True,
        voice_style: str = "robot",
    ):
        self.tts = tts
        self.amplitude_callback = amplitude_callback
        self.on_speaking_start = on_speaking_start
        self.on_speaking_end = on_speaking_end
        self.streaming = streaming
        self.voice_style = voice_style
        
        # Initialize voice effects
        self._effects = None
        self._init_effects()
        
        # Text queue for sentences to synthesize
        self._text_queue: queue.Queue[str | None] = queue.Queue()
        self._running = False
        self._synth_thread: Optional[threading.Thread] = None
        self._playback_thread: Optional[threading.Thread] = None
        self._stream = None
        
        # State tracking
        self._speaking = False
        self._synthesizing = False
        self._pending_sentences = 0
        self._state_lock = threading.Lock()
        
        # Audio buffer - chunks ready for playback
        self._audio_chunks: queue.Queue[np.ndarray] = queue.Queue()
        self._current_chunk: Optional[np.ndarray] = None
        self._chunk_pos = 0
        self._buffer_lock = threading.Lock()
    
    def _init_effects(self):
        """Initialize voice effects processor."""
        try:
            from voice.effects import VoiceEffects, VoiceStyle
            style = VoiceStyle(self.voice_style.lower())
            self._effects = VoiceEffects(
                sample_rate=self.tts.sample_rate,
                style=style
            )
            print(f"Voice effects initialized: {self.voice_style}")
        except Exception as e:
            print(f"Voice effects not available: {e}")
            self._effects = None
    
    def set_voice_style(self, style: str):
        """Change the voice style."""
        self.voice_style = style
        self._init_effects()
    
    def _apply_effects(self, audio: np.ndarray) -> np.ndarray:
        """Apply voice effects to audio."""
        if self._effects is None:
            return audio
        try:
            return self._effects.process(audio)
        except Exception as e:
            print(f"Effects error: {e}")
            return audio
    
    def start(self):
        """Start the TTS queue processor."""
        self._running = True
        # Synthesis thread - converts text to audio chunks
        self._synth_thread = threading.Thread(target=self._synthesis_loop, daemon=True)
        self._synth_thread.start()
    
    def stop(self):
        """Stop the TTS queue processor."""
        self._running = False
        self._text_queue.put(None)  # Signal to exit
        self._stop_audio()
    
    def clear_queue(self):
        """Clear pending text from queue without stopping the processor."""
        # Clear text queue
        while not self._text_queue.empty():
            try:
                self._text_queue.get_nowait()
            except queue.Empty:
                break
        
        # Clear audio chunks
        while not self._audio_chunks.empty():
            try:
                self._audio_chunks.get_nowait()
            except queue.Empty:
                break
        
        # Clear sentence buffer
        self._sentence_buffer = ""
    
    def speak(self, text: str):
        """Add text to the speech queue for immediate synthesis."""
        self._text_queue.put(text)
    
    def _ensure_stream_running(self):
        """Ensure audio stream is running."""
        import sounddevice as sd
        
        if self._stream is not None:
            return
        
        def audio_callback(outdata, frames, time, status):
            """Audio callback - pulls from chunk queue."""
            filled = 0
            
            while filled < frames:
                with self._buffer_lock:
                    # Need new chunk?
                    if self._current_chunk is None or self._chunk_pos >= len(self._current_chunk):
                        try:
                            self._current_chunk = self._audio_chunks.get_nowait()
                            self._chunk_pos = 0
                        except queue.Empty:
                            break
                    
                    # Copy from current chunk
                    remaining = len(self._current_chunk) - self._chunk_pos
                    to_copy = min(remaining, frames - filled)
                    
                    outdata[filled:filled + to_copy, 0] = self._current_chunk[self._chunk_pos:self._chunk_pos + to_copy]
                    self._chunk_pos += to_copy
                    filled += to_copy
            
            # Fill rest with silence
            if filled < frames:
                outdata[filled:].fill(0)
            
            # Amplitude for avatar
            if self.amplitude_callback:
                if filled > 0:
                    amp = np.sqrt(np.mean(outdata[:filled] ** 2))
                    self.amplitude_callback(min(1.0, amp * 5))
                else:
                    self.amplitude_callback(0.0)
        
        self._stream = sd.OutputStream(
            samplerate=self.tts.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=audio_callback,
            blocksize=2048,
        )
        self._stream.start()
    
    def _stop_audio(self):
        """Stop the audio stream and clear buffers."""
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        
        # Clear buffers
        with self._buffer_lock:
            self._current_chunk = None
            self._chunk_pos = 0
            while not self._audio_chunks.empty():
                try:
                    self._audio_chunks.get_nowait()
                except queue.Empty:
                    break
        
        if self.amplitude_callback:
            self.amplitude_callback(0.0)
    
    def _synthesis_loop(self):
        """Main loop - synthesizes text with parallel synthesis for seamless playback.
        
        Key improvement: Multiple sentences can synthesize concurrently while
        maintaining correct playback order. Each sentence gets its own chunk queue,
        and we drain them in FIFO order to the main audio queue.
        """
        import time
        from collections import deque
        
        # Track in-flight syntheses: (chunk_queue, done_event, text_snippet)
        pending_syntheses: deque = deque()
        MAX_CONCURRENT = 3  # Allow up to 3 sentences synthesizing in parallel
        
        while self._running:
            # === Phase 1: Start new syntheses if we have capacity ===
            while len(pending_syntheses) < MAX_CONCURRENT:
                try:
                    text = self._text_queue.get_nowait()
                except queue.Empty:
                    break
                
                if text is None:
                    # Shutdown signal - but first drain pending
                    self._running = False
                    break
                
                # Start speaking if not already
                self._start_speaking_if_needed()
                
                # Create a dedicated queue for this sentence's audio chunks
                sentence_chunks: queue.Queue[np.ndarray] = queue.Queue()
                done_event = threading.Event()
                
                # Create callbacks that capture the right queue/event
                def make_callbacks(chunks_q: queue.Queue, done_evt: threading.Event):
                    def on_chunk(chunk: np.ndarray):
                        processed = self._apply_effects(chunk)
                        chunks_q.put(processed)
                    
                    def on_complete():
                        done_evt.set()
                    
                    return on_chunk, on_complete
                
                on_chunk, on_complete = make_callbacks(sentence_chunks, done_event)
                
                print(f"TTS: '{text[:40]}...'")
                self.tts.synthesize_streaming(text, on_chunk, on_complete)
                
                pending_syntheses.append((sentence_chunks, done_event, text[:20]))
                
                with self._state_lock:
                    self._pending_sentences += 1
            
            # === Phase 2: Drain chunks from front synthesis to main audio queue ===
            # This maintains order: we only output chunks from the oldest synthesis
            while pending_syntheses:
                front_chunks, front_done, front_snippet = pending_syntheses[0]
                
                # Drain all available chunks from the front synthesis
                drained = False
                while True:
                    try:
                        chunk = front_chunks.get_nowait()
                        self._audio_chunks.put(chunk)
                        drained = True
                    except queue.Empty:
                        break
                
                # If this synthesis is complete and fully drained, move to next
                if front_done.is_set() and front_chunks.empty():
                    pending_syntheses.popleft()
                    with self._state_lock:
                        self._pending_sentences = max(0, self._pending_sentences - 1)
                else:
                    # Still waiting for more chunks from current synthesis
                    break
            
            # Small sleep to avoid busy-looping
            time.sleep(0.005)
            
            # Check if we should go idle (no more pending work)
            if not pending_syntheses:
                self._check_idle()
        
        # === Cleanup: drain any remaining syntheses ===
        while pending_syntheses:
            front_chunks, front_done, _ = pending_syntheses[0]
            
            # Wait for synthesis to complete (with timeout)
            front_done.wait(timeout=5.0)
            
            # Drain remaining chunks
            while True:
                try:
                    chunk = front_chunks.get_nowait()
                    self._audio_chunks.put(chunk)
                except queue.Empty:
                    break
            
            pending_syntheses.popleft()
            with self._state_lock:
                self._pending_sentences = max(0, self._pending_sentences - 1)
        
        # Final cleanup
        self._finish_speaking()
    
    def _start_speaking_if_needed(self):
        """Start speaking state and audio stream if not already speaking."""
        with self._state_lock:
            if self._speaking:
                return
            self._speaking = True
        
        # Ensure stream is running
        self._ensure_stream_running()
        
        # Notify callback
        if self.on_speaking_start:
            self.on_speaking_start()
    
    def _check_idle(self):
        """Check if we're done speaking and should go idle."""
        import time
        
        with self._state_lock:
            if not self._speaking:
                return
            
            # Still have pending sentences?
            if self._pending_sentences > 0:
                return
        
        # Check if audio buffer is empty
        with self._buffer_lock:
            has_audio = (not self._audio_chunks.empty() or 
                        (self._current_chunk is not None and 
                         self._chunk_pos < len(self._current_chunk)))
        
        if has_audio:
            return  # Still playing audio
        
        # Wait a tiny bit to ensure no more sentences coming
        time.sleep(0.1)
        
        # Double-check
        with self._state_lock:
            if self._pending_sentences > 0:
                return
        
        if not self._text_queue.empty():
            return
        
        # Truly done
        self._finish_speaking()
    
    def _finish_speaking(self):
        """Finish speaking and notify callback."""
        with self._state_lock:
            if not self._speaking:
                return
            self._speaking = False
        
        # Stop audio stream
        self._stop_audio()
        
        # Notify callback
        if self.on_speaking_end:
            self.on_speaking_end()
    

    
    def clear(self):
        """Clear pending speech."""
        # Clear text queue
        while not self._text_queue.empty():
            try:
                self._text_queue.get_nowait()
            except queue.Empty:
                break
        
        with self._state_lock:
            self._pending_sentences = 0
        
        self._stop_audio()


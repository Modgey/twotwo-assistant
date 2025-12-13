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
    """Queue for managing TTS synthesis and playback with streaming support."""
    
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
        
        self._queue: queue.Queue[str] = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stream = None
        self._speaking = False
    
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
        self._thread = threading.Thread(target=self._process_queue, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop the TTS queue processor."""
        self._running = False
        self._queue.put(None)  # Signal to exit
        self._stop_audio()
    
    def speak(self, text: str):
        """Add text to the speech queue."""
        self._queue.put(text)
    
    def _start_audio_stream(self):
        """Start the audio output stream."""
        import sounddevice as sd
        
        if self._stream is not None:
            return
        
        self._audio_buffer = []
        self._buffer_lock = threading.Lock()
        self._buffer_pos = 0
        
        def audio_callback(outdata, frames, time, status):
            with self._buffer_lock:
                if not self._audio_buffer:
                    outdata.fill(0)
                    if self.amplitude_callback:
                        self.amplitude_callback(0.0)
                    return
                
                # Flatten buffer and get frames
                all_audio = np.concatenate(self._audio_buffer)
                
                if self._buffer_pos >= len(all_audio):
                    outdata.fill(0)
                    if self.amplitude_callback:
                        self.amplitude_callback(0.0)
                    return
                
                remaining = len(all_audio) - self._buffer_pos
                to_copy = min(frames, remaining)
                
                outdata[:to_copy, 0] = all_audio[self._buffer_pos:self._buffer_pos + to_copy]
                outdata[to_copy:].fill(0)
                
                self._buffer_pos += to_copy
                
                # Amplitude for avatar
                if self.amplitude_callback and to_copy > 0:
                    amp = np.sqrt(np.mean(outdata[:to_copy] ** 2))
                    self.amplitude_callback(min(1.0, amp * 5))
        
        self._stream = sd.OutputStream(
            samplerate=self.tts.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=audio_callback,
            blocksize=512,  # Small buffer for low latency
        )
        self._stream.start()
    
    def _stop_audio(self):
        """Stop the audio stream."""
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        
        if self.amplitude_callback:
            self.amplitude_callback(0.0)
    
    def _add_audio_chunk(self, chunk: np.ndarray):
        """Add audio chunk to the buffer."""
        with self._buffer_lock:
            self._audio_buffer.append(chunk)
    
    def _wait_for_playback(self, timeout: float = 30.0):
        """Wait for all buffered audio to finish playing."""
        import time
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self._buffer_lock:
                if not self._audio_buffer:
                    break
                all_audio = np.concatenate(self._audio_buffer)
                if self._buffer_pos >= len(all_audio):
                    break
            time.sleep(0.05)
        else:
            print("TTS: Playback wait timeout")
    
    def _process_queue(self):
        """Process the TTS queue with streaming."""
        while self._running:
            try:
                text = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if text is None:
                break
            
            # Notify speaking start
            self._speaking = True
            if self.on_speaking_start:
                self.on_speaking_start()
            
            if self.streaming:
                self._process_streaming(text)
            else:
                self._process_blocking(text)
            
            # Notify speaking end
            self._speaking = False
            if self.on_speaking_end:
                self.on_speaking_end()
    
    def _process_streaming(self, text: str):
        """Process text with streaming TTS."""
        try:
            # Start audio stream
            self._start_audio_stream()
            
            with self._buffer_lock:
                self._audio_buffer = []
                self._buffer_pos = 0
            
            # Stream synthesis
            done_event = threading.Event()
            
            def on_chunk(chunk):
                # Apply robot voice effects to each chunk
                processed = self._apply_effects(chunk)
                self._add_audio_chunk(processed)
            
            def on_complete():
                print("TTS synthesis complete")
                done_event.set()
            
            print(f"TTS: Synthesizing '{text[:50]}...'")
            self.tts.synthesize_streaming(text, on_chunk, on_complete)
            
            if not done_event.wait(timeout=30):
                print("TTS: Synthesis timeout")
            
            # Wait for playback to finish
            self._wait_for_playback()
            
        except Exception as e:
            print(f"TTS streaming error: {e}")
        finally:
            self._stop_audio()
    
    def _process_blocking(self, text: str):
        """Process text with blocking TTS (non-streaming fallback)."""
        from voice.audio import AudioPlayer
        
        audio = self.tts.synthesize(text)
        if audio is None:
            return
        
        # Apply robot voice effects
        audio = self._apply_effects(audio)
        
        player = AudioPlayer(amplitude_callback=self.amplitude_callback)
        done_event = threading.Event()
        
        player.play(audio, self.tts.sample_rate, lambda: done_event.set())
        done_event.wait(timeout=30)
        player.stop()
    
    def clear(self):
        """Clear pending speech."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        
        self._stop_audio()


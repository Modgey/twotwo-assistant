"""TwoTwo Audio Utilities

Audio capture and playback using sounddevice.
Handles recording, playback, and amplitude extraction for avatar sync.
"""

import numpy as np
import sounddevice as sd
import threading
import queue
from pathlib import Path
from typing import Callable, Optional
import wave
import io


class AudioRecorder:
    """Records audio from microphone with real-time amplitude callback."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: Optional[int] = None,
        amplitude_callback: Optional[Callable[[float], None]] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.amplitude_callback = amplitude_callback
        
        self._recording = False
        self._audio_data: list[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time, status):
        """Called for each audio chunk during recording."""
        if status:
            print(f"Audio recording status: {status}")
        
        # Store audio data
        with self._lock:
            if self._recording:
                self._audio_data.append(indata.copy())
        
        # Calculate amplitude for avatar sync
        if self.amplitude_callback:
            # RMS amplitude, normalized to 0-1
            amplitude = np.sqrt(np.mean(indata ** 2))
            # Amplify and clamp
            amplitude = min(1.0, amplitude * 10)
            self.amplitude_callback(amplitude)
    
    def start(self):
        """Start recording audio."""
        with self._lock:
            self._audio_data = []
            self._recording = True
        
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            device=self.device,
            dtype=np.float32,
            callback=self._audio_callback,
        )
        self._stream.start()
    
    def stop(self) -> np.ndarray:
        """Stop recording and return audio data."""
        with self._lock:
            self._recording = False
        
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        # Reset amplitude
        if self.amplitude_callback:
            self.amplitude_callback(0.0)
        
        # Concatenate all audio chunks
        with self._lock:
            if self._audio_data:
                return np.concatenate(self._audio_data, axis=0)
            return np.array([], dtype=np.float32)
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording
    
    def get_audio_bytes(self) -> bytes:
        """Get recorded audio as WAV bytes (for whisper)."""
        audio = self.stop()
        if len(audio) == 0:
            return b""
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Write to WAV format in memory
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())
        
        return buffer.getvalue()
    
    def save_wav(self, filepath: Path) -> bool:
        """Save recorded audio to WAV file."""
        audio = self.stop()
        if len(audio) == 0:
            return False
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        with wave.open(str(filepath), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())
        
        return True


class AudioPlayer:
    """Plays audio with real-time amplitude callback for avatar sync."""
    
    def __init__(
        self,
        device: Optional[int] = None,
        amplitude_callback: Optional[Callable[[float], None]] = None,
    ):
        self.device = device
        self.amplitude_callback = amplitude_callback
        
        self._playing = False
        self._stream: Optional[sd.OutputStream] = None
        self._audio_queue: queue.Queue = queue.Queue()
        self._current_audio: Optional[np.ndarray] = None
        self._audio_pos = 0
        self._lock = threading.Lock()
        self._finish_callback: Optional[Callable[[], None]] = None
        self._finish_called = False
    
    def _audio_callback(self, outdata: np.ndarray, frames: int, time, status):
        """Called for each audio chunk during playback."""
        if status:
            print(f"Audio playback status: {status}")
        
        playback_complete = False
        
        with self._lock:
            if self._current_audio is None or self._audio_pos >= len(self._current_audio):
                # Try to get next chunk from queue
                try:
                    self._current_audio = self._audio_queue.get_nowait()
                    self._audio_pos = 0
                except queue.Empty:
                    # No more audio - playback is complete
                    playback_complete = True
                    outdata.fill(0)
                    if self.amplitude_callback:
                        self.amplitude_callback(0.0)
            
            if not playback_complete:
                # Fill output buffer
                remaining = len(self._current_audio) - self._audio_pos
                to_copy = min(frames, remaining)
                
                outdata[:to_copy] = self._current_audio[self._audio_pos:self._audio_pos + to_copy].reshape(-1, 1)
                outdata[to_copy:].fill(0)
                
                self._audio_pos += to_copy
                
                # Check if we've finished playing all audio
                if self._audio_pos >= len(self._current_audio):
                    playback_complete = True
                
                # Calculate amplitude for avatar sync
                if self.amplitude_callback:
                    chunk = outdata[:to_copy]
                    amplitude = np.sqrt(np.mean(chunk ** 2))
                    amplitude = min(1.0, amplitude * 5)
                    self.amplitude_callback(amplitude)
        
        # Handle playback completion outside the lock (schedule on thread to avoid blocking callback)
        if playback_complete:
            threading.Thread(target=self._handle_playback_complete, daemon=True).start()
    
    def play(
        self,
        audio: np.ndarray,
        sample_rate: int = 22050,
        on_finish: Optional[Callable[[], None]] = None,
    ):
        """Play audio data."""
        with self._lock:
            self._finish_callback = on_finish
            self._finish_called = False
            self._current_audio = audio.astype(np.float32)
            self._audio_pos = 0
            self._playing = True
        
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            device=self.device,
            dtype=np.float32,
            callback=self._audio_callback,
            finished_callback=self._on_stream_finished,
        )
        self._stream.start()
    
    def play_file(
        self,
        filepath: Path,
        on_finish: Optional[Callable[[], None]] = None,
    ):
        """Play audio from WAV file."""
        with wave.open(str(filepath), "rb") as wf:
            sample_rate = wf.getframerate()
            audio = np.frombuffer(wf.readframes(-1), dtype=np.int16)
            audio = audio.astype(np.float32) / 32767.0
        
        self.play(audio, sample_rate, on_finish)
    
    def stream_audio(self, audio_chunk: np.ndarray):
        """Add audio chunk to streaming queue."""
        self._audio_queue.put(audio_chunk.astype(np.float32))
    
    def _handle_playback_complete(self):
        """Handle playback completion."""
        with self._lock:
            if not self._playing or self._finish_called:
                return  # Already handled
            self._playing = False
            self._finish_called = True
            callback = self._finish_callback
            self._finish_callback = None
        
        # Stop the stream (outside lock to avoid deadlock)
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            finally:
                self._stream = None
        
        # Call finish callback
        if callback:
            callback()
    
    def _on_stream_finished(self):
        """Called when stream finishes (from sounddevice)."""
        self._handle_playback_complete()
    
    def stop(self):
        """Stop playback."""
        with self._lock:
            self._playing = False
            self._current_audio = None
            self._audio_pos = 0
            self._finish_callback = None
            self._finish_called = False
        
        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        if self.amplitude_callback:
            self.amplitude_callback(0.0)
    
    def is_playing(self) -> bool:
        """Check if currently playing."""
        return self._playing


def list_audio_devices() -> dict:
    """List available audio input and output devices."""
    devices = sd.query_devices()
    
    inputs = []
    outputs = []
    
    for i, device in enumerate(devices):
        info = {
            "id": i,
            "name": device["name"],
            "sample_rate": device["default_samplerate"],
        }
        
        if device["max_input_channels"] > 0:
            inputs.append(info)
        if device["max_output_channels"] > 0:
            outputs.append(info)
    
    return {"inputs": inputs, "outputs": outputs}


def get_default_devices() -> dict:
    """Get default input and output device IDs."""
    return {
        "input": sd.default.device[0],
        "output": sd.default.device[1],
    }


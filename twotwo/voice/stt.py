"""TwoTwo Speech-to-Text

Whisper.cpp integration for local speech recognition.
Supports both pywhispercpp bindings and subprocess fallback.
"""

import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Callable, Optional
import numpy as np

from config import get_config, get_config_dir


class WhisperSTT:
    """Speech-to-text using whisper.cpp."""
    
    # Model sizes and their file names
    MODELS = {
        "tiny": "ggml-tiny.bin",
        "base": "ggml-base.bin",
        "small": "ggml-small.bin",
        "medium": "ggml-medium.bin",
    }
    
    def __init__(
        self,
        model_name: str = "tiny",
        model_dir: Optional[Path] = None,
        n_threads: int = 6,
    ):
        self.model_name = model_name
        self.n_threads = n_threads
        
        # Model directory
        if model_dir:
            self.model_dir = model_dir
        else:
            self.model_dir = get_config_dir() / "whisper-models"
        
        self._model = None
        self._use_subprocess = False
        self._whisper_cpp_path: Optional[Path] = None
        self._transcribe_lock = threading.Lock()
        
        self._init_whisper()
    
    def _init_whisper(self):
        """Initialize whisper - try pywhispercpp first, fall back to subprocess."""
        try:
            from pywhispercpp.model import Model
            
            model_path = self.model_dir / self.MODELS.get(self.model_name, "ggml-base.bin")
            
            if not model_path.exists():
                print(f"Whisper model not found at {model_path}")
                print(f"Please download the model from https://huggingface.co/ggerganov/whisper.cpp")
                self._model = None
                return
            
            self._model = Model(str(model_path), n_threads=self.n_threads)
            self._use_subprocess = False
            print(f"Loaded whisper model: {self.model_name}")
            
        except ImportError:
            print("pywhispercpp not available, will try subprocess fallback")
            self._use_subprocess = True
            self._find_whisper_cpp()
    
    def _find_whisper_cpp(self):
        """Find whisper.cpp executable."""
        # Check common locations
        possible_paths = [
            Path("whisper-cpp") / "main.exe",
            Path("whisper-cpp") / "main",
            get_config_dir() / "whisper-cpp" / "main.exe",
            get_config_dir() / "whisper-cpp" / "main",
        ]
        
        for path in possible_paths:
            if path.exists():
                self._whisper_cpp_path = path
                print(f"Found whisper.cpp at {path}")
                return
        
        # Try to find in PATH
        try:
            result = subprocess.run(
                ["where" if subprocess.sys.platform == "win32" else "which", "whisper-cpp"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                self._whisper_cpp_path = Path(result.stdout.strip())
                return
        except Exception:
            pass
        
        print("whisper.cpp not found. Please install pywhispercpp or whisper.cpp")
    
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str = "en",
    ) -> str:
        """Transcribe audio to text.
        
        Args:
            audio: Audio data as numpy array (float32, mono)
            sample_rate: Sample rate of the audio
            language: Language code (e.g., "en", "es", "fr")
            
        Returns:
            Transcribed text
        """
        # Prevent concurrent transcriptions (pywhispercpp isn't thread-safe)
        with self._transcribe_lock:
            if self._use_subprocess:
                return self._transcribe_subprocess(audio, sample_rate, language)
            else:
                return self._transcribe_pywhisper(audio, sample_rate, language)
    
    def _transcribe_pywhisper(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: str,
    ) -> str:
        """Transcribe using pywhispercpp bindings."""
        if self._model is None:
            print("STT: Model not loaded")
            return ""
        
        try:
            # Ensure audio is the right format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Ensure contiguous array
            if not audio.flags['C_CONTIGUOUS']:
                audio = np.ascontiguousarray(audio)
            
            # Resample if needed (whisper expects 16kHz)
            if sample_rate != 16000:
                # Simple linear interpolation resample
                duration = len(audio) / sample_rate
                new_length = int(duration * 16000)
                indices = np.linspace(0, len(audio) - 1, new_length)
                audio = np.interp(indices, np.arange(len(audio)), audio)
            
            # Flatten if needed
            if len(audio.shape) > 1:
                audio = audio.flatten()
            
            # Transcribe
            segments = self._model.transcribe(audio, language=language)
            
            # Combine segments
            text = " ".join(segment.text.strip() for segment in segments)
            return text.strip()
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
    
    def _transcribe_subprocess(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: str,
    ) -> str:
        """Transcribe using whisper.cpp subprocess."""
        if self._whisper_cpp_path is None:
            return ""
        
        try:
            # Save audio to temp WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = Path(f.name)
            
            # Write WAV
            import wave
            audio_int16 = (audio * 32767).astype(np.int16)
            with wave.open(str(temp_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            # Run whisper.cpp
            model_path = self.model_dir / self.MODELS.get(self.model_name, "ggml-base.bin")
            
            result = subprocess.run(
                [
                    str(self._whisper_cpp_path),
                    "-m", str(model_path),
                    "-f", str(temp_path),
                    "-l", language,
                    "-t", str(self.n_threads),
                    "--no-timestamps",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            # Clean up
            temp_path.unlink(missing_ok=True)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"whisper.cpp error: {result.stderr}")
                return ""
                
        except Exception as e:
            print(f"Subprocess transcription error: {e}")
            return ""
    
    def transcribe_file(self, filepath: Path, language: str = "en") -> str:
        """Transcribe audio from file."""
        import wave
        
        with wave.open(str(filepath), "rb") as wf:
            sample_rate = wf.getframerate()
            audio = np.frombuffer(wf.readframes(-1), dtype=np.int16)
            audio = audio.astype(np.float32) / 32767.0
        
        return self.transcribe(audio, sample_rate, language)
    
    def transcribe_async(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str = "en",
        callback: Optional[Callable[[str], None]] = None,
        timeout: float = 15.0,
    ) -> threading.Thread:
        """Transcribe audio asynchronously.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            language: Language code
            callback: Called with transcription result
            timeout: Maximum time to wait for transcription
            
        Returns:
            Thread that will run the transcription
        """
        result_container = {"text": "", "done": False}
        
        def worker():
            try:
                print(f"STT: Starting transcription ({len(audio)} samples)...")
                result_container["text"] = self.transcribe(audio, sample_rate, language)
                print(f"STT: Transcription complete: '{result_container['text']}'")
            except Exception as e:
                print(f"STT: Transcription error: {e}")
                result_container["text"] = ""
            finally:
                result_container["done"] = True
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        
        # Wait with timeout in a separate thread, then call callback
        def wait_and_callback():
            thread.join(timeout=timeout)
            if not result_container["done"]:
                print(f"STT: Transcription timeout after {timeout}s")
                result_container["text"] = ""
            if callback:
                callback(result_container["text"])
        
        callback_thread = threading.Thread(target=wait_and_callback, daemon=True)
        callback_thread.start()
        
        return thread
    
    def is_available(self) -> bool:
        """Check if whisper is available and ready."""
        if self._use_subprocess:
            return self._whisper_cpp_path is not None
        else:
            return self._model is not None
    
    @classmethod
    def get_model_path(cls, model_name: str) -> Path:
        """Get the expected path for a model."""
        config_dir = get_config_dir()
        model_dir = config_dir / "whisper-models"
        return model_dir / cls.MODELS.get(model_name, "ggml-base.bin")
    
    @classmethod
    def list_available_models(cls) -> list[str]:
        """List models that are downloaded."""
        config_dir = get_config_dir()
        model_dir = config_dir / "whisper-models"
        
        available = []
        for name, filename in cls.MODELS.items():
            if (model_dir / filename).exists():
                available.append(name)
        
        return available


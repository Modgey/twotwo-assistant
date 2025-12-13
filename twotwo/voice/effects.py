"""TwoTwo Voice Effects

Audio effects for robotic voice processing.
"""

import numpy as np
from typing import Optional
from enum import Enum


class VoiceStyle(Enum):
    """Preset voice styles."""
    NORMAL = "normal"
    ROBOT = "robot"           # Generic robot voice
    CLAPTRAP = "claptrap"     # High-pitched, energetic robot
    GLADOS = "glados"         # Smooth, slightly sinister AI
    METALLIC = "metallic"     # Heavy ring modulation


# Style presets
VOICE_PRESETS = {
    VoiceStyle.NORMAL: {
        "pitch_shift": 0,
        "ring_mod_freq": 0,
        "ring_mod_mix": 0,
        "chorus_depth": 0,
        "distortion": 0,
    },
    VoiceStyle.ROBOT: {
        "pitch_shift": 0,
        "ring_mod_freq": 30,
        "ring_mod_mix": 0.15,
        "chorus_depth": 0.02,
        "distortion": 0.1,
    },
    VoiceStyle.CLAPTRAP: {
        "pitch_shift": 4,       # Higher pitch (semitones)
        "ring_mod_freq": 50,
        "ring_mod_mix": 0.12,
        "chorus_depth": 0.03,
        "distortion": 0.15,
    },
    VoiceStyle.GLADOS: {
        "pitch_shift": -2,      # Slightly lower
        "ring_mod_freq": 20,
        "ring_mod_mix": 0.08,
        "chorus_depth": 0.01,
        "distortion": 0.05,
    },
    VoiceStyle.METALLIC: {
        "pitch_shift": 0,
        "ring_mod_freq": 80,
        "ring_mod_mix": 0.25,
        "chorus_depth": 0.04,
        "distortion": 0.2,
    },
}


class VoiceEffects:
    """Apply robotic voice effects to audio."""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        style: VoiceStyle = VoiceStyle.ROBOT,
    ):
        self.sample_rate = sample_rate
        self.style = style
        self._params = VOICE_PRESETS[style].copy()
    
    def set_style(self, style: VoiceStyle):
        """Change voice style preset."""
        self.style = style
        self._params = VOICE_PRESETS[style].copy()
    
    def set_param(self, name: str, value: float):
        """Set individual effect parameter."""
        if name in self._params:
            self._params[name] = value
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply all effects to audio."""
        if self.style == VoiceStyle.NORMAL:
            return audio
        
        result = audio.copy().astype(np.float32)
        
        # Apply pitch shift
        if self._params["pitch_shift"] != 0:
            result = self._pitch_shift(result, self._params["pitch_shift"])
        
        # Apply ring modulation (key robot effect)
        if self._params["ring_mod_mix"] > 0:
            result = self._ring_modulate(
                result,
                self._params["ring_mod_freq"],
                self._params["ring_mod_mix"]
            )
        
        # Apply chorus for synthetic doubling
        if self._params["chorus_depth"] > 0:
            result = self._chorus(result, self._params["chorus_depth"])
        
        # Apply soft distortion for digital edge
        if self._params["distortion"] > 0:
            result = self._distort(result, self._params["distortion"])
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(result))
        if max_val > 0.95:
            result = result * (0.95 / max_val)
        
        return result
    
    def _pitch_shift(self, audio: np.ndarray, semitones: float) -> np.ndarray:
        """Simple pitch shift using resampling."""
        if semitones == 0:
            return audio
        
        # Calculate ratio (2^(semitones/12))
        ratio = 2 ** (semitones / 12)
        
        # Resample
        old_length = len(audio)
        new_length = int(old_length / ratio)
        
        if new_length < 10:
            return audio
        
        # Interpolate to new length
        old_indices = np.arange(old_length)
        new_indices = np.linspace(0, old_length - 1, new_length)
        shifted = np.interp(new_indices, old_indices, audio)
        
        # Resample back to original length to maintain duration
        shifted_indices = np.arange(len(shifted))
        final_indices = np.linspace(0, len(shifted) - 1, old_length)
        result = np.interp(final_indices, shifted_indices, shifted)
        
        return result.astype(np.float32)
    
    def _ring_modulate(
        self,
        audio: np.ndarray,
        freq: float,
        mix: float
    ) -> np.ndarray:
        """Apply ring modulation for metallic robot effect."""
        if freq <= 0 or mix <= 0:
            return audio
        
        # Generate carrier wave
        t = np.arange(len(audio)) / self.sample_rate
        carrier = np.sin(2 * np.pi * freq * t)
        
        # Apply ring modulation
        modulated = audio * carrier
        
        # Mix with original
        result = audio * (1 - mix) + modulated * mix
        
        return result.astype(np.float32)
    
    def _chorus(self, audio: np.ndarray, depth: float) -> np.ndarray:
        """Apply chorus effect for synthetic doubling."""
        if depth <= 0:
            return audio
        
        # Create delayed copy with slight pitch variation
        delay_samples = int(0.02 * self.sample_rate)  # 20ms delay
        
        if len(audio) < delay_samples * 2:
            return audio
        
        # Simple delay-based chorus
        delayed = np.zeros_like(audio)
        delayed[delay_samples:] = audio[:-delay_samples]
        
        # Slight pitch modulation on delayed signal
        mod_freq = 0.5  # Hz
        t = np.arange(len(audio)) / self.sample_rate
        mod = 1 + depth * np.sin(2 * np.pi * mod_freq * t)
        
        # Mix
        result = audio * 0.7 + delayed * 0.3 * mod
        
        return result.astype(np.float32)
    
    def _distort(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """Apply soft distortion for digital edge."""
        if amount <= 0:
            return audio
        
        # Soft clipping using tanh
        drive = 1 + amount * 5
        result = np.tanh(audio * drive) / np.tanh(drive)
        
        return result.astype(np.float32)


def apply_robot_voice(
    audio: np.ndarray,
    sample_rate: int = 22050,
    style: str = "robot"
) -> np.ndarray:
    """Convenience function to apply robot voice effect.
    
    Args:
        audio: Input audio (float32)
        sample_rate: Audio sample rate
        style: One of "robot", "claptrap", "glados", "metallic"
        
    Returns:
        Processed audio
    """
    try:
        voice_style = VoiceStyle(style.lower())
    except ValueError:
        voice_style = VoiceStyle.ROBOT
    
    effects = VoiceEffects(sample_rate=sample_rate, style=voice_style)
    return effects.process(audio)


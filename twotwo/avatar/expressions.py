"""TwoTwo Avatar Expressions

Expression parameters for Emo-inspired minimal avatar with aperture frame.
"""

from dataclasses import dataclass
from core.state import AvatarState


@dataclass
class EyeParams:
    """Parameters for a single eye."""
    width: float = 1.0           # Width scale
    height: float = 1.0          # Height scale
    roundness: float = 0.5       # 0=rectangle, 1=circle
    rotation: float = 0.0        # Rotation in degrees
    offset_x: float = 0.0        # Horizontal offset from default position
    offset_y: float = 0.0        # Vertical offset from default position


@dataclass
class ApertureParams:
    """Parameters for the aperture lens frame."""
    openness: float = 1.0        # How open (0=closed, 1=normal, 1.5=wide)
    rotation: float = 0.0        # Current rotation offset in degrees
    rotation_speed: float = 0.0  # Rotation speed (degrees per second)
    pulse_speed: float = 0.0     # Pulse frequency (Hz)
    pulse_amount: float = 0.0    # Pulse amplitude


@dataclass
class ExpressionParams:
    """Visual parameters for an avatar expression state."""
    
    # Left eye
    left_eye: EyeParams = None
    
    # Right eye
    right_eye: EyeParams = None
    
    # Aperture frame
    aperture: ApertureParams = None
    
    # Both eyes (spacing, etc)
    eye_spacing: float = 1.0     # Spacing multiplier
    
    # Animation
    bob_speed: float = 0.0       # Vertical bobbing (Hz)
    bob_amount: float = 0.0      # Bob amplitude (pixels)
    
    # Blink
    blink_enabled: bool = True
    blink_interval: float = 4.0
    
    # Transition
    transition_time: float = 0.3
    
    def __post_init__(self):
        if self.left_eye is None:
            self.left_eye = EyeParams()
        if self.right_eye is None:
            self.right_eye = EyeParams()
        if self.aperture is None:
            self.aperture = ApertureParams()


# Expression definitions - Emo-inspired with aperture frame
EXPRESSIONS = {
    AvatarState.IDLE: ExpressionParams(
        left_eye=EyeParams(
            width=1.0,
            height=1.0,
            roundness=0.65,
            rotation=0.0,
        ),
        right_eye=EyeParams(
            width=1.0,
            height=1.0,
            roundness=0.65,
            rotation=0.0,
        ),
        aperture=ApertureParams(
            openness=1.0,
            rotation=0.0,
            rotation_speed=0.0,
            pulse_speed=0.15,   # Slow breathing
            pulse_amount=0.02,
        ),
        eye_spacing=1.0,
        bob_speed=0.5,
        bob_amount=2.0,
        blink_enabled=True,
        blink_interval=3.8,
        transition_time=0.5,
    ),
    
    AvatarState.LISTENING: ExpressionParams(
        left_eye=EyeParams(
            width=0.95,
            height=0.92,        # Slightly smaller (head tilt effect)
            roundness=0.55,
            rotation=5.0,       # Tilted (leaning in)
            offset_y=1.0,       # Lower
        ),
        right_eye=EyeParams(
            width=1.08,
            height=1.15,        # Bigger, alert
            roundness=0.65,
            rotation=5.0,       # Same tilt (head tilt)
            offset_y=-1.0,      # Higher (asymmetric)
        ),
        aperture=ApertureParams(
            openness=1.08,      # Opens slightly (attentive)
            rotation=3.0,       # Subtle offset
            rotation_speed=0.0,
            pulse_speed=0.0,    # No pulse - calm focus
            pulse_amount=0.0,
        ),
        eye_spacing=0.94,       # Slightly closer (focused)
        bob_speed=0.0,
        bob_amount=0.0,
        blink_enabled=True,    # No blinking (attentive)
        transition_time=0.25,
    ),
    
    AvatarState.THINKING: ExpressionParams(
        left_eye=EyeParams(
            width=1.0,
            height=0.4,
            roundness=0.3,
            rotation=-5.0,
            offset_y=-3.0,
        ),
        right_eye=EyeParams(
            width=1.0,
            height=0.5,
            roundness=0.3,
            rotation=3.0,
            offset_y=-2.0,
        ),
        aperture=ApertureParams(
            openness=0.95,      # Slightly tighter
            rotation=0.0,
            rotation_speed=8.0, # Slow rotation
            pulse_speed=0.3,
            pulse_amount=0.03,
        ),
        eye_spacing=0.95,
        bob_speed=0.0,
        bob_amount=0.0,
        blink_enabled=True,
        blink_interval=5.0,
        transition_time=0.2,
    ),
    
    AvatarState.SPEAKING: ExpressionParams(
        left_eye=EyeParams(
            width=1.0,
            height=1.0,
            roundness=0.7,
            rotation=0.0,
        ),
        right_eye=EyeParams(
            width=1.0,
            height=1.0,
            roundness=0.7,
            rotation=0.0,
        ),
        aperture=ApertureParams(
            openness=0.85,
            rotation=0.5,
            rotation_speed=2.0,
            pulse_speed=0.0,    # Driven by audio
            pulse_amount=0.0,
        ),
        eye_spacing=1.0,
        bob_speed=2.0,
        bob_amount=3.0,
        blink_enabled=True,
        blink_interval=3.5,
        transition_time=0.2,
    ),
}


def get_expression(state: AvatarState) -> ExpressionParams:
    """Get expression parameters for a given avatar state."""
    return EXPRESSIONS.get(state, EXPRESSIONS[AvatarState.IDLE])

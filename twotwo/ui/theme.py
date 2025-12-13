"""TwoTwo Theme

Shared colors, effects, and styling for the hologram aesthetic.
"""

# =============================================================================
# COLORS
# =============================================================================

# Primary amber color (RGB + Alpha variants)
AMBER = (255, 191, 0)
AMBER_FULL = (255, 191, 0, 220)      # Main avatar/text color
AMBER_GLOW = (255, 210, 100, 80)     # Soft glow effect
AMBER_DIM = (255, 191, 0, 120)       # Dimmed for backgrounds

# Background colors
BG_DARK = (0, 0, 0, 120)             # Semi-transparent black
BG_SOLID = (10, 10, 10)              # Solid dark background

# Shadow
SHADOW = (0, 0, 0)
SHADOW_ALPHA = 100

# Border/accent
BORDER_GLOW = (*AMBER, 40)           # Subtle amber border


# =============================================================================
# HOLOGRAM EFFECTS
# =============================================================================

# Scanlines
SCANLINE_SPACING = 3                 # Pixels between scan lines
SCANLINE_ALPHA = 40                  # Darkness of scan lines
SCANLINE_SPEED = 20                  # Animation speed

# Glow
GLOW_RADIUS = 8
GLOW_ALPHA = 80

# Shadow
SHADOW_OFFSET = 3


# =============================================================================
# TYPOGRAPHY
# =============================================================================

FONT_FAMILY = "Consolas"
FONT_SIZE_SMALL = 12
FONT_SIZE_MEDIUM = 14
FONT_SIZE_LARGE = 16


# =============================================================================
# ANIMATION
# =============================================================================

TYPING_SPEED = 40                    # Characters per second
FADE_SPEED = 4.0                     # Alpha units per second


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def with_alpha(color: tuple, alpha: int) -> tuple:
    """Return color tuple with specified alpha."""
    if len(color) == 3:
        return (*color, alpha)
    return (color[0], color[1], color[2], alpha)


def lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
    """Linearly interpolate between two colors."""
    if len(c1) == 3:
        c1 = (*c1, 255)
    if len(c2) == 3:
        c2 = (*c2, 255)
    
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


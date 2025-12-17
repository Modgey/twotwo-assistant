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

# Theme Definitions
THEMES = {
    "amber": {
        "main": (255, 191, 0),
        "glow": (255, 210, 100),
        "dim": (255, 191, 0)
    },
    "blue": {
        "main": (0, 191, 255),
        "glow": (100, 220, 255),
        "dim": (0, 191, 255)
    },
    "green": {
        "main": (0, 255, 128),
        "glow": (100, 255, 180),
        "dim": (0, 255, 128)
    },
    "red": {
        "main": (255, 64, 64),
        "glow": (255, 128, 128),
        "dim": (255, 64, 64)
    },
    "purple": {
        "main": (200, 100, 255),
        "glow": (220, 150, 255),
        "dim": (200, 100, 255)
    },
    "cyan": {
        "main": (0, 255, 255),
        "glow": (150, 255, 255),
        "dim": (0, 255, 255)
    },
    "white": {
        "main": (220, 220, 255),
        "glow": (255, 255, 255),
        "dim": (220, 220, 255)
    }
}

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


def get_theme_colors(theme_name: str = "amber") -> dict:
    """Get color palette for specified theme.
    
    Returns:
        dict with keys: main (full), glow, dim
    """
    theme = THEMES.get(theme_name.lower(), THEMES["amber"])
    
    main = theme["main"]
    glow = theme["glow"]
    dim = theme["dim"]
    
    return {
        "main": (*main, 220),       # AMBER_FULL equivalent
        "glow": (*glow, 80),        # AMBER_GLOW equivalent
        "dim": (*dim, 120),         # AMBER_DIM equivalent
        "base": main                # RGB only
    }


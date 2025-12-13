"""TwoTwo Text Display

Pygame-based text display with clean hologram styling.
Prioritizes readability while matching avatar aesthetic.
"""

import pygame
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QWidget

from config import get_config
from ui.theme import (
    AMBER, AMBER_GLOW, 
    FONT_FAMILY, FONT_SIZE_MEDIUM, TYPING_SPEED, FADE_SPEED,
)


class TextDisplay(QWidget):
    """Clean hologram-styled text display using Pygame."""
    
    FONT_SIZE = 20  # Main response text
    
    def __init__(self, parent=None):
        """Create text display."""
        super().__init__(parent)
        
        self.config = get_config()
        
        # Display properties
        self._max_width = 280
        self._padding = 1  # Minimal padding - tight to text
        
        # Colors
        self._text_color = AMBER
        self._glow_color = (*AMBER_GLOW[:3], 50)
        self._shadow_color = (0, 0, 0)
        self._shadow_offset = 5
        self._bg_color = (0, 0, 0, 60)
        self._bg_blur_passes = 2
        
        # Text state
        self._text = ""
        self._display_text = ""
        self._char_index = 0
        self._visible = False
        self._fade_alpha = 0.0
        self._target_alpha = 0.0
        
        # Listening animation state
        self._listening_mode = False
        self._listening_dots = 0
        self._listening_timer = 0.0
        
        # Text alignment: "left", "center", "right"
        self._alignment = "left"
        
        # Initialize pygame
        if not pygame.get_init():
            pygame.init()
        pygame.font.init()
        
        # Load terminal font
        self._font = self._load_font(self.FONT_SIZE)
        
        # Surface
        self._surface = None
        self._rendered_size = (0, 0)
        
        # Widget setup
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(50, 30)
        
        # Animation timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start(16)
        
        self._last_time = pygame.time.get_ticks()
    
    def _load_font(self, size: int):
        """Load VT323 terminal font with fallback."""
        from pathlib import Path
        
        # Try VT323 from assets folder (Fallout/Portal terminal aesthetic)
        try:
            font_path = Path(__file__).parent.parent / "assets" / "fonts" / "VT323-Regular.ttf"
            if font_path.exists():
                # VT323 looks best at slightly larger sizes
                return pygame.font.Font(str(font_path), int(size * 1.2))
        except Exception:
            pass
        
        # Try system VT323
        try:
            return pygame.font.SysFont("VT323", int(size * 1.2))
        except Exception:
            pass
        
        # Fallback to Consolas (similar monospace feel)
        return pygame.font.SysFont("Consolas", size, bold=True)
    
    def set_alignment(self, alignment: str):
        """Set text alignment: 'left', 'center', or 'right'."""
        if alignment not in ("left", "center", "right"):
            alignment = "left"
        self._alignment = alignment
    
    def show_text(self, text: str, animate: bool = True):
        """Show text with optional typing animation."""
        self._listening_mode = False
        self._text = text
        self._char_index = 1 if animate else len(text)
        self._display_text = text[:int(self._char_index)] if text else ""
        self._visible = True
        self._fade_alpha = 1.0
        self._target_alpha = 1.0
        self._update_surface()
        self.show()
        self.raise_()
    
    def show_listening(self):
        """Show animated listening indicator."""
        self._listening_mode = True
        self._listening_dots = 1
        self._listening_timer = 0.0
        self._text = "Listening"
        self._display_text = "Listening."
        self._visible = True
        self._fade_alpha = 1.0
        self._target_alpha = 1.0
        self._update_surface()
        self.show()
        self.raise_()
    
    def stop_listening(self):
        """Stop listening mode."""
        self._listening_mode = False
    
    def hide_text(self):
        """Hide with fade out."""
        self._target_alpha = 0.0
    
    def clear(self):
        """Immediately clear and hide."""
        self._text = ""
        self._display_text = ""
        self._visible = False
        self._fade_alpha = 0.0
        self.hide()
    
    def _on_timer(self):
        """Animation update."""
        current_time = pygame.time.get_ticks()
        dt = (current_time - self._last_time) / 1000.0
        self._last_time = current_time
        
        if not self._visible and self._fade_alpha <= 0:
            return
        
        # Listening animation (animated dots)
        if self._listening_mode:
            self._listening_timer += dt
            if self._listening_timer > 0.4:  # Update every 0.4 seconds
                self._listening_timer = 0.0
                self._listening_dots = (self._listening_dots % 3) + 1
                dots = "." * self._listening_dots
                self._display_text = f"Listening{dots}"
                self._update_surface()
        # Typing animation
        elif self._char_index < len(self._text):
            self._char_index += TYPING_SPEED * dt
            new_display = self._text[:int(self._char_index)]
            if new_display != self._display_text:
                self._display_text = new_display
                self._update_surface()
        
        # Fade
        if self._target_alpha > self._fade_alpha:
            self._fade_alpha = min(self._target_alpha, self._fade_alpha + FADE_SPEED * dt)
        elif self._target_alpha < self._fade_alpha:
            self._fade_alpha = max(self._target_alpha, self._fade_alpha - FADE_SPEED * dt)
            if self._fade_alpha <= 0:
                self._visible = False
                self.hide()
        
        self.update()
    
    def _update_surface(self):
        """Update the pygame surface with current text."""
        if not self._display_text:
            self._surface = None
            self.setFixedSize(1, 1)
            return
        
        # Use FULL text for size calculation (for stable positioning during animation)
        # But only display the partial text
        size_text = self._text if self._text else self._display_text
        display_text = self._display_text
        
        # Word wrap for size calculation (using full text)
        def wrap_text(text):
            words = text.split(' ')
            lines = []
            current_line = ""
            for word in words:
                test_line = current_line + (" " if current_line else "") + word
                w = self._font.size(test_line)[0]
                if w > self._max_width - self._padding * 2:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
            if current_line:
                lines.append(current_line)
            return lines
        
        size_lines = wrap_text(size_text)
        display_lines = wrap_text(display_text)
        
        if not size_lines:
            self._surface = None
            return
        
        # Calculate size based on FULL text (for stable positioning)
        line_height = self._font.get_linesize()
        text_height = line_height * len(size_lines)
        text_width = max(self._font.size(line)[0] for line in size_lines) if size_lines else 0
        
        # Add minimal extra space for glow and shadow
        shadow_padding = 4
        width = text_width + self._padding * 2 + shadow_padding * 2
        height = text_height + self._padding * 2 + shadow_padding * 2
        
        self._rendered_size = (width, height)
        self.setFixedSize(width, height)
        
        # Create surface
        self._surface = pygame.Surface((width, height), pygame.SRCALPHA)
        
        # Draw subtle blurred background first
        self._draw_subtle_background(width, height, shadow_padding)
        
        # Draw DISPLAY text with shadow (with alignment)
        text_y = self._padding + shadow_padding
        left_edge = self._padding + shadow_padding
        right_edge = width - self._padding - shadow_padding
        
        for i, line in enumerate(display_lines):
            line_width = self._font.size(line)[0]
            
            # Calculate x based on alignment
            if self._alignment == "center":
                text_x = (width - line_width) // 2
            elif self._alignment == "right":
                text_x = right_edge - line_width
            else:  # left (default)
                text_x = left_edge
            
            y = text_y + i * line_height
            self._draw_text_with_shadow(line, text_x, y)
    
    def _draw_subtle_background(self, width: int, height: int, padding: int):
        """Draw a very subtle, blurred background behind text."""
        # Calculate tight bounds around text (not full surface)
        text_bounds = pygame.Rect(
            padding - 2,
            padding - 2,
            width - (padding * 2) + 4,
            height - (padding * 2) + 4
        )
        
        # Create background surface
        bg_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        
        # Draw rounded rect with very low alpha
        pygame.draw.rect(
            bg_surface,
            self._bg_color,
            text_bounds,
            border_radius=6
        )
        
        # Simple blur effect: average with shifted versions
        if self._bg_blur_passes > 0:
            blurred = bg_surface.copy()
            for _ in range(self._bg_blur_passes):
                temp = pygame.Surface((width, height), pygame.SRCALPHA)
                # Blend multiple offset copies for blur
                for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                    temp.blit(blurred, (dx, dy))
                temp.set_alpha(200 // 5)
                blurred = temp
            bg_surface = blurred
        
        # Blit to main surface
        self._surface.blit(bg_surface, (0, 0))
    
    def _draw_text_with_shadow(self, text: str, x: int, y: int):
        """Draw text with drop shadow and glow."""
        # Draw drop shadow (multiple layers for depth)
        shadow_surface = self._font.render(text, True, self._shadow_color)
        for offset in range(self._shadow_offset, 0, -1):
            alpha = 80 // offset  # Fade out as we go outward
            shadow_surface.set_alpha(alpha)
            self._surface.blit(shadow_surface, (x + offset, y + offset))
        
        # Draw glow
        glow_offsets = [(-1, -1), (1, -1), (-1, 1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)]
        glow_surface = self._font.render(text, True, self._glow_color[:3])
        glow_surface.set_alpha(30)
        for dx, dy in glow_offsets:
            self._surface.blit(glow_surface, (x + dx, y + dy))
        
        # Main text
        text_surface = self._font.render(text, True, self._text_color)
        self._surface.blit(text_surface, (x, y))
        
        # Extra boldness
        text_surface.set_alpha(200)
        self._surface.blit(text_surface, (x + 1, y))
    
    def paintEvent(self, event):
        """Render to Qt."""
        if self._surface is None or self._fade_alpha <= 0.01:
            return
        
        render_surface = self._surface.copy()
        
        # Apply fade
        if self._fade_alpha < 0.99:
            render_surface.set_alpha(int(self._fade_alpha * 255))
        
        # Convert to QImage
        width, height = render_surface.get_size()
        data = pygame.image.tostring(render_surface, "RGBA")
        qimage = QImage(data, width, height, QImage.Format.Format_RGBA8888)
        
        # Draw
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(0, 0, qimage)
        painter.end()
    
    def cleanup(self):
        """Clean up resources."""
        self._timer.stop()

"""TwoTwo Avatar Renderer

Minimal, cute, expressive avatar inspired by Emo with aperture frame.
"""

import math
import random
import pygame
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPainter, QCursor
from PySide6.QtWidgets import QWidget

from core.state import AvatarState
from avatar.expressions import get_expression, ExpressionParams, EyeParams, ApertureParams
from ui.theme import (
    get_theme_colors,
    SCANLINE_SPACING, SCANLINE_ALPHA
)


class AvatarRenderer(QWidget):
    """Minimal cute avatar with two expressive eyes and aperture frame."""
    
    state_changed = Signal(AvatarState)
    
    def __init__(self, size: int = 200, parent=None):
        super().__init__(parent)
        
        from config import get_config
        self.config = get_config()
        
        self.avatar_size = size
        self.setFixedSize(size, size)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Initialize pygame
        pygame.init()
        
        # Supersampling for anti-aliasing (render at 2x, scale down)
        self._supersample = 2
        self._render_size = size * self._supersample
        self._surface = pygame.Surface((self._render_size, self._render_size), pygame.SRCALPHA)
        
        # Avatar color - from shared theme
        theme_name = self.config.get("ui", "display_color", default="amber")
        colors = get_theme_colors(theme_name)
        
        self._color = colors["main"]
        self._glow_color = colors["glow"]
        self._opacity = 1.0  # 0.0 to 1.0
        
        # Hologram effect settings - from shared theme
        self._scanline_spacing = SCANLINE_SPACING
        self._scanline_alpha = SCANLINE_ALPHA
        self._scanline_phase = 0.0  # Animated scan line movement
        
        # Animation state
        self._current_state = AvatarState.IDLE
        self._current_expression: ExpressionParams = get_expression(AvatarState.IDLE)
        
        # Eye animation state
        self._left_eye_current = EyeParams()
        self._right_eye_current = EyeParams()
        self._spacing_current = 1.0
        
        # Cursor tracking for eye follow (more expressive)
        self._look_offset_x = 0.0
        self._look_offset_y = 0.0
        self._target_look_x = 0.0
        self._target_look_y = 0.0
        self._max_look_offset = 6.0
        
        # Expressive look parameters
        self._look_rotation = 0.0
        self._look_scale_left = 1.0
        self._look_scale_right = 1.0
        self._look_squash = 1.0
        self._target_look_rotation = 0.0
        self._target_scale_left = 1.0
        self._target_scale_right = 1.0
        self._target_squash = 1.0
        
        # Aperture animation state
        self._aperture_openness = 1.0
        self._aperture_rotation = 0.0
        self._aperture_pulse_phase = 0.0
        self._aperture_micro_offset = 0.0  # Subtle random jitter
        
        # Blink state
        self._blink_enabled = True
        self._blink_interval = 4.0
        self._blink_timer = 0.0
        self._blink_progress = 0.0
        self._is_blinking = False
        self._double_blink_pending = False  # For occasional double-blinks
        self._blink_asymmetry = 0.0  # Left eye leads slightly
        
        # Bob animation
        self._bob_phase = 0.0
        
        # Breathing pulse (subtle size oscillation)
        self._breath_phase = 0.0
        self._breath_speed = 0.8  # Hz
        
        # Listening state effects
        self._listen_pulse_phase = 0.0  # Attentive pulse
        self._listen_lean = 0.0  # Subtle lean-in effect
        self._listen_anticipation = 0.0  # Eager anticipation bounce
        
        # Thinking eye darts
        self._think_dart_x = 0.0
        self._think_dart_y = 0.0
        self._think_dart_timer = 0.0
        
        # Speaking happy squint
        self._speak_squint = 1.0
        
        # Audio
        self._audio_amplitude = 0.0
        
        # Apply initial expression
        self._apply_expression(self._current_expression)
        
        # Animation timer (high frame rate for smooth animations)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start(7)  # ~144 FPS
        
        self._last_time = pygame.time.get_ticks()
    
    def set_state(self, state: AvatarState):
        """Set the avatar state."""
        if state != self._current_state:
            self._current_state = state
            self._current_expression = get_expression(state)
            self._apply_expression(self._current_expression)
            self.state_changed.emit(state)
    
    def set_audio_amplitude(self, amplitude: float):
        """Set audio amplitude for speaking animation."""
        self._audio_amplitude = max(0, min(1, amplitude))
    
    def set_opacity(self, opacity: float):
        """Set avatar opacity (0.0 to 1.0)."""
        self._opacity = max(0.0, min(1.0, opacity))
        
    def set_theme(self, theme_name: str):
        """Update avatar theme colors."""
        colors = get_theme_colors(theme_name)
        self._color = colors["main"]
        self._glow_color = colors["glow"]
    
    def _apply_expression(self, expr: ExpressionParams):
        """Apply expression parameters."""
        self._blink_enabled = expr.blink_enabled
        self._blink_interval = expr.blink_interval
    
    def _lerp_eye(self, current: EyeParams, target: EyeParams, t: float) -> EyeParams:
        """Interpolate between two eye states."""
        return EyeParams(
            width=current.width + (target.width - current.width) * t,
            height=current.height + (target.height - current.height) * t,
            roundness=current.roundness + (target.roundness - current.roundness) * t,
            rotation=current.rotation + (target.rotation - current.rotation) * t,
            offset_x=current.offset_x + (target.offset_x - current.offset_x) * t,
            offset_y=current.offset_y + (target.offset_y - current.offset_y) * t,
        )
    
    def _update_cursor_tracking(self, dt: float):
        """Update eye tracking based on cursor position with expressive enhancements."""
        # Only track cursor in IDLE state
        if self._current_state != AvatarState.IDLE:
            self._target_look_x = 0.0
            self._target_look_y = 0.0
            self._target_look_rotation = 0.0
            self._target_scale_left = 1.0
            self._target_scale_right = 1.0
            self._target_squash = 1.0
        else:
            cursor_pos = QCursor.pos()
            widget_pos = self.mapToGlobal(self.rect().center())
            
            dx = cursor_pos.x() - widget_pos.x()
            dy = cursor_pos.y() - widget_pos.y()
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance > 10:
                look_strength = min(1.0, distance / 300.0)
                dx_norm = dx / distance if distance > 0 else 0
                dy_norm = dy / distance if distance > 0 else 0
                
                self._target_look_x = dx_norm * self._max_look_offset * look_strength
                self._target_look_y = dy_norm * self._max_look_offset * look_strength
                
                horizontal_tilt = dx_norm * 3.0 * look_strength
                vertical_tilt = dy_norm * 2.5 * look_strength
                self._target_look_rotation = horizontal_tilt + vertical_tilt
                
                perspective_amount = 0.08 * look_strength
                if dx > 0:
                    self._target_scale_left = 1.0 - perspective_amount * 0.5
                    self._target_scale_right = 1.0 + perspective_amount
                else:
                    self._target_scale_left = 1.0 + perspective_amount
                    self._target_scale_right = 1.0 - perspective_amount * 0.5
                
                vertical_strain = abs(dy_norm) * look_strength
                self._target_squash = 1.0 - vertical_strain * 0.1
            else:
                self._target_look_x = 0.0
                self._target_look_y = 0.0
                self._target_look_rotation = 0.0
                self._target_scale_left = 1.0
                self._target_scale_right = 1.0
                self._target_squash = 1.0
        
        lerp_speed = 8.0 * dt
        self._look_offset_x += (self._target_look_x - self._look_offset_x) * lerp_speed
        self._look_offset_y += (self._target_look_y - self._look_offset_y) * lerp_speed
        self._look_rotation += (self._target_look_rotation - self._look_rotation) * lerp_speed
        self._look_scale_left += (self._target_scale_left - self._look_scale_left) * lerp_speed
        self._look_scale_right += (self._target_scale_right - self._look_scale_right) * lerp_speed
        self._look_squash += (self._target_squash - self._look_squash) * lerp_speed
    
    def _update_blink(self, dt: float):
        """Update blink animation with occasional double-blinks."""
        if self._blink_enabled and not self._is_blinking:
            self._blink_timer += dt
            trigger_time = self._blink_interval + random.uniform(-1, 1)
            
            if self._blink_timer >= trigger_time:
                self._is_blinking = True
                self._blink_timer = 0
                self._blink_progress = 0
                # 15% chance of double-blink
                self._double_blink_pending = random.random() < 0.15
                # Random asymmetry (left eye leads slightly)
                self._blink_asymmetry = random.uniform(0, 0.08)
        
        if self._is_blinking:
            speed = 8.0 if self._blink_progress < 0.5 else 6.0
            self._blink_progress += dt * speed
            
            if self._blink_progress >= 1.0:
                self._blink_progress = 0.0
                self._is_blinking = False
                
                # Trigger second blink if pending
                if self._double_blink_pending:
                    self._double_blink_pending = False
                    self._is_blinking = True
                    self._blink_progress = 0
    
    def _get_blink_scale(self, is_left: bool = True) -> float:
        """Get vertical scale for blink with asymmetry."""
        if not self._is_blinking:
            return 1.0
        
        # Apply asymmetry (left eye leads)
        progress = self._blink_progress
        if is_left:
            progress = min(1.0, progress + self._blink_asymmetry)
        
        if progress <= 0.5:
            return 1.0 - (progress * 2)
        else:
            return (progress - 0.5) * 2
    
    def _update_micro_expressions(self, dt: float):
        """Update subtle micro-expressions for more life-like feel."""
        # Breathing pulse
        self._breath_phase += dt * self._breath_speed * 2 * math.pi
        self._breath_phase %= 2 * math.pi
        
        # Aperture micro-jitter during idle
        if self._current_state == AvatarState.IDLE:
            self._aperture_micro_offset += (random.uniform(-0.5, 0.5) - self._aperture_micro_offset) * dt * 2.0
        else:
            self._aperture_micro_offset *= 0.95
        
        # Listening state effects - focused and attentive, minimal movement
        if self._current_state == AvatarState.LISTENING:
            # Slow, subtle drift - like holding focus but alive
            self._listen_pulse_phase += dt * 0.4 * 2 * math.pi
            self._listen_pulse_phase %= 2 * math.pi
            
            # Very subtle asymmetric drift (right eye drifts slightly)
            self._listen_lean = math.sin(self._listen_pulse_phase) * 0.3
            
            # Minimal vertical - just a hint of life, not bouncy
            self._listen_anticipation = math.sin(self._listen_pulse_phase * 0.7) * 0.4
        else:
            self._listen_pulse_phase = 0.0
            self._listen_lean *= 0.9
            self._listen_anticipation *= 0.9
        
        # Thinking eye darts
        if self._current_state == AvatarState.THINKING:
            self._think_dart_timer += dt
            if self._think_dart_timer > random.uniform(0.8, 2.0):
                self._think_dart_timer = 0
                self._think_dart_x = random.uniform(-3, 3)
                self._think_dart_y = random.uniform(-2, 2)
        else:
            self._think_dart_timer = 0
        self._think_dart_x *= 0.92
        self._think_dart_y *= 0.92
        
        # Speaking happy squint
        if self._current_state == AvatarState.SPEAKING:
            target_squint = 0.92 - self._audio_amplitude * 0.05
            self._speak_squint += (target_squint - self._speak_squint) * dt * 5.0
        else:
            self._speak_squint += (1.0 - self._speak_squint) * dt * 5.0
    
    def _on_timer(self):
        """Animation timer callback."""
        current_time = pygame.time.get_ticks()
        dt = (current_time - self._last_time) / 1000.0
        self._last_time = current_time
        dt = min(dt, 0.1)
        
        aperture = self._current_expression.aperture
        
        # Update all animations
        self._update_cursor_tracking(dt)
        self._update_micro_expressions(dt)
        
        # Smooth interpolation to target expression
        lerp_speed = 5.0 * dt
        self._left_eye_current = self._lerp_eye(
            self._left_eye_current, 
            self._current_expression.left_eye, 
            lerp_speed
        )
        self._right_eye_current = self._lerp_eye(
            self._right_eye_current, 
            self._current_expression.right_eye, 
            lerp_speed
        )
        
        spacing_diff = self._current_expression.eye_spacing - self._spacing_current
        self._spacing_current += spacing_diff * lerp_speed
        
        # Update aperture
        openness_diff = aperture.openness - self._aperture_openness
        self._aperture_openness += openness_diff * lerp_speed
        
        if aperture.rotation_speed > 0:
            self._aperture_rotation += aperture.rotation_speed * dt
            self._aperture_rotation %= 360
        else:
            rot_diff = aperture.rotation - self._aperture_rotation
            self._aperture_rotation += rot_diff * lerp_speed
        
        if aperture.pulse_speed > 0:
            self._aperture_pulse_phase += aperture.pulse_speed * dt * 2 * math.pi
            self._aperture_pulse_phase %= 2 * math.pi
        
        # Update blink
        self._update_blink(dt)
        
        # Update bob
        if self._current_expression.bob_speed > 0:
            self._bob_phase += dt * self._current_expression.bob_speed * 2 * math.pi
            self._bob_phase %= 2 * math.pi
        
        # Animate scan lines (slow downward drift)
        self._scanline_phase += dt * 15  # Pixels per second
        self._scanline_phase %= self._scanline_spacing * 10
        
        self.update()
    
    def _draw_aperture(self, surface: pygame.Surface, center_x: float, center_y: float, ss: int = 1):
        """Draw the aperture lens frame."""
        aperture = self._current_expression.aperture
        
        outer_radius = 75 * ss
        inner_radius = 45 * ss
        num_blades = 6
        
        pulse_offset = math.sin(self._aperture_pulse_phase) * aperture.pulse_amount
        current_openness = self._aperture_openness + pulse_offset
        
        if self._current_state == AvatarState.SPEAKING:
            current_openness += self._audio_amplitude * 0.08
        
        adjusted_inner = inner_radius * current_openness
        
        blade_angle = 360 / num_blades
        gap_ratio = 0.16
        slant_offset = 24
        
        # Add micro-jitter to rotation
        rotation = self._aperture_rotation + self._aperture_micro_offset
        
        for i in range(num_blades):
            angle = i * blade_angle + rotation
            
            half_blade = blade_angle * (0.5 - gap_ratio / 2)
            outer_start = math.radians(angle - half_blade)
            outer_end = math.radians(angle + half_blade)
            
            inner_gap_adjust = gap_ratio * (outer_radius / adjusted_inner) if adjusted_inner > 0 else gap_ratio
            inner_half_blade = blade_angle * (0.5 - inner_gap_adjust / 2)
            
            inner_start = math.radians(angle - inner_half_blade + slant_offset)
            inner_end = math.radians(angle + inner_half_blade + slant_offset)
            
            points = []
            
            for t in range(5):
                a = outer_start + (outer_end - outer_start) * (t / 4)
                x = center_x + outer_radius * math.cos(a)
                y = center_y + outer_radius * math.sin(a)
                points.append((x, y))
            
            for t in range(4, -1, -1):
                a = inner_start + (inner_end - inner_start) * (t / 4)
                x = center_x + adjusted_inner * math.cos(a)
                y = center_y + adjusted_inner * math.sin(a)
                points.append((x, y))
            
            if len(points) >= 3:
                pygame.draw.polygon(surface, self._color, points)
    
    def _draw_eye(self, surface: pygame.Surface, center_x: float, center_y: float, 
                  eye_params: EyeParams, blink_scale: float, look_scale: float = 1.0, ss: int = 1):
        """Draw a single eye with expressive enhancements."""
        base_w = 20 * ss
        base_h = 38 * ss
        
        w = base_w * eye_params.width
        h = base_h * eye_params.height
        
        # Apply perspective scale from cursor tracking
        w *= look_scale
        h *= look_scale
        
        # Apply breathing pulse (subtle)
        breath_scale = 1.0 + math.sin(self._breath_phase) * 0.015
        w *= breath_scale
        h *= breath_scale
        
        # Apply listening state pulse (subtle, focused)
        if self._current_state == AvatarState.LISTENING:
            listen_scale = 1.0 + math.sin(self._listen_pulse_phase) * 0.012
            h *= listen_scale
        
        # Apply vertical squash from looking up/down
        h *= self._look_squash
        
        # Apply speaking happy squint
        h *= self._speak_squint
        
        # Apply audio amplitude for speaking
        if self._current_state == AvatarState.SPEAKING and self._audio_amplitude > 0:
            audio_scale = 1.0 + self._audio_amplitude * 0.2
            h *= audio_scale
        
        # Apply blink
        h *= blink_scale
        
        if h < 2 * ss:
            line_start = (int(center_x - w/2), int(center_y))
            line_end = (int(center_x + w/2), int(center_y))
            pygame.draw.line(surface, self._color, line_start, line_end, 2 * ss)
            return
        
        # Apply position offset (expression + cursor + thinking darts)
        offset_x = eye_params.offset_x + self._look_offset_x + self._think_dart_x
        offset_y = eye_params.offset_y + self._look_offset_y + self._think_dart_y
        
        center_x += offset_x * ss
        center_y += offset_y * ss
        
        total_rotation = eye_params.rotation + self._look_rotation
        
        eye_surf = pygame.Surface((int(w + 20), int(h + 20)), pygame.SRCALPHA)
        eye_rect = pygame.Rect(10, 10, int(w), int(h))
        
        border_radius = int(min(w, h) / 2 * eye_params.roundness)
        
        pygame.draw.rect(eye_surf, self._color, eye_rect, border_radius=border_radius)
        
        if abs(total_rotation) > 0.1:
            eye_surf = pygame.transform.rotate(eye_surf, total_rotation)
        
        surf_rect = eye_surf.get_rect(center=(int(center_x), int(center_y)))
        surface.blit(eye_surf, surf_rect)
    
    def _apply_hologram_effects(self, surface: pygame.Surface) -> pygame.Surface:
        """Apply drop shadow, hologram scan lines and glow effect."""
        w, h = surface.get_size()
        
        # Create drop shadow (only where avatar exists)
        shadow_surface = pygame.Surface((w, h), pygame.SRCALPHA)
        shadow_offset = 4
        shadow_opacity = 50  # 0-255
        
        # Get alpha mask from surface to know where avatar is
        alpha_mask = pygame.surfarray.pixels_alpha(surface).copy()
        
        # Create shadow by blitting a darkened, offset copy
        for blur in range(3):  # Soft blur
            ox = shadow_offset + blur
            oy = shadow_offset + blur
            
            # Create a black silhouette of the avatar
            shadow_layer = pygame.Surface((w, h), pygame.SRCALPHA)
            shadow_layer_alpha = pygame.surfarray.pixels_alpha(shadow_layer)
            
            # Copy alpha from original, offset by shadow amount
            src_w = min(w - ox, w)
            src_h = min(h - oy, h)
            if src_w > 0 and src_h > 0:
                shadow_layer_alpha[ox:ox+src_w, oy:oy+src_h] = (
                    alpha_mask[:src_w, :src_h].astype(float) * (shadow_opacity / 255) / (blur + 1)
                ).astype('uint8')
            del shadow_layer_alpha
            
            # Fill the shadow layer with black (keeping alpha)
            shadow_layer.fill((0, 0, 0), special_flags=pygame.BLEND_RGB_MULT)
            shadow_surface.blit(shadow_layer, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Create soft feathered glow with multiple layers at different distances
        glow_surface = pygame.Surface((w, h), pygame.SRCALPHA)
        
        # Multiple glow layers - further = more transparent (feathered effect)
        glow_layers = [
            # (offsets, intensity)
            ([(-1, -1), (1, -1), (-1, 1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)], 0.12),
            ([(-2, -2), (2, -2), (-2, 2), (2, 2), (0, -2), (0, 2), (-2, 0), (2, 0)], 0.08),
            ([(-3, -3), (3, -3), (-3, 3), (3, 3), (0, -3), (0, 3), (-3, 0), (3, 0)], 0.05),
            ([(-4, -4), (4, -4), (-4, 4), (4, 4), (0, -4), (0, 4), (-4, 0), (4, 0)], 0.03),
            ([(-5, 0), (5, 0), (0, -5), (0, 5)], 0.02),
        ]
        
        for offsets, intensity in glow_layers:
            layer = pygame.Surface((w, h), pygame.SRCALPHA)
            for ox, oy in offsets:
                layer.blit(surface, (ox, oy), special_flags=pygame.BLEND_RGBA_ADD)
            # Reduce this layer's intensity
            layer_alpha = pygame.surfarray.pixels_alpha(layer)
            layer_alpha[:] = (layer_alpha * intensity).astype('uint8')
            del layer_alpha
            glow_surface.blit(layer, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Composite: shadow behind, then glow, then main
        result = pygame.Surface((w, h), pygame.SRCALPHA)
        result.blit(shadow_surface, (0, 0))
        result.blit(glow_surface, (0, 0))
        result.blit(surface, (0, 0))
        
        # Apply subtle scan lines only where avatar exists
        result_array = pygame.surfarray.pixels3d(result)
        alpha_array = pygame.surfarray.pixels_alpha(result)
        
        scanline_offset = int(self._scanline_phase) % self._scanline_spacing
        for y in range(scanline_offset, h, self._scanline_spacing):
            mask = alpha_array[:, y] > 0
            # Subtle darken (0.88 = only 12% darker)
            darken_factor = 0.88
            result_array[:, y, 0][mask] = (result_array[:, y, 0][mask] * darken_factor).astype('uint8')
            result_array[:, y, 1][mask] = (result_array[:, y, 1][mask] * darken_factor).astype('uint8')
            result_array[:, y, 2][mask] = (result_array[:, y, 2][mask] * darken_factor).astype('uint8')
        
        del result_array
        del alpha_array
        
        return result
    
    def paintEvent(self, event):
        """Paint the avatar with supersampled anti-aliasing and hologram effects."""
        self._surface.fill((0, 0, 0, 0))
        
        ss = self._supersample
        
        center_x = self._render_size / 2
        center_y = self._render_size / 2
        
        bob_y = 0
        if self._current_expression.bob_speed > 0:
            bob_y = math.sin(self._bob_phase) * self._current_expression.bob_amount * ss
        
        self._draw_aperture(self._surface, center_x, center_y + bob_y, ss)
        
        base_spacing = 28 * ss
        spacing = base_spacing * self._spacing_current
        
        left_x = center_x - spacing / 2
        right_x = center_x + spacing / 2
        eye_y = center_y + bob_y
        
        # Get asymmetric blink scales
        blink_scale_left = self._get_blink_scale(is_left=True)
        blink_scale_right = self._get_blink_scale(is_left=False)
        
        # Draw eyes
        self._draw_eye(self._surface, left_x, eye_y, self._left_eye_current, blink_scale_left, self._look_scale_left, ss)
        self._draw_eye(self._surface, right_x, eye_y, self._right_eye_current, blink_scale_right, self._look_scale_right, ss)
        
        # Scale down for anti-aliasing
        scaled_surface = pygame.transform.smoothscale(self._surface, (self.avatar_size, self.avatar_size))
        
        # Apply hologram effects
        hologram_surface = self._apply_hologram_effects(scaled_surface)
        
        # Apply opacity
        if self._opacity < 1.0:
            alpha = pygame.surfarray.pixels_alpha(hologram_surface)
            alpha[:] = (alpha * self._opacity).astype('uint8')
            del alpha
        
        data = pygame.image.tostring(hologram_surface, "RGBA")
        qimage = QImage(data, self.avatar_size, self.avatar_size, QImage.Format.Format_RGBA8888)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(0, 0, qimage)
        painter.end()
    
    def cleanup(self):
        """Clean up pygame resources."""
        self._timer.stop()
        pygame.quit()
    
    @property
    def current_state(self) -> AvatarState:
        """Get the current avatar state."""
        return self._current_state

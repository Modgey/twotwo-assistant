# Project TwoTwo: Local AI Assistant

## Overview

TwoTwo is a local AI assistant inspired by Jarvis from Iron Man. It features a futuristic holographic-style interface that floats on top of other windows, with real-time voice interaction and an animated avatar with expressive eyes.

**Core Philosophy:** Everything runs locally for privacy and low latency. No cloud dependencies except optional web search.

**Current Status:** Phases 1-6 complete. Ultra-low latency speech-to-speech pipeline with comprehensive performance optimizations. The system achieves near real-time response times with comprehensive timing instrumentation and debugging capabilities.

---

## Technical Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| Language | Python 3.11+ | Primary development language |
| UI Framework | PySide6 | Transparent overlay window, settings panel, UI chrome |
| Avatar Rendering | Pygame (embedded in PySide6) | Animated aperture-eye avatar |
| LLM Backend | Ollama / llama.cpp | Local language model inference |
| STT | whisper.cpp | Local speech-to-text |
| TTS | Piper | Local text-to-speech |
| Web Search | Brave Search API | Optional internet search capability |

### Why This Stack

- **PySide6:** Native transparent frameless window support on Windows, good animation capabilities, can embed other renderers
- **Pygame embedded:** Creative control over avatar rendering while PySide6 handles windowing
- **whisper.cpp:** Fast local transcription, `base` or `small` model balances speed/accuracy
- **Piper:** Fastest local TTS option, low latency, multiple voice options
- **Ollama:** Easy model management, auto-detection of installed models

---

## Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        TwoTwo Application                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Overlay UI    │  │  Settings Panel │  │   System Tray   │  │
│  │   (PySide6)     │  │   (PySide6)     │  │   (PySide6)     │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
│           │                    │                    │           │
│  ┌────────▼────────────────────▼────────────────────▼────────┐  │
│  │                    Core Controller                         │  │
│  │         (State Management, Event Coordination)             │  │
│  └────────┬────────────────────┬────────────────────┬────────┘  │
│           │                    │                    │           │
│  ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐  │
│  │  Avatar Engine  │  │   Voice Engine  │  │    AI Engine    │  │
│  │    (Pygame)     │  │ (whisper/Piper) │  │ (Ollama/llama)  │  │
│  └─────────────────┘  └─────────────────┘  └────────┬────────┘  │
│                                                     │           │
│                                            ┌────────▼────────┐  │
│                                            │  Search Engine  │  │
│                                            │  (Brave API)    │  │
│                                            └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
twotwo/
├── main.py                     # Application entry point
├── config.py                   # Configuration management
├── requirements.txt            # Python dependencies
│
├── core/
│   ├── __init__.py
│   ├── controller.py           # Main application controller
│   ├── state.py                # Application state management
│   └── hotkey.py               # Global hotkey handling
│
├── ui/
│   ├── __init__.py
│   ├── overlay_window.py       # Main transparent overlay window
│   ├── settings_window.py      # Settings/config panel
│   ├── tray_icon.py            # System tray icon
│   └── text_display.py         # Response text streaming widget
│
├── avatar/
│   ├── __init__.py
│   ├── renderer.py             # Pygame avatar renderer with hologram effects
│   └── expressions.py          # Expression state definitions and parameters
│
├── voice/
│   ├── __init__.py
│   ├── stt.py                  # Speech-to-text (whisper.cpp)
│   ├── tts.py                  # Text-to-speech (Piper)
│   └── audio.py                # Audio capture/playback utilities
│
├── ai/
│   ├── __init__.py
│   ├── llm.py                  # LLM interface (Ollama/llama.cpp)
│   ├── model_manager.py        # Model detection and selection
│   └── search.py               # Brave Search integration
│
└── assets/
    ├── fonts/                  # UI fonts
    ├── sounds/                 # UI sounds (optional)
    └── icons/                  # App icons
```

---

## Feature Specifications

### 1. Overlay Window

**File:** `ui/overlay_window.py`

**Requirements:**
- Frameless, transparent PySide6 window
- Always on top (`Qt.WindowStaysOnTopHint`)
- Click-through except on avatar area (`Qt.WA_TranslucentBackground`)
- Works with borderless windowed applications and games
- Does NOT work with exclusive fullscreen (documented limitation)

**Implementation Notes:**
```python
# Key window flags for transparent overlay
self.setWindowFlags(
    Qt.FramelessWindowHint |
    Qt.WindowStaysOnTopHint |
    Qt.Tool  # Prevents taskbar entry
)
self.setAttribute(Qt.WA_TranslucentBackground)
self.setAttribute(Qt.WA_ShowWithoutActivating)
```

**Dragging Behavior:**
- Avatar can be dragged while holding `Alt` key
- Without `Alt`, clicks pass through to underlying windows
- Save position to config on drag end

---

### 2. Avatar Rendering

**Files:** `avatar/renderer.py`, `avatar/expressions.py`

**Visual Design:**
- Emo-inspired minimal design with two expressive eyes
- Aperture lens frame (6-blade camera aperture) surrounding the eyes
- Amber color scheme (RGB: 255, 191, 0) with transparency
- Hologram effects:
  - Animated scan lines (subtle, drifting downward)
  - Soft feathered glow around avatar
  - Drop shadow for visibility on busy backgrounds
- 2x supersampling for anti-aliasing (crisp edges)
- Rendered at 144 FPS for smooth animations

**Avatar States & Animations:**

| State | Aperture Behavior | Eye Behavior | Trigger |
|-------|-------------------|--------------|---------|
| Idle | Slow breathing pulse | Cursor tracking, gentle breathing, occasional double-blinks | Default state |
| Listening | Opens wider, subtle pulse | Focused, asymmetric tilt, minimal movement | Push-to-talk active |
| Thinking | Slow rotation, gentle pulse | Squinted, occasional eye darts, looks up-right | Waiting for LLM response |
| Speaking | Slightly open, audio-reactive | Happy squint, bounces with speech rhythm | TTS playing |

**Micro-Expressions:**
- Breathing pulse (subtle size oscillation)
- Double-blinks (15% chance, natural feel)
- Asymmetric blinks (left eye leads slightly)
- Thinking eye darts (quick glances while processing)
- Speaking happy squint (eyes narrow slightly when talking)
- Cursor tracking in idle (eyes follow mouse with perspective/rotation effects)

**Pygame Embedding in PySide6:**
```python
# Embed pygame surface in QWidget with supersampling
class AvatarRenderer(QWidget):
    def __init__(self, size=200):
        self._supersample = 2
        self._render_size = size * self._supersample
        self._surface = pygame.Surface((self._render_size, self._render_size), pygame.SRCALPHA)
        
    def paintEvent(self, event):
        # Render at 2x, scale down for anti-aliasing
        # Apply hologram effects (glow, scan lines, shadow)
        # Convert to QImage and display
```

**Avatar Size:** ~200x200 pixels (configurable: small/medium/large)

---

### 3. Voice Engine

#### 3.1 Speech-to-Text (STT)

**File:** `voice/stt.py`

**Technology:** whisper.cpp via `pywhispercpp` or subprocess

**Model:** Auto-detected from installed models in `%APPDATA%/TwoTwo/whisper-models/`
- Settings panel dynamically shows only available models
- Common models: `tiny`, `base`, `small`, `medium`
- Model can be switched live without restart

**Behavior:**
- Activated by push-to-talk hotkey (`X` key)
- Records audio while hotkey held
- Processes on hotkey release
- Returns transcribed text to controller

**Implementation Approach:**
```python
# Using pywhispercpp bindings
from pywhispercpp.model import Model

model = Model('base', n_threads=4)
result = model.transcribe(audio_file)
```

#### 3.2 Text-to-Speech (TTS)

**File:** `voice/tts.py`

**Technology:** Piper TTS

**Requirements:**
- Low latency is priority
- Stream audio as it's generated (don't wait for full synthesis)
- Natural-sounding voice

**Voice Model:** Start with `en_US-lessac-medium` (good quality/speed balance)

**Performance Optimizations:**
- TTS speed default: 1.25x (configurable 0.5x-1.5x)
- Minimum recording duration: 0.15s (reduced from 0.3s)
- Audio buffer size: 512 samples (reduced from 1024 for lower latency)

**Implementation Approach:**
```python
# Piper can stream output
import subprocess

process = subprocess.Popen(
    ['piper', '--model', model_path, '--output-raw'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE
)
# Stream text in, stream audio out
```

---

### 4. AI Engine

#### 4.1 LLM Interface

**File:** `ai/llm.py`

**Supported Backends:**
1. **Ollama** 
2. **llama.cpp** 

User can select the backend and model from the settings panel.

**Ollama Integration:**
```python
import requests

def chat(messages, model="gemma3:4b"):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": model, "messages": messages, "stream": True},
        stream=True
    )
    # Yield tokens as they arrive for streaming display
```

**Target Models:** Small models for good performance on consumer hardware
- `gemma3:4b`
- `gemma3:12b`

#### 4.2 Model Detection

**File:** `ai/model_manager.py`

**Requirements:**
- Auto-detect installed Ollama models via `ollama list` API
- Auto-detect llama.cpp models in configured directory
- Present unified list in settings

**Ollama Detection:**
```python
def get_ollama_models():
    response = requests.get("http://localhost:11434/api/tags")
    return [model['name'] for model in response.json()['models']]
```

#### 4.3 Web Search

**File:** `ai/search.py`

**Technology:** Brave Search API

**Behavior:**
- LLM autonomously decides when to search
- Inject search capability via system prompt
- Parse and summarize results before feeding back to LLM

**System Prompt Integration:**
```
You have access to web search. When you need current information or facts you're uncertain about, indicate you want to search by responding with:
[SEARCH: your search query]

The search results will be provided, then continue your response.
```

**Implementation:**
```python
import requests

def brave_search(query, api_key):
    response = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"X-Subscription-Token": api_key},
        params={"q": query, "count": 5}
    )
    return response.json()['web']['results']
```

---

### 5. Settings Panel

**File:** `ui/settings_window.py`

**Access:** Right-click on avatar or system tray icon

**Settings Categories:**

#### General
- Avatar position (top left / top right / bottom left / bottom right) - or "Reset to default"
- Avatar size (small / medium / large) - requires restart to apply
- Opacity slider (50% - 100%) - applies live

#### Voice
- Push-to-talk hotkey (key capture)
- STT model selection (dynamically populated from installed models)
- TTS voice selection (dropdown of available Piper voices)
- TTS speed slider (50-150%, default 125%)
- Input/output device selection

#### AI
- LLM backend (Ollama / llama.cpp)
- Model selection (dropdown, auto-populated)
- Personality/system prompt (multiline text edit)
- Brave Search API key (password field)
- Enable/disable web search

#### About
- Version info
- Links to documentation

**Config Persistence:**
- Save to `config.json` in app directory or `%APPDATA%/TwoTwo/`
- Load on startup

---

### 6. Global Hotkey

**File:** `core/hotkey.py`

**Technology:** `pynput` or `keyboard` library

**Hotkeys:**
| Key | Action |
|-----|--------|
| `X` | Push-to-talk (hold to record) |
| `Alt + Drag` | Move avatar |
| `Escape` | Cancel current operation |

**Implementation Notes:**
- Must work when application is not focused
- Use low-level keyboard hooks on Windows

---

### 7. Response Display

**File:** `ui/text_display.py`

**Requirements:**
- Show streaming text response next to avatar
- Fade in/out animation
- Auto-dismiss after TTS completes (configurable delay)
- Semi-transparent background
- Positioned relative to avatar (under or beside)

**Behavior:**
- Appears when response starts streaming
- Text appears character-by-character or word-by-word
- Remains visible during TTS playback
- Fades out 2-3 seconds after TTS ends

---

## Interaction Flow

### Complete Voice Interaction Sequence

```
1. User holds X key
   └─► Avatar: Transition to LISTENING state
   └─► Audio: Begin recording from microphone

2. User speaks, releases X key
   └─► Audio: Stop recording
   └─► Avatar: Transition to THINKING state
   └─► STT: Process audio → text

3. STT complete
   └─► Controller: Send text to LLM
   └─► Text Display: Show user's transcribed text briefly

4. LLM streams response
   └─► Text Display: Stream response text
   └─► If [SEARCH: query] detected:
       └─► Search Engine: Execute search
       └─► LLM: Continue with search results

5. LLM complete
   └─► Avatar: Transition to SPEAKING state
   └─► TTS: Synthesize and play audio
   └─► Avatar: Animate eye with audio amplitude

6. TTS complete
   └─► Avatar: Transition to IDLE state
   └─► Text Display: Fade out after delay
```

---

## Configuration Schema

**File:** `config.json`

```json
{
  "version": "1.0",
  
  "ui": {
    "avatar_position": {"x": 100, "y": 100},
    "avatar_size": "medium",
    "opacity": 0.85,
    "text_display_duration": 3.0
  },
  
  "voice": {
    "hotkey": "x",
    "stt_model": "base",
    "tts_voice": "en_US-lessac-medium",
    "tts_speed": 1.0,
    "input_device": null,
    "output_device": null
  },
  
  "ai": {
    "backend": "ollama",
    "model": "llama3.2:3b",
    "personality": "You are TwoTwo, a helpful AI assistant...",
    "brave_api_key": "",
    "enable_search": true
  }
}
```

---

## Dependencies

### requirements.txt

```
# UI
PySide6>=6.5.0
pygame>=2.5.0
numpy>=1.24.0

# Voice (Phase 3)
# pywhispercpp>=1.0.0
# piper-tts>=1.0.0
# sounddevice>=0.4.6

# AI (Phase 4)
# requests>=2.31.0

# System (Phase 3)
# pynput>=1.7.6
```

**Note:** Currently only Phase 1 & 2 dependencies are included. Voice and AI dependencies will be added in future phases.

### External Dependencies (must be installed separately)

1. **Ollama** - https://ollama.ai
   - Install and run `ollama serve`
   - Pull desired models: `ollama pull llama3.2:3b`

2. **Piper TTS Models** - https://github.com/rhasspy/piper
   - Download voice models to `assets/piper-voices/`

3. **Whisper.cpp Models** - https://github.com/ggerganov/whisper.cpp
   - Download `ggml-base.bin` or `ggml-small.bin` to `assets/whisper-models/`

---

## Development Phases

### Phase 1: Foundation ✅ COMPLETE
- [x] Project setup, directory structure
- [x] Config management system
- [x] Basic PySide6 transparent overlay window
- [x] System tray icon with quit option

### Phase 2: Avatar ✅ COMPLETE
- [x] Pygame embedding in PySide6
- [x] Aperture lens frame rendering (6-blade design)
- [x] Two-eye system with expressive animations
- [x] Idle animation loop with cursor tracking
- [x] All avatar states (listening, thinking, speaking)
- [x] Alt+drag movement
- [x] Hologram effects (scan lines, glow, drop shadow)
- [x] Micro-expressions (breathing, double-blinks, eye darts)
- [x] 2x supersampling anti-aliasing
- [x] 144 FPS rendering

**Implemented Features:**
- Amber color scheme with transparency
- Cursor tracking in idle state (eyes follow mouse with perspective effects)
- Smooth state transitions with expression interpolation
- Breathing pulse animation
- Double-blink system (15% chance)
- Asymmetric blinking (left eye leads)
- Thinking state eye darts
- Speaking state happy squint
- Drop shadow for visibility
- Soft feathered glow effect
- Animated hologram scan lines

### Phase 3: Voice ✅ COMPLETE
- [x] Audio capture with sounddevice
- [x] Whisper.cpp STT integration
- [x] Piper TTS integration
- [x] Push-to-talk hotkey (pynput)
- [x] Audio amplitude extraction for avatar sync

**Implemented Features:**
- `voice/audio.py` - AudioRecorder and AudioPlayer with real-time amplitude callbacks
- `voice/stt.py` - WhisperSTT with pywhispercpp bindings and subprocess fallback
- `voice/tts.py` - PiperTTS with streaming synthesis and TTSQueue for managed playback
- `core/hotkey.py` - Global hotkey handler with push-to-talk (X key) and cancel (Escape)
- Full integration in Controller with state management and avatar sync

### Phase 4: UI Polish ✅ COMPLETE
- [x] Settings panel (all sections)
- [x] Response text display with streaming
- [x] Smooth state transitions
- [x] Error handling and user feedback
- [x] Smart text positioning and alignment

**Implemented Features:**
- `ui/settings_window.py` - Premium minimal black/white settings panel
  - Voice style selection (Normal, Robot, Claptrap, GLaDOS, Metallic)
  - TTS speed slider (50-150%, default 125%)
  - STT model selection (dynamically populated from installed models)
  - PTT key capture
  - Avatar size and opacity controls (opacity works live, size requires restart)
  - Draggable, frameless window with amber accents
  - Screen boundary clamping to prevent window going off-screen
- `ui/text_display.py` - Pygame-based hologram text display
  - Shared theme with avatar (amber colors, effects)
  - VT323 terminal font (Fallout/Portal aesthetic, 24px main text)
  - Typing animation with stable positioning
  - Smart adaptive positioning:
    - Left/right/center based on avatar screen position
    - Text alignment matches position (left/right/center)
    - Fixed-size container prevents shifting during animation
    - Optimized spacing to keep text close to avatar without overlap
  - Subtle blurred background, drop shadow, glow effects
  - Fade in/out transitions
- `ui/theme.py` - Shared visual constants (colors, effects, typography)
- Full integration with controller and overlay window

### Phase 5: AI ✅ COMPLETE
- [x] Ollama integration with streaming
- [x] Model auto-detection and auto-start
- [x] Brave Search integration
- [x] Autonomous search decision logic
- [x] Sentence-level streaming TTS for real-time response
- [x] LLM pre-warming for reduced latency
- [x] Model persistence across sessions
- [x] Tabbed settings UI with AI configuration

**Implemented Features:**
- `ai/llm.py` - OllamaLLM with streaming chat, conversation history, and pre-warming
- `ai/model_manager.py` - Auto-detection of Ollama models, auto-start functionality, best model selection
- `ai/search.py` - Brave Search API integration with autonomous search decision logic
- Continuous streaming TTS queue - seamless sentence playback without gaps
- Aggressive sentence chunking - starts speaking after 8-40 chars for low latency
- Settings UI redesign - tabbed interface (Voice/AI/Display) with larger, more usable controls

**Performance Improvements (Phases 5-6):**
- Fixed critical TTS audio buffer bug causing exponential slowdown after multiple uses
- Implemented continuous streaming architecture - audio stream stays open for seamless playback
- Sentence-level streaming with 4-tier chunking strategy (punctuation → newlines → pauses → length)
- LLM model pre-warming on startup and model switch to eliminate cold start latency
- Optimized sentence extraction thresholds (8-40 chars) for faster speech initiation
- Model persistence - selected model saved and restored on restart
- **ULTRA-LOW LATENCY OPTIMIZATION:** 8x faster LLM response time via HTTP pooling and DNS bypass
- **Timing Instrumentation:** Comprehensive per-stage latency tracking for debugging
- **Model Keep-Alive:** Background thread prevents model unloading between requests
- **Conversation History Limiting:** Last 4 messages only to reduce prompt processing overhead

### Phase 6: Ultra-Low Latency Optimization ✅ COMPLETE
- [x] Comprehensive timing instrumentation (baseline measurement)
- [x] HTTP connection pooling and DNS optimization
- [x] Model keep-alive system to prevent unloading
- [x] Conversation history limiting for faster prompt processing
- [x] Achieved 8x faster LLM response time (2.3s → 0.29s first token)
- [x] End-to-end latency reduced from 2.9s to 0.96s to first speech
- [x] Performance optimization (TTS buffer, streaming, chunking)
- [x] Real-time latency improvements
- [x] Documentation (project plan updated)

**Implemented Features:**
- `core/controller.py` - Comprehensive timing instrumentation at each pipeline stage
- `ai/llm.py` - HTTP connection pooling, DNS bypass, keep-alive system, history limiting
- Pipeline timing breakdown: PTT → STT → LLM First Token → TTS First Audio → End
- Real-time performance monitoring for debugging and optimization tracking

**Performance Achievements:**
- **LLM First Token:** 2.3s → 0.29s (8x improvement)
- **Time to First Audio:** 2.9s → 0.96s (3x improvement)
- **STT Processing:** 0.4s (already optimized)
- **End-to-end:** 6s average (TTS playback duration dominates remaining time)

---

## Performance Considerations

### Memory Budget
- Target: <500MB RAM total
- Whisper base model: ~150MB
- Piper voice model: ~60MB
- LLM: Handled by Ollama (separate process)
- UI/Avatar: ~100MB

### Latency Targets (ACHIEVED)
- STT processing: <0.4 seconds for typical utterance
- LLM first token: <0.3 seconds (ultra-low latency optimization)
- TTS first audio: <0.5s (sentence-level streaming)
- Time to first speech: <1.0s end-to-end from PTT release

### Performance Optimizations (Latest)
- **Ultra-Low Latency Pipeline**: 8x faster LLM response (2.3s → 0.29s first token)
- **HTTP Connection Pooling**: `requests.Session()` eliminates TCP handshake overhead
- **DNS Bypass**: `127.0.0.1` instead of `localhost` avoids Windows DNS delay
- **Model Keep-Alive**: Background thread prevents model unloading between requests
- **Conversation History Limiting**: Last 4 messages only to reduce prompt processing time
- **Comprehensive Timing Instrumentation**: Detailed per-stage latency tracking for debugging
- **TTS Audio Buffer**: Fixed O(n²) performance issue - now O(1) per callback
- **Continuous Streaming**: Single audio stream for multiple sentences (no gaps)
- **Sentence Chunking**: Aggressive thresholds (8-40 chars) for immediate speech start
- **LLM Pre-warming**: Model loaded on startup and model switch to eliminate cold start
- **Queue-based Audio**: Replaced list concatenation with efficient queue system

### CPU Considerations
- Whisper.cpp: Uses multiple threads, brief spike during transcription
- Piper: Lightweight, minimal CPU
- Avatar rendering: 144 FPS with 2x supersampling, optimized pygame rendering
- Idle: <5% CPU when not processing

---

## Error Handling

### Graceful Degradation
- If Ollama not running: Show error, prompt to start
- If no models found: Direct to settings, show install instructions
- If microphone unavailable: Disable PTT, show warning
- If Brave API fails: Continue without search, notify user

### User Feedback
- Avatar visual state for errors (red tint? shake animation?)
- Toast notifications for non-critical errors
- Settings panel shows connection status for services

---

## Future Enhancements (Post-V1)

- Multiple personality presets
- Conversation history (opt-in)
- Screenshot analysis (send screen region to vision model)
- Custom avatar skins/themes
- Audio wake word (always listening mode)
- Keyboard-only interaction mode
- Plugin system for extending capabilities
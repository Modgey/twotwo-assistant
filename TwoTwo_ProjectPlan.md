# Project TwoTwo: Local AI Assistant

## Overview

TwoTwo is a local AI assistant inspired by Jarvis from Iron Man. It features a futuristic holographic-style interface that floats on top of other windows, with real-time voice interaction and an animated avatar with an expressive "eye."

**Core Philosophy:** Everything runs locally for privacy and low latency. No cloud dependencies except optional web search.

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
│   ├── renderer.py             # Pygame avatar renderer
│   ├── eye.py                  # Eye/pupil animation logic
│   ├── aperture.py             # Aperture ring animation logic
│   └── expressions.py          # Expression state definitions
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

**Files:** `avatar/renderer.py`, `avatar/eye.py`, `avatar/aperture.py`

**Visual Design:**
- Aperture-lens sphere shape (overlapping ring segments forming a circular opening)
- Central "eye" (pupil) that can move and change shape
- Neutral white/light gray color palette
- Slight transparency (80-90% opacity)
- Soft glow effect around edges (optional)

**Avatar States & Animations:**

| State | Aperture Behavior | Eye Behavior | Trigger |
|-------|-------------------|--------------|---------|
| Idle | Slow, subtle breathing pulse | Gentle random look-around, occasional blink | Default state |
| Listening | Aperture opens slightly wider | Pupil focused center, slightly dilated | Push-to-talk active |
| Thinking | Rings rotate slowly, pulsing | Pupil contracts, looks up-right | Waiting for LLM response |
| Speaking | Subtle pulse synced to audio amplitude | Pupil animates with speech rhythm | TTS playing |

**Pygame Embedding in PySide6:**
```python
# Embed pygame surface in QWidget
class AvatarWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.pygame_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        
    def paintEvent(self, event):
        # Render pygame surface to QImage, then to widget
        # Use QImage.Format_ARGB32 for transparency
```

**Avatar Size:** ~200x200 pixels (configurable)

---

### 3. Voice Engine

#### 3.1 Speech-to-Text (STT)

**File:** `voice/stt.py`

**Technology:** whisper.cpp via `pywhispercpp` or subprocess

**Model:** `whisper-base` or `whisper-small` (user configurable)
- `base`: ~74MB, faster, slightly less accurate
- `small`: ~244MB, slower, more accurate

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
1. **Ollama** (preferred) - Easy model management, REST API
2. **llama.cpp** (fallback) - Direct integration if Ollama unavailable

**Ollama Integration:**
```python
import requests

def chat(messages, model="llama3.2:3b"):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": model, "messages": messages, "stream": True},
        stream=True
    )
    # Yield tokens as they arrive for streaming display
```

**Target Models:** 4B-8B parameter models for good performance on consumer hardware
- `llama3.2:3b`
- `llama3.2:8b` 
- `phi3:3b`
- `mistral:7b`

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
- Avatar position (x, y) - or "Reset to default"
- Avatar size (small / medium / large)
- Opacity slider (50% - 100%)

#### Voice
- Push-to-talk hotkey (key capture)
- STT model selection (base / small)
- TTS voice selection (dropdown of available Piper voices)
- TTS speed slider
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
- Show streaming text response near avatar
- Fade in/out animation
- Auto-dismiss after TTS completes (configurable delay)
- Semi-transparent background
- Positioned relative to avatar (above or to the side)

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

# Voice
pywhispercpp>=1.0.0
piper-tts>=1.0.0
sounddevice>=0.4.6
numpy>=1.24.0

# AI
requests>=2.31.0

# System
pynput>=1.7.6
```

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

### Phase 1: Foundation
- [ ] Project setup, directory structure
- [ ] Config management system
- [ ] Basic PySide6 transparent overlay window
- [ ] System tray icon with quit option

### Phase 2: Avatar
- [ ] Pygame embedding in PySide6
- [ ] Basic aperture ring rendering
- [ ] Eye/pupil rendering
- [ ] Idle animation loop
- [ ] All avatar states (listening, thinking, speaking)
- [ ] Alt+drag movement

### Phase 3: Voice
- [ ] Audio capture with sounddevice
- [ ] Whisper.cpp STT integration
- [ ] Piper TTS integration
- [ ] Push-to-talk hotkey
- [ ] Audio amplitude extraction for avatar sync

### Phase 4: AI
- [ ] Ollama integration with streaming
- [ ] Model auto-detection
- [ ] Brave Search integration
- [ ] Autonomous search decision logic

### Phase 5: UI Polish
- [ ] Settings panel (all sections)
- [ ] Response text display with streaming
- [ ] Smooth state transitions
- [ ] Error handling and user feedback

### Phase 6: Refinement
- [ ] Performance optimization
- [ ] Edge case handling
- [ ] User testing and feedback
- [ ] Documentation

---

## Performance Considerations

### Memory Budget
- Target: <500MB RAM total
- Whisper base model: ~150MB
- Piper voice model: ~60MB
- LLM: Handled by Ollama (separate process)
- UI/Avatar: ~100MB

### Latency Targets
- STT processing: <2 seconds for typical utterance
- LLM first token: <1 second
- TTS first audio: <500ms

### CPU Considerations
- Whisper.cpp: Uses multiple threads, brief spike during transcription
- Piper: Lightweight, minimal CPU
- Avatar rendering: Target 30 FPS, use dirty rect updates
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
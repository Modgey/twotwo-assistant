
import queue
import threading
import time
import numpy as np
from unittest.mock import MagicMock

# Mock dependencies
import sys
# sys.modules['voice'] = MagicMock() # DO NOT MOCK PACKAGE WE ARE IMPORTING FROM
sys.modules['voice.effects'] = MagicMock()
sys.modules['config'] = MagicMock()
sys.modules['sounddevice'] = MagicMock()

# Import the class to test (we need to be careful with imports)
# We'll rely on the fact that we can import TTSQueue if we mock its deps properly
# But TTSQueue imports PiperTTS which imports config...
# Let's just copy the relevant queue logic to test it, OR just import it if possible.

# It's better to import the real class.
# We need to make sure 'config' is importable.
from typing import Any

# Mock get_config
def get_config():
    return MagicMock()

sys.modules['config'].get_config = get_config

# Now we can try traversing the path to import. 
# Add current directory to path so we can import modules
import os
sys.path.append(os.getcwd())

try:
    from voice.tts import TTSQueue
except ImportError as e:
    print(f"Import failed: {e}")
    try:
        # Try adjusting path for parent
        sys.path.append(os.path.dirname(os.getcwd()))
        from twotwo.voice.tts import TTSQueue
    except ImportError as e2:
         print(f"Second import failed: {e2}")
         raise

def test_tts_queue_sync():
    print("Testing TTSQueue synchronization...")
    
    # Mock TTS
    mock_tts = MagicMock()
    mock_tts.sample_rate = 22050
    # Mock synthesize_streaming to return immediate chunks
    def mock_synthesize(text, chunk_callback, on_complete):
        # Simulate synthesis delay
        time.sleep(0.1)
        # Return 2 chunks
        chunk_callback(np.zeros(100, dtype=np.float32))
        chunk_callback(np.zeros(100, dtype=np.float32))
        if on_complete:
            on_complete()
        return MagicMock()
    
    mock_tts.synthesize_streaming.side_effect = mock_synthesize
    
    # Capture events
    events = []
    
    def on_sentence_start(text):
        events.append(f"START: {text}")
        print(f"Callback: Start '{text}'")
        
    def chunk_consumer(tts_queue):
        # Simulate the audio callback consuming chunks
        print("Consumer started")
        while True:
            try:
                # We need to access the private queue to simulate the audio callback's view
                item = tts_queue._audio_chunks.get(timeout=2.0)
                
                if isinstance(item, dict) and item.get("type") == "marker":
                    if tts_queue.on_sentence_start:
                        tts_queue.on_sentence_start(item["text"])
                else:
                    events.append("AUDIO_CHUNK")
                    # print("Processed audio chunk")
            except queue.Empty:
                break
                
    # Initialize Queue
    tts_queue = TTSQueue(
        tts=mock_tts,
        on_sentence_start=on_sentence_start,
        streaming=True
    )
    
    # Override _ensure_stream_running to do nothing (we consume manually)
    tts_queue._ensure_stream_running = lambda: None
    
    # Start the processor
    tts_queue.start()
    
    # Speak two sentences
    tts_queue.speak("Sentence One")
    tts_queue.speak("Sentence Two")
    
    # Run consumer in parallel (simulating audio thread)
    consumer_thread = threading.Thread(target=chunk_consumer, args=(tts_queue,))
    consumer_thread.start()
    
    # Wait
    time.sleep(1.0)
    tts_queue.stop()
    consumer_thread.join()
    
    print("\nEvents recorded:")
    for e in events:
        print(e)
        
    # Verify order
    expected_order = [
        "START: Sentence One",
        "AUDIO_CHUNK",
        "AUDIO_CHUNK", 
        "START: Sentence Two",
        "AUDIO_CHUNK",
        "AUDIO_CHUNK"
    ]
    
    # We might have more audio chunks depending on how we mocked it, but the order of START vs AUDIO is key
    
    # Check that we have at least these markers in this relative order
    sentence_one_idx = -1
    sentence_two_idx = -1
    
    for i, e in enumerate(events):
        if e == "START: Sentence One":
            sentence_one_idx = i
        elif e == "START: Sentence Two":
            sentence_two_idx = i
            
    if sentence_one_idx != -1 and sentence_two_idx != -1:
        if sentence_one_idx < sentence_two_idx:
            # Check for chunks in between
            chunks_between = 0
            for i in range(sentence_one_idx + 1, sentence_two_idx):
                if events[i] == "AUDIO_CHUNK":
                    chunks_between += 1
            
            if chunks_between > 0:
                print("\nSUCCESS: Markers appeared in correct order with audio chunks in between.")
            else:
                print("\nFAILURE: No audio chunks between sentences.")
        else:
            print("\nFAILURE: Sentence Two started before Sentence One.")
    else:
        print("\nFAILURE: Missing sentence markers.")

if __name__ == "__main__":
    test_tts_queue_sync()

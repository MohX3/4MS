# utils/kokoro_tts_local.py - Kokoro TTS (Open-source, High-quality)

import os
import hashlib
from pathlib import Path
import tempfile
import subprocess
import json

try:
    from kokoro_onnx import Kokoro
    import numpy as np
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    print("Kokoro-TTS not installed. Install with: pip install kokoro-tts")

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    if KOKORO_AVAILABLE:
        print("Warning: pydub not available. WAV to MP3 conversion will be skipped.")

# Cache directory
KOKORO_CACHE_DIR = Path("./cache/tts_kokoro")
KOKORO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Voice options - Kokoro TTS voices (50 available!)
VOICE_OPTIONS = {
    # American Female Voices (Most popular for interviews)
    "af_sarah": "af_sarah",           # Female - Clear, Professional ⭐
    "af_bella": "af_bella",           # Female - Warm, Friendly
    "af_nicole": "af_nicole",         # Female - Natural, Smooth
    "af_sky": "af_sky",               # Female - Bright, Energetic
    "af_nova": "af_nova",             # Female - Modern, Dynamic
    "af_jessica": "af_jessica",       # Female - Confident, Articulate
    "af_alloy": "af_alloy",           # Female - Tech-savvy, Professional
    "af_river": "af_river",           # Female - Calm, Flowing
    "af_heart": "af_heart",           # Female - Warm, Empathetic
    
    # American Male Voices
    "am_adam": "am_adam",             # Male - Deep, Confident ⭐ DEFAULT
    "am_michael": "am_michael",       # Male - Professional, Clear
    "am_eric": "am_eric",             # Male - Friendly, Approachable
    "am_liam": "am_liam",             # Male - Charismatic, Engaging
    "am_echo": "am_echo",             # Male - Resonant, Powerful
    "am_onyx": "am_onyx",             # Male - Strong, Authoritative
    "am_puck": "am_puck",             # Male - Playful, Energetic
    
    # British Female Voices
    "bf_emma": "bf_emma",             # Female - British, Elegant
    "bf_isabella": "bf_isabella",     # Female - British, Professional
    "bf_alice": "bf_alice",           # Female - British, Refined
    "bf_lily": "bf_lily",             # Female - British, Gentle
    
    # British Male Voices  
    "bm_george": "bm_george",         # Male - British, Authoritative
    "bm_lewis": "bm_lewis",           # Male - British, Friendly
    "bm_daniel": "bm_daniel",         # Male - British, Distinguished
    "bm_fable": "bm_fable",           # Male - British, Storytelling
}

# Default voice for interviews
DEFAULT_VOICE = VOICE_OPTIONS["bm_daniel"]  # Male - Deep, Confident


def get_text_hash(text, voice, speed):
    """Generate hash for caching"""
    return hashlib.md5(f"{text}_{voice}_{speed}".encode('utf-8')).hexdigest()


# Global Kokoro instance (lazy loaded)
_kokoro_instance = None
_kokoro_lock = None

def _get_kokoro_instance():
    """Get or create Kokoro TTS instance"""
    global _kokoro_instance, _kokoro_lock
    
    if _kokoro_lock is None:
        import threading
        _kokoro_lock = threading.Lock()
    
    with _kokoro_lock:
        if _kokoro_instance is None:
            # Initialize Kokoro with model files
            model_path = Path("./kokoro-v1.0.onnx")
            voices_path = Path("./voices-v1.0.bin")
            
            if not model_path.exists() or not voices_path.exists():
                raise FileNotFoundError(
                    "Kokoro model files not found. Please download:\n"
                    "  kokoro-v1.0.onnx\n"
                    "  voices-v1.0.bin\n"
                    "from https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/"
                )
            
            _kokoro_instance = Kokoro(str(model_path), str(voices_path))
    
    return _kokoro_instance


def kokoro_tts_generate(
    text,
    voice=DEFAULT_VOICE,
    speed=0.90,
    lang="en-us",
    use_cache=True
):
    """
    Generate speech using Kokoro TTS (Open-source, High-quality)
    
    Args:
        text (str): Text to convert to speech
        voice (str): Voice to use (see VOICE_OPTIONS)
        speed (float): Speech speed (0.5 = slow, 1.0 = normal, 1.5 = fast)
        lang (str): Language code (e.g., "en-us")
        use_cache (bool): Whether to use caching
    
    Returns:
        tuple: (audio_path, error_message) or (None, error_message) if failed
    """
    if not KOKORO_AVAILABLE:
        return None, "Kokoro-TTS not installed. Install with: pip install kokoro-tts"
    
    if not text or text.strip() == "":
        return None, "Empty text input"
    
    try:
        # Check cache first (look for MP3, the final output format)
        if use_cache:
            text_hash = get_text_hash(text, voice, speed)
            cache_file_mp3 = KOKORO_CACHE_DIR / f"{text_hash}.mp3"
            cache_file_wav = KOKORO_CACHE_DIR / f"{text_hash}.wav"
            
            if cache_file_mp3.exists():
                return str(cache_file_mp3), None
        else:
            cache_file_wav = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
            cache_file_mp3 = Path(tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name)
        
        # Generate speech using Kokoro CLI (reliable and fast with caching)
        # Create temp input file
        temp_input = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
        temp_input.write(text)
        temp_input.close()
        
        try:
            command = [
                "python", "-m", "kokoro_tts",
                temp_input.name,
                str(cache_file_wav),
                "--voice", voice,
                "--lang", lang,
                "--speed", str(speed),
                "--format", "wav"
            ]
            
            # Suppress CLI output by redirecting stderr to devnull
            # Use DEVNULL for stdin/stdout/stderr to make it faster
            result = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=30,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0  # Hide window on Windows
            )
            
            if result.returncode != 0:
                error_msg = result.stderr.decode('utf-8', errors='ignore') if result.stderr else "Unknown error"
                return None, f"Kokoro-TTS error: {error_msg}"
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_input.name)
            except:
                pass
        
        # Convert WAV to MP3 (app expects MP3 format)
        if not cache_file_wav.exists():
            return None, "WAV file generation failed"
        
        try:
            if PYDUB_AVAILABLE:
                audio = AudioSegment.from_wav(str(cache_file_wav))
                audio.export(str(cache_file_mp3), format="mp3", bitrate="128k")
                
                # Clean up WAV file if not caching
                if not use_cache:
                    try:
                        os.unlink(str(cache_file_wav))
                    except:
                        pass
            else:
                # If pydub not available, return WAV and hope for the best
                print("[WARNING] pydub not available, returning WAV instead of MP3")
                return str(cache_file_wav), None
        except Exception as conv_error:
            print(f"[WARNING] MP3 conversion failed: {conv_error}, returning WAV")
            return str(cache_file_wav), None
        
        return str(cache_file_mp3), None
        
    except Exception as e:
        return None, f"Kokoro-TTS error: {str(e)}"


def kokoro_tts_fast(text, use_cache=True):
    """
    Quick wrapper with optimized defaults for interview responses
    
    Args:
        text (str): Text to convert to speech
        use_cache (bool): Whether to use caching
    
    Returns:
        tuple: (audio_path, error_message)
    """
    return kokoro_tts_generate(
        text=text,
        voice=DEFAULT_VOICE,
        speed=1.1,  # Slightly faster for dynamic conversation
        lang="en-us",
        use_cache=use_cache
    )


def list_available_voices():
    """
    List all available Kokoro-TTS voices
    
    Returns:
        list: Available voice names
    """
    if not KOKORO_AVAILABLE:
        return []
    
    try:
        # Try to get voices from CLI
        result = subprocess.run(
            ["kokoro-tts", "--help-voices"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            # Parse output to get voice names
            voices = []
            for line in result.stdout.split('\n'):
                if line.strip() and not line.startswith('-'):
                    voices.append(line.strip())
            return voices
    except:
        pass
    
    # Return predefined voices if CLI fails
    return list(VOICE_OPTIONS.values())


# Preload common interview phrases for instant responses
def preload_common_phrases(voice=DEFAULT_VOICE):
    """Preload common phrases for instant playback"""
    if not KOKORO_AVAILABLE:
        return
    
    common_phrases = [
        "Hello! I'm your AI interviewer today. Let's begin with a brief introduction.",
        "That's interesting. Can you tell me more about that?",
        "Thank you for sharing that with me.",
        "Could you elaborate on that point?",
        "That's a great answer. Let's move on to the next question.",
        "Thank you. That's it for today.",
    ]
    
    print(f"Preloading common phrases with {voice}...")
    success = 0
    for phrase in common_phrases:
        try:
            result, error = kokoro_tts_generate(phrase, voice=voice, use_cache=True)
            if result:
                success += 1
        except:
            pass
    print(f"Preloaded {success}/{len(common_phrases)} phrases!")


# Test function
def test_kokoro_tts():
    """Test Kokoro-TTS installation and functionality"""
    print("Testing Kokoro-TTS...")
    
    if not KOKORO_AVAILABLE:
        print("[X] Kokoro-TTS not installed")
        print("Install with: pip install kokoro-tts")
        return False
    
    print("[OK] Kokoro-TTS is installed")
    
    test_text = "Hello! This is a test of Kokoro Text to Speech."
    print(f"Generating test audio: '{test_text}'")
    
    audio_path, error = kokoro_tts_fast(test_text)
    
    if error:
        print(f"[ERROR] {error}")
        return False
    
    print(f"[SUCCESS] Audio generated: {audio_path}")
    if os.path.exists(audio_path):
        file_size = os.path.getsize(audio_path) / 1024
        file_ext = os.path.splitext(audio_path)[1]
        print(f"File format: {file_ext}")
        print(f"File size: {file_size:.2f} KB")
        
        if file_ext == ".mp3":
            print("[OK] MP3 format (compatible with app)")
        else:
            print(f"[WARNING] {file_ext} format - app expects MP3")
    else:
        print("[WARNING] Audio file not found")
    
    return True


if __name__ == "__main__":
    # Run test when module is executed directly
    test_kokoro_tts()


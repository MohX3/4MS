# utils/realtime_stt.py - AssemblyAI Real-Time Streaming STT

import os
import asyncio
import websockets
import json
import base64
import threading
import queue
from typing import Optional, Callable

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
REALTIME_STT_AVAILABLE = bool(ASSEMBLYAI_API_KEY)

class RealtimeTranscriber:
    """
    AssemblyAI Real-Time Streaming Transcription
    Provides instant transcription as user speaks
    """
    
    def __init__(self, api_key: str = None, sample_rate: int = 16000):
        """
        Initialize real-time transcriber
        
        Args:
            api_key: AssemblyAI API key
            sample_rate: Audio sample rate (default 16000 Hz)
        """
        self.api_key = api_key or ASSEMBLYAI_API_KEY
        self.sample_rate = sample_rate
        self.ws = None
        self.session_begins = False
        self.final_transcript = ""
        self.partial_transcript = ""
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.on_transcript_callback = None
        self.on_final_callback = None
        
    async def connect(self):
        """Establish WebSocket connection to AssemblyAI"""
        url = f"wss://api.assemblyai.com/v2/realtime/ws?sample_rate={self.sample_rate}"
        
        # Create custom headers for authentication
        extra_headers = [
            ("Authorization", self.api_key)
        ]
        
        self.ws = await websockets.connect(
            url, 
            additional_headers=extra_headers,
            ping_interval=5, 
            ping_timeout=20
        )
        
    async def send_audio(self, audio_data: bytes):
        """Send audio data to AssemblyAI"""
        if self.ws and not self.ws.closed:
            # Convert audio to base64
            audio_b64 = base64.b64encode(audio_data).decode()
            
            # Send as JSON
            message = json.dumps({"audio_data": audio_b64})
            await self.ws.send(message)
    
    async def receive_transcripts(self):
        """Receive and process transcription results"""
        async for message in self.ws:
            data = json.loads(message)
            
            if data.get("message_type") == "SessionBegins":
                self.session_begins = True
                print("Real-time session started")
                
            elif data.get("message_type") == "PartialTranscript":
                # Partial (interim) transcription
                self.partial_transcript = data.get("text", "")
                if self.on_transcript_callback:
                    self.on_transcript_callback(self.partial_transcript, is_final=False)
                    
            elif data.get("message_type") == "FinalTranscript":
                # Final transcription for this segment
                final_text = data.get("text", "")
                self.final_transcript += " " + final_text
                if self.on_final_callback:
                    self.on_final_callback(final_text.strip())
    
    async def close(self):
        """Close the WebSocket connection"""
        if self.ws:
            # Send terminate message
            await self.ws.send(json.dumps({"terminate_session": True}))
            await self.ws.close()
            self.ws = None
    
    def set_callbacks(self, on_transcript: Callable = None, on_final: Callable = None):
        """
        Set callback functions for transcription events
        
        Args:
            on_transcript: Called for each partial transcript (text, is_final)
            on_final: Called for each final transcript (text)
        """
        self.on_transcript_callback = on_transcript
        self.on_final_callback = on_final


# Synchronous wrapper for Streamlit
class StreamingSTT:
    """Simplified streaming STT for Streamlit"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or ASSEMBLYAI_API_KEY
        self.transcriber = None
        self.loop = None
        self.thread = None
        self.transcripts = []
        self.is_active = False
        
    def start_session(self, on_transcript=None, on_final=None):
        """Start a new streaming session"""
        self.transcriber = RealtimeTranscriber(self.api_key)
        self.transcriber.set_callbacks(on_transcript, on_final)
        self.transcripts = []
        self.is_active = True
        
        # Start async event loop in background thread
        self.thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.thread.start()
        
    def _run_async_loop(self):
        """Run async event loop in background thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._async_session())
        
    async def _async_session(self):
        """Async session management"""
        await self.transcriber.connect()
        
        # Start receiving in background
        receive_task = asyncio.create_task(self.transcriber.receive_transcripts())
        
        # Keep session alive
        while self.is_active:
            await asyncio.sleep(0.1)
        
        # Close connection
        receive_task.cancel()
        await self.transcriber.close()
    
    def send_audio(self, audio_bytes: bytes):
        """Send audio data for transcription"""
        if self.loop and self.is_active:
            asyncio.run_coroutine_threadsafe(
                self.transcriber.send_audio(audio_bytes),
                self.loop
            )
    
    def stop_session(self):
        """Stop the streaming session"""
        self.is_active = False
        if self.thread:
            self.thread.join(timeout=2)
        return self.transcriber.final_transcript.strip() if self.transcriber else ""


# Simple function for batch audio (like current implementation)
def transcribe_audio_realtime(audio_file_path: str, timeout: int = 30) -> tuple:
    """
    Transcribe a complete audio file using AssemblyAI Real-Time API
    (Faster than batch API but requires streaming setup)
    
    Args:
        audio_file_path: Path to audio file
        timeout: Maximum wait time in seconds
    
    Returns:
        tuple: (transcribed_text, error_message)
    """
    import wave
    
    try:
        # Open audio file
        with wave.open(audio_file_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        
        # Create transcriber
        final_transcript = []
        error_occurred = None
        
        def on_final(text):
            final_transcript.append(text)
        
        async def process_audio():
            nonlocal error_occurred
            try:
                transcriber = RealtimeTranscriber(sample_rate=sample_rate)
                transcriber.on_final_callback = on_final
                
                await transcriber.connect()
                
                # Start receiving task
                receive_task = asyncio.create_task(transcriber.receive_transcripts())
                
                # Wait for session to begin
                while not transcriber.session_begins:
                    await asyncio.sleep(0.1)
                
                # Send audio in chunks (simulate real-time)
                chunk_size = 8192  # 8KB chunks
                for i in range(0, len(frames), chunk_size):
                    chunk = frames[i:i + chunk_size]
                    await transcriber.send_audio(chunk)
                    await asyncio.sleep(0.01)  # Small delay to simulate streaming
                
                # Wait a bit for final transcripts
                await asyncio.sleep(2)
                
                # Close connection
                receive_task.cancel()
                await transcriber.close()
                
            except Exception as e:
                error_occurred = str(e)
        
        # Run async processing
        asyncio.run(process_audio())
        
        if error_occurred:
            return None, f"Streaming error: {error_occurred}"
        
        if final_transcript:
            return " ".join(final_transcript), None
        else:
            return None, "No transcription received"
            
    except Exception as e:
        return None, f"Audio processing error: {str(e)}"


# Faster version using AssemblyAI streaming
def transcribe_audio_file_streaming(audio_file_path: str, max_wait: int = 30) -> tuple:
    """
    Fast transcription using AssemblyAI Real-Time Streaming
    
    Args:
        audio_file_path: Path to audio file
        max_wait: Maximum wait time
    
    Returns:
        tuple: (text, error) or (None, error_message)
    """
    try:
        # Check if file exists and is valid
        if not os.path.exists(audio_file_path):
            return None, "Audio file not found"
        
        file_size = os.path.getsize(audio_file_path)
        if file_size == 0:
            return None, "Audio file is empty"
        
        # Use real-time streaming for faster results
        text, error = transcribe_audio_realtime(audio_file_path, timeout=max_wait)
        
        return text, error
        
    except Exception as e:
        return None, f"Transcription failed: {str(e)}"


# Test function
def test_realtime_stt():
    """Test real-time streaming STT"""
    print("Testing AssemblyAI Real-Time Streaming...")
    
    if not ASSEMBLYAI_API_KEY:
        print("[ERROR] ASSEMBLYAI_API_KEY not found in environment")
        return False
    
    print("[OK] API key found")
    print("Real-time streaming transcription is ready!")
    print("\nNote: This requires audio input to test fully.")
    print("Use transcribe_audio_file_streaming() with a WAV file.")
    
    return True


if __name__ == "__main__":
    test_realtime_stt()


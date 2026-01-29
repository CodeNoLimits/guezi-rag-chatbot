"""
GUEZI Voice Conversation Module
Real-time voice conversations using Gemini Live API
Supports barge-in (interruption) for natural conversation flow
"""

import os
import asyncio
import base64
from typing import Optional, Callable, AsyncGenerator
from dotenv import load_dotenv

load_dotenv("config/.env")


class VoiceConversation:
    """
    Real-time voice conversation using Gemini Live API.

    Features:
    - Bidirectional audio streaming
    - Voice Activity Detection (VAD)
    - Barge-in support (interrupt the AI while speaking)
    - Multiple voice options
    - Transcription of both user and AI speech
    """

    MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"

    SYSTEM_PROMPT = """You are GUEZI (גואזי), a warm and knowledgeable AI assistant
    specializing in Rabbi Nachman of Breslov's teachings.

    You speak naturally and conversationally. When discussing teachings:
    - Reference specific sources (Likutei Moharan, Sippurei Maasiyot, etc.)
    - Be encouraging and uplifting
    - Remember: "There is no despair in the world at all!"

    Keep responses concise for voice conversation - about 2-3 sentences typically.
    If asked complex questions, you can elaborate but remain conversational."""

    VOICES = {
        'kore': 'Kore',      # Female, clear
        'puck': 'Puck',      # Male, friendly
        'charon': 'Charon',  # Male, deep
        'aoede': 'Aoede',    # Female, warm
        'fenrir': 'Fenrir',  # Male, strong
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice: str = "Kore",
        language: str = "en-US"
    ):
        from google import genai
        from google.genai import types

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY required")

        self.client = genai.Client(api_key=self.api_key)
        self.types = types
        self.voice = voice
        self.language = language

        # Callbacks
        self.on_transcription: Optional[Callable[[str, str], None]] = None
        self.on_audio: Optional[Callable[[bytes], None]] = None
        self.on_interrupted: Optional[Callable[[], None]] = None

        # State
        self.is_connected = False
        self.session = None

    def _get_config(self):
        """Get Live API configuration"""
        return self.types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=self.SYSTEM_PROMPT,

            # Voice configuration
            speech_config=self.types.SpeechConfig(
                voice_config=self.types.VoiceConfig(
                    prebuilt_voice_config=self.types.PrebuiltVoiceConfig(
                        voice_name=self.voice
                    )
                ),
                language_code=self.language
            ),

            # VAD configuration for natural conversation
            realtime_input_config=self.types.RealtimeInputConfig(
                automatic_activity_detection=self.types.AutomaticActivityDetection(
                    start_of_speech_sensitivity=self.types.StartOfSpeechSensitivity.MEDIUM,
                    end_of_speech_sensitivity=self.types.EndOfSpeechSensitivity.MEDIUM,
                    prefix_padding_ms=300,
                    silence_duration_ms=700,
                ),
                activity_handling=self.types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
            ),

            # Enable transcriptions
            input_audio_transcription=self.types.AudioTranscriptionConfig(),
            output_audio_transcription=self.types.AudioTranscriptionConfig(),
        )

    async def connect(self):
        """Connect to Gemini Live API"""
        try:
            self.session = await self.client.aio.live.connect(
                model=self.MODEL,
                config=self._get_config()
            )
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    async def disconnect(self):
        """Disconnect from the API"""
        if self.session:
            try:
                await self.session.close()
            except:
                pass
        self.is_connected = False
        self.session = None

    async def send_audio(self, audio_data: bytes):
        """
        Send audio data to the API.

        Args:
            audio_data: Raw 16-bit PCM audio at 16kHz, mono
        """
        if not self.session:
            return

        try:
            await self.session.send_realtime_input(
                audio=self.types.Blob(
                    data=audio_data,
                    mime_type='audio/pcm;rate=16000'
                )
            )
        except Exception as e:
            print(f"Send audio error: {e}")

    async def receive_responses(self) -> AsyncGenerator[dict, None]:
        """
        Receive responses from the API.

        Yields:
            dict with keys:
            - type: 'transcription' | 'audio' | 'interrupted'
            - role: 'user' | 'assistant' (for transcription)
            - text: transcription text
            - audio: audio bytes (for audio type)
        """
        if not self.session:
            return

        try:
            async for response in self.session.receive():
                server_content = response.server_content

                # Handle interruption (barge-in)
                if server_content.interrupted:
                    yield {'type': 'interrupted'}
                    if self.on_interrupted:
                        self.on_interrupted()
                    continue

                # Handle user transcription
                if server_content.input_transcription:
                    text = server_content.input_transcription.text
                    yield {
                        'type': 'transcription',
                        'role': 'user',
                        'text': text
                    }
                    if self.on_transcription:
                        self.on_transcription('user', text)

                # Handle AI transcription
                if server_content.output_transcription:
                    text = server_content.output_transcription.text
                    yield {
                        'type': 'transcription',
                        'role': 'assistant',
                        'text': text
                    }
                    if self.on_transcription:
                        self.on_transcription('assistant', text)

                # Handle audio output
                if server_content.model_turn:
                    for part in server_content.model_turn.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            audio_data = part.inline_data.data
                            yield {
                                'type': 'audio',
                                'audio': audio_data
                            }
                            if self.on_audio:
                                self.on_audio(audio_data)

        except Exception as e:
            print(f"Receive error: {e}")

    async def send_text(self, text: str):
        """Send text message instead of audio"""
        if not self.session:
            return

        try:
            await self.session.send_client_content(
                turns=[self.types.Content(
                    role="user",
                    parts=[self.types.Part(text=text)]
                )]
            )
        except Exception as e:
            print(f"Send text error: {e}")


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000) -> bytes:
    """Convert raw PCM to WAV format"""
    import struct
    import io

    channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(pcm_data)

    wav_buffer = io.BytesIO()
    wav_buffer.write(b'RIFF')
    wav_buffer.write(struct.pack('<I', 36 + data_size))
    wav_buffer.write(b'WAVE')
    wav_buffer.write(b'fmt ')
    wav_buffer.write(struct.pack('<I', 16))
    wav_buffer.write(struct.pack('<H', 1))
    wav_buffer.write(struct.pack('<H', channels))
    wav_buffer.write(struct.pack('<I', sample_rate))
    wav_buffer.write(struct.pack('<I', byte_rate))
    wav_buffer.write(struct.pack('<H', block_align))
    wav_buffer.write(struct.pack('<H', bits_per_sample))
    wav_buffer.write(b'data')
    wav_buffer.write(struct.pack('<I', data_size))
    wav_buffer.write(pcm_data)

    return wav_buffer.getvalue()


# Simple test
if __name__ == "__main__":
    import sys

    async def test():
        print("Testing Gemini Live API connection...")

        conv = VoiceConversation()

        # Try to connect
        if await conv.connect():
            print("✅ Connected to Gemini Live API")

            # Send a text message
            await conv.send_text("Hello, can you introduce yourself briefly?")

            # Receive response
            audio_chunks = []
            async for response in conv.receive_responses():
                if response['type'] == 'transcription':
                    print(f"[{response['role']}]: {response['text']}")
                elif response['type'] == 'audio':
                    audio_chunks.append(response['audio'])
                elif response['type'] == 'interrupted':
                    print("[Interrupted]")

                # Exit after getting response
                if response['type'] == 'transcription' and response['role'] == 'assistant':
                    break

            if audio_chunks:
                print(f"✅ Received {len(audio_chunks)} audio chunks")

            await conv.disconnect()
            print("✅ Disconnected")
        else:
            print("❌ Failed to connect")

    asyncio.run(test())

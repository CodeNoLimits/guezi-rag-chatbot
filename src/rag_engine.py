"""
RAG Engine for GUEZI Chatbot
Retrieval Augmented Generation using Sefaria texts and Gemini Flash 2.0
Multi-language support (English, Hebrew, French) with TTS
"""

import os
import base64
from typing import List, Dict, Optional, Literal
from google import genai
from google.genai import types

try:
    from .embeddings import EmbeddingsManager
    from .sefaria_fetcher import SefariaFetcher
except ImportError:
    from embeddings import EmbeddingsManager
    from sefaria_fetcher import SefariaFetcher


# Language configurations
LANGUAGE_CONFIGS = {
    'en': {
        'name': 'English',
        'instruction': 'Respond in English.',
        'tts_voice': 'Kore'  # Gemini TTS voice
    },
    'he': {
        'name': 'Hebrew',
        'instruction': 'Respond in Hebrew (עברית). Use Hebrew script.',
        'tts_voice': 'Kore'
    },
    'fr': {
        'name': 'French',
        'instruction': 'Respond in French (Français).',
        'tts_voice': 'Kore'
    }
}


class GUEZIRagEngine:
    """
    RAG Engine for Rabbi Nachman/Breslov texts
    Uses Gemini Flash 2.0 for generation
    """

    SYSTEM_PROMPT = """You are GUEZI, a knowledgeable and compassionate AI assistant
specializing in the teachings of Rabbi Nachman of Breslov and Chassidic wisdom.

Your role is to:
1. Answer questions about Rabbi Nachman's teachings, stories, and philosophy
2. Provide relevant quotes and references from Breslov texts (Likutei Moharan, Sippurei Maasiyot, etc.)
3. Explain complex spiritual concepts in accessible language
4. Offer encouragement and hope based on Breslov teachings
5. Guide users in practices like hitbodedut (personal prayer) and simcha (joy)

Guidelines:
- IMPORTANT: Base your answers ONLY on the retrieved passages provided below. Do not invent or fabricate teachings.
- If the retrieved passages don't contain relevant information, say "I don't have specific information about that in my sources."
- Always cite your sources exactly as they appear (e.g., "Likutei Moharan 1" or "Sippurei Maasiyot 3")
- Be warm, encouraging, and compassionate - this reflects Rabbi Nachman's approach
- If you're unsure about something, say so honestly
- Respect the depth and nuance of the teachings
- Include both Hebrew terms and their explanations when relevant
- Remember: "There is no despair in the world at all!" (אין שום יאוש בעולם כלל)

You have access to retrieved passages from authentic Breslov texts. Use ONLY these passages to ground your responses."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        embeddings_manager: Optional[EmbeddingsManager] = None
    ):
        self.api_key = api_key
        self.model = model

        # Initialize Gemini client
        self.client = genai.Client(api_key=api_key)

        # Initialize embeddings manager
        if embeddings_manager:
            self.embeddings = embeddings_manager
        else:
            self.embeddings = EmbeddingsManager(api_key)

        # Initialize Sefaria fetcher for real-time lookups
        self.sefaria = SefariaFetcher()

        # Chat history for context
        self.chat_history: List[Dict] = []

    def retrieve_context(self, query: str, n_results: int = 5) -> str:
        """
        Retrieve relevant context from vector store

        Args:
            query: User's question
            n_results: Number of passages to retrieve

        Returns:
            Formatted context string
        """
        results = self.embeddings.search(query, n_results=n_results)

        if not results:
            return ""

        context_parts = []
        for i, doc in enumerate(results, 1):
            ref = doc['metadata'].get('ref', 'Unknown')
            title = doc['metadata'].get('title', '')

            # For chunked documents, the full text is in 'text' field
            text_content = doc.get('text', '')

            part = f"[Source {i}: {title} - {ref}]\n"
            # Use the full text from the chunk (up to 1500 chars for good context)
            if text_content:
                part += f"Content: {text_content[:1500]}\n"

            context_parts.append(part)

        return "\n---\n".join(context_parts)

    def generate_response(
        self,
        user_message: str,
        use_rag: bool = True,
        temperature: float = 0.7,
        language: str = 'en'
    ) -> Dict:
        """
        Generate a response using RAG

        Args:
            user_message: User's question or message
            use_rag: Whether to use RAG (retrieve context)
            temperature: Generation temperature
            language: Response language ('en', 'he', 'fr')

        Returns:
            Dict with response, sources, and metadata
        """
        # Retrieve relevant context
        context = ""
        sources = []
        if use_rag:
            context = self.retrieve_context(user_message, n_results=7)  # More context for better answers
            sources = self.embeddings.search(user_message, n_results=5)

        # Get language configuration
        lang_config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS['en'])

        # Build the prompt
        prompt_parts = [self.SYSTEM_PROMPT]

        # Add language instruction
        prompt_parts.append(f"\n\nIMPORTANT: {lang_config['instruction']}")

        if context:
            prompt_parts.append(f"\n\nRelevant passages from Breslov texts (USE ONLY THESE SOURCES):\n{context}")
        else:
            prompt_parts.append("\n\nNote: No relevant passages were found. Please inform the user that you don't have information on this topic.")

        # Add chat history for context
        if self.chat_history:
            history_text = "\n\nPrevious conversation:\n"
            for msg in self.chat_history[-6:]:  # Last 3 exchanges
                role = "User" if msg['role'] == 'user' else "GUEZI"
                history_text += f"{role}: {msg['content'][:500]}\n"
            prompt_parts.append(history_text)

        prompt_parts.append(f"\n\nUser's question: {user_message}\n\nGUEZI's response:")

        full_prompt = "\n".join(prompt_parts)

        # Generate response
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=2048,
                    top_p=0.9,
                )
            )

            response_text = response.text

            # Update chat history
            self.chat_history.append({'role': 'user', 'content': user_message})
            self.chat_history.append({'role': 'assistant', 'content': response_text})

            return {
                'response': response_text,
                'sources': [
                    {
                        'title': s['metadata'].get('title', ''),
                        'ref': s['metadata'].get('ref', ''),
                        'relevance': s.get('relevance_score', 0)
                    }
                    for s in sources
                ],
                'model': self.model,
                'used_rag': use_rag and bool(context),
                'language': language,
                'context_chunks': len(sources)
            }

        except Exception as e:
            return {
                'response': f"I apologize, but I encountered an error: {str(e)}. Please try again.",
                'sources': [],
                'model': self.model,
                'error': str(e)
            }

    def lookup_reference(self, ref: str) -> Dict:
        """
        Look up a specific reference in real-time

        Args:
            ref: Sefaria reference (e.g., "Likutey_Moharan.1.1")

        Returns:
            Text data from Sefaria
        """
        return self.sefaria.get_text(ref)

    def text_to_speech(self, text: str, voice: str = "Kore") -> Optional[bytes]:
        """
        Convert text to speech using Gemini 2.5 Flash TTS

        Args:
            text: Text to convert
            voice: Voice name (Kore, Charon, Puck, etc.)

        Returns:
            Audio bytes (WAV format) or None on error
        """
        try:
            # Use Gemini 2.5 Flash TTS
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice
                            )
                        )
                    )
                )
            )

            # Extract audio data
            if response.candidates and response.candidates[0].content.parts:
                audio_part = response.candidates[0].content.parts[0]
                if hasattr(audio_part, 'inline_data') and audio_part.inline_data:
                    return audio_part.inline_data.data

            return None
        except Exception as e:
            print(f"TTS error: {e}")
            return None

    def generate_response_with_audio(
        self,
        user_message: str,
        language: str = 'en',
        enable_tts: bool = True
    ) -> Dict:
        """
        Generate response with optional TTS audio

        Args:
            user_message: User's question
            language: Response language
            enable_tts: Whether to generate audio

        Returns:
            Dict with response, sources, and optional audio
        """
        # Get text response
        result = self.generate_response(user_message, language=language)

        # Generate audio if enabled
        if enable_tts and result.get('response'):
            lang_config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS['en'])
            audio = self.text_to_speech(
                result['response'],
                voice=lang_config.get('tts_voice', 'Kore')
            )
            if audio:
                result['audio'] = base64.b64encode(audio).decode('utf-8')
                result['audio_format'] = 'wav'

        return result

    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []

    def get_stats(self) -> Dict:
        """Get engine statistics"""
        return {
            'model': self.model,
            'chat_history_length': len(self.chat_history),
            'embeddings_stats': self.embeddings.get_collection_stats()
        }


class ConversationManager:
    """Manages multi-turn conversations with the RAG engine"""

    def __init__(self, rag_engine: GUEZIRagEngine):
        self.engine = rag_engine
        self.sessions: Dict[str, List[Dict]] = {}

    def get_or_create_session(self, session_id: str) -> List[Dict]:
        """Get or create a conversation session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]

    def chat(self, session_id: str, message: str) -> Dict:
        """
        Process a message in a conversation session

        Args:
            session_id: Unique session identifier
            message: User's message

        Returns:
            Response with metadata
        """
        session = self.get_or_create_session(session_id)

        # Temporarily set engine history to session history
        original_history = self.engine.chat_history
        self.engine.chat_history = session

        response = self.engine.generate_response(message)

        # Update session with new messages
        self.sessions[session_id] = self.engine.chat_history.copy()

        # Restore original history
        self.engine.chat_history = original_history

        return response

    def end_session(self, session_id: str):
        """End a conversation session"""
        if session_id in self.sessions:
            del self.sessions[session_id]


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv("config/.env")
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Please set GEMINI_API_KEY in config/.env")
    else:
        engine = GUEZIRagEngine(api_key)
        print("GUEZI RAG Engine initialized!")
        print(f"Stats: {engine.get_stats()}")

        # Test query
        response = engine.generate_response(
            "What does Rabbi Nachman teach about joy and happiness?"
        )
        print(f"\nResponse: {response['response'][:500]}...")
        print(f"Sources: {response['sources']}")

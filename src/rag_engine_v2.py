"""
GUEZI RAG Engine V2
- Recherche hybride (sémantique + références)
- Multi-langue (EN/HE/FR)
- TTS avec Gemini 2.5
- Génération d'images (Nano Banana)
"""

import os
import re
import base64
import asyncio
from typing import List, Dict, Optional
from google import genai
from google.genai import types

try:
    from .embeddings import EmbeddingsManager
    from .sefaria_fetcher import SefariaFetcher
except ImportError:
    from embeddings import EmbeddingsManager
    from sefaria_fetcher import SefariaFetcher


# Configuration des langues
LANGUAGE_CONFIGS = {
    'en': {
        'name': 'English',
        'instruction': 'Respond in English.',
        'greeting': 'How can I help you with Rabbi Nachman\'s teachings?'
    },
    'he': {
        'name': 'Hebrew',
        'instruction': 'Respond in Hebrew (עברית). Use Hebrew script.',
        'greeting': 'במה אוכל לעזור לך בתורת רבי נחמן?'
    },
    'fr': {
        'name': 'French',
        'instruction': 'Respond in French (Français).',
        'greeting': 'Comment puis-je vous aider avec les enseignements de Rabbi Nachman?'
    }
}


class GUEZIRagEngineV2:
    """
    RAG Engine amélioré pour les textes de Breslov
    """

    # Modèles Gemini
    MODEL_CHAT = "gemini-2.0-flash"
    MODEL_TTS = "gemini-2.5-flash-preview-tts"
    MODEL_IMAGE = "gemini-2.5-flash-image"
    MODEL_LIVE = "gemini-2.5-flash-native-audio-preview-12-2025"

    SYSTEM_PROMPT = """You are GUEZI (גואזי), a knowledgeable AI assistant for Rabbi Nachman of Breslov's teachings.

CRITICAL RULES:
1. ONLY use information from the retrieved passages below
2. If no relevant passages found, say "I don't have specific information about that in my sources"
3. NEVER invent teachings, stories, or quotes
4. Always cite the exact source (e.g., "Likutei Moharan 1", "Sippurei Maasiyot 3")
5. Be warm and encouraging - reflect Rabbi Nachman's spirit of hope

Available sources include:
- Likutei Moharan (Part I and II) - Main teachings
- Sippurei Maasiyot - Stories/Tales
- Sichot HaRan - Conversations
- Chayei Moharan - Life of Rabbi Nachman
- Likutei Tefilot - Prayers
- Shivchei HaRan - Praises
- Tikkun HaKlali - Ten Psalms

Remember: "There is no despair in the world at all!" (אין שום יאוש בעולם כלל)"""

    def __init__(
        self,
        api_key: str,
        embeddings_manager: Optional[EmbeddingsManager] = None
    ):
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)

        # Embeddings
        if embeddings_manager:
            self.embeddings = embeddings_manager
        else:
            self.embeddings = EmbeddingsManager(
                api_key,
                persist_dir="./data/faiss_db",
                collection_name="breslov_chunked"
            )

        # Sefaria pour lookups directs
        self.sefaria = SefariaFetcher()

        # Historique
        self.chat_history: List[Dict] = []

        # Cache des références connues
        self._build_reference_index()

    def _build_reference_index(self):
        """Construit un index des références pour recherche directe"""
        self.reference_patterns = {
            r'likute?i?\s*moharan\s*(\d+)': 'Likutei Moharan {}',
            r'likute?i?\s*moharan\s*(?:part\s*)?(?:ii|2)\s*(\d+)': 'Likutei Moharan, Part II {}',
            r'torah\s*(\d+)': 'Likutei Moharan {}',
            r'sippure?i?\s*maasiy?ot\s*(\d+)': 'Sippurei Maasiyot {}',
            r'sichot\s*ha?ran\s*(\d+)': 'Sichot HaRan {}',
            r'chaye?i?\s*moharan\s*(\d+)': 'Chayei Moharan {}',
            r'likute?i?\s*tefilot\s*(\d+)': 'Likutei Tefilot, Volume I {}',
            r'tikkun\s*ha?klali': 'Tikkun HaKlali',
        }

    def _extract_reference(self, query: str) -> Optional[str]:
        """Extrait une référence de livre du texte"""
        query_lower = query.lower()

        for pattern, template in self.reference_patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                if match.groups():
                    return template.format(match.group(1))
                return template

        return None

    def _search_by_reference(self, ref: str, n_results: int = 3) -> List[Dict]:
        """Recherche par référence exacte dans les métadonnées"""
        results = []

        for i, metadata in enumerate(self.embeddings.metadatas):
            if metadata.get('ref', '').lower() == ref.lower():
                results.append({
                    'id': f'doc_{i}',
                    'text': self.embeddings.documents[i],
                    'metadata': metadata,
                    'relevance_score': 1.0,
                    'match_type': 'exact_reference'
                })

        return results[:n_results]

    def hybrid_search(self, query: str, n_results: int = 7) -> List[Dict]:
        """
        Recherche hybride: référence exacte + sémantique
        """
        results = []

        # 1. Essayer d'extraire une référence
        ref = self._extract_reference(query)
        if ref:
            ref_results = self._search_by_reference(ref, n_results=3)
            results.extend(ref_results)

        # 2. Recherche sémantique
        semantic_results = self.embeddings.search(query, n_results=n_results)

        # Combiner et dédupliquer
        seen_refs = {r['metadata'].get('ref') for r in results}
        for sr in semantic_results:
            if sr['metadata'].get('ref') not in seen_refs:
                sr['match_type'] = 'semantic'
                results.append(sr)
                seen_refs.add(sr['metadata'].get('ref'))

        return results[:n_results]

    def retrieve_context(self, query: str, n_results: int = 7) -> str:
        """Récupère le contexte avec recherche hybride"""
        results = self.hybrid_search(query, n_results=n_results)

        if not results:
            return ""

        context_parts = []
        for i, doc in enumerate(results, 1):
            ref = doc['metadata'].get('ref', 'Unknown')
            title = doc['metadata'].get('title', '')
            text = doc.get('text', '')
            match_type = doc.get('match_type', 'unknown')

            part = f"[Source {i}: {title} - {ref}]\n"
            part += f"Content: {text[:1500]}\n"

            context_parts.append(part)

        return "\n---\n".join(context_parts)

    def generate_response(
        self,
        user_message: str,
        language: str = 'en',
        temperature: float = 0.7
    ) -> Dict:
        """Génère une réponse RAG"""

        # Contexte
        context = self.retrieve_context(user_message, n_results=7)
        sources = self.hybrid_search(user_message, n_results=5)

        # Config langue
        lang_config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS['en'])

        # Construction du prompt
        prompt_parts = [self.SYSTEM_PROMPT]
        prompt_parts.append(f"\nIMPORTANT: {lang_config['instruction']}")

        if context:
            prompt_parts.append(f"\n\nRetrieved passages (USE ONLY THESE):\n{context}")
        else:
            prompt_parts.append("\n\nNo relevant passages found. Inform the user.")

        # Historique
        if self.chat_history:
            history_text = "\n\nPrevious conversation:\n"
            for msg in self.chat_history[-4:]:
                role = "User" if msg['role'] == 'user' else "GUEZI"
                history_text += f"{role}: {msg['content'][:300]}\n"
            prompt_parts.append(history_text)

        prompt_parts.append(f"\n\nUser's question: {user_message}\n\nGUEZI:")
        full_prompt = "\n".join(prompt_parts)

        try:
            response = self.client.models.generate_content(
                model=self.MODEL_CHAT,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=2048,
                )
            )

            response_text = response.text

            # Mise à jour historique
            self.chat_history.append({'role': 'user', 'content': user_message})
            self.chat_history.append({'role': 'assistant', 'content': response_text})

            return {
                'response': response_text,
                'sources': [
                    {
                        'title': s['metadata'].get('title', ''),
                        'ref': s['metadata'].get('ref', ''),
                        'relevance': s.get('relevance_score', 0),
                        'match_type': s.get('match_type', 'semantic')
                    }
                    for s in sources
                ],
                'language': language,
                'context_found': bool(context)
            }

        except Exception as e:
            return {
                'response': f"Error: {str(e)}",
                'sources': [],
                'error': str(e)
            }

    def text_to_speech(self, text: str, voice: str = "Kore") -> Optional[bytes]:
        """
        TTS avec Gemini 2.5 Flash TTS
        """
        try:
            response = self.client.models.generate_content(
                model=self.MODEL_TTS,
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

            # Extraire l'audio
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        return part.inline_data.data

            return None
        except Exception as e:
            print(f"TTS error: {e}")
            return None

    def generate_image(self, prompt: str) -> Optional[bytes]:
        """
        Génération d'image avec Nano Banana (Gemini 2.5 Flash Image)
        """
        try:
            # Ajouter style spirituel/judaïque
            enhanced_prompt = f"Spiritual, mystical Jewish art style: {prompt}. Beautiful, inspirational, suitable for meditation."

            response = self.client.models.generate_content(
                model=self.MODEL_IMAGE,
                contents=[enhanced_prompt],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                )
            )

            # Extraire l'image
            for part in response.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    return part.inline_data.data

            return None
        except Exception as e:
            print(f"Image generation error: {e}")
            return None

    def clear_history(self):
        """Efface l'historique"""
        self.chat_history = []

    def get_stats(self) -> Dict:
        """Statistiques"""
        return {
            'model': self.MODEL_CHAT,
            'tts_model': self.MODEL_TTS,
            'image_model': self.MODEL_IMAGE,
            'chat_history': len(self.chat_history),
            'embeddings': self.embeddings.get_collection_stats()
        }


# Test
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv("config/.env")
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("GEMINI_API_KEY not found")
    else:
        engine = GUEZIRagEngineV2(api_key)
        print(f"Stats: {engine.get_stats()}")

        # Test recherche hybride
        test_queries = [
            "What is Torah 1 in Likutei Moharan?",
            "Likutei Moharan 1",
            "Tell me about hitbodedut",
            "Story of the Seven Beggars"
        ]

        for q in test_queries:
            print(f"\n=== {q} ===")
            results = engine.hybrid_search(q, n_results=3)
            for r in results:
                print(f"  [{r.get('match_type')}] {r['metadata'].get('ref')} - {r.get('relevance_score', 0):.3f}")

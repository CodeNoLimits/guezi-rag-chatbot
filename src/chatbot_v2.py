"""
GUEZI Chatbot V2 - Complete Streamlit Interface
- Multi-language (EN/HE/FR)
- Text-to-Speech with Gemini
- Voice Input (via browser Web Speech API)
- Image Generation
"""

import os
import sys
import base64
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_engine_v2 import GUEZIRagEngineV2
from embeddings import EmbeddingsManager


# Voice Input JavaScript Component
VOICE_INPUT_HTML = """
<div id="voice-container" style="
    background: linear-gradient(135deg, #1e1e3f 0%, #252550 100%);
    padding: 15px;
    border-radius: 12px;
    margin: 10px 0;
    text-align: center;
">
    <button id="voice-btn" onclick="toggleVoice()" style="
        width: 100%;
        padding: 12px 20px;
        font-size: 16px;
        font-weight: bold;
        color: white;
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    ">
        🎤 Click to Speak
    </button>
    <p id="voice-status" style="color: #a0a0b0; margin-top: 8px; font-size: 12px;">
        Voice input ready
    </p>
</div>

<script>
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let isListening = false;

if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = '%LANG%';

    recognition.onstart = function() {
        isListening = true;
        document.getElementById('voice-btn').innerHTML = '🔴 Listening...';
        document.getElementById('voice-btn').style.background = 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)';
        document.getElementById('voice-status').textContent = 'Speak now...';
    };

    recognition.onresult = function(event) {
        const transcript = event.results[0][0].transcript;
        document.getElementById('voice-status').textContent = 'Heard: ' + transcript;

        // Send to Streamlit via URL parameter (workaround for component communication)
        const currentUrl = new URL(window.parent.location.href);
        currentUrl.searchParams.set('voice_input', encodeURIComponent(transcript));
        window.parent.history.replaceState({}, '', currentUrl);
        window.parent.location.reload();
    };

    recognition.onerror = function(event) {
        document.getElementById('voice-status').textContent = 'Error: ' + event.error;
        resetButton();
    };

    recognition.onend = function() {
        resetButton();
    };
}

function resetButton() {
    isListening = false;
    document.getElementById('voice-btn').innerHTML = '🎤 Click to Speak';
    document.getElementById('voice-btn').style.background = 'linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)';
}

function toggleVoice() {
    if (!recognition) {
        alert('Speech recognition is not supported in this browser. Please use Chrome, Edge, or Safari.');
        return;
    }

    if (isListening) {
        recognition.stop();
    } else {
        try {
            recognition.start();
        } catch (e) {
            document.getElementById('voice-status').textContent = 'Click again to start';
        }
    }
}
</script>
"""


def init_session_state():
    """Initialize session state"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'language' not in st.session_state:
        st.session_state.language = 'en'
    if 'enable_tts' not in st.session_state:
        st.session_state.enable_tts = False
    if 'engine' not in st.session_state:
        st.session_state.engine = None


def get_api_key():
    """Get API key from Streamlit secrets or environment"""
    # Try Streamlit Cloud secrets first
    try:
        if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
            return st.secrets['GEMINI_API_KEY']
    except:
        pass
    # Fall back to environment variable
    return os.getenv("GEMINI_API_KEY")


def get_supabase_config():
    """Get Supabase config from Streamlit secrets or environment"""
    config = {}
    # Try Streamlit Cloud secrets first
    try:
        if hasattr(st, 'secrets'):
            config['url'] = st.secrets.get('SUPABASE_URL', '')
            config['key'] = st.secrets.get('SUPABASE_KEY', '')
    except:
        pass
    # Fall back to environment variables
    if not config.get('url'):
        config['url'] = os.getenv("SUPABASE_URL", "")
    if not config.get('key'):
        config['key'] = os.getenv("SUPABASE_KEY", "")
    return config


def is_cloud_environment():
    """Check if running on Streamlit Cloud"""
    # Streamlit Cloud sets this environment variable
    return os.getenv("STREAMLIT_SHARING_MODE") is not None or \
           os.getenv("STREAMLIT_SERVER_HEADLESS") == "true"


def get_engine():
    """Get or create engine"""
    if st.session_state.engine is None:
        api_key = get_api_key()
        if api_key:
            try:
                # Check if we should use Supabase (cloud) or FAISS (local)
                supabase_config = get_supabase_config()
                use_supabase = bool(supabase_config.get('url') and supabase_config.get('key'))

                if use_supabase:
                    # Use Supabase for cloud deployment
                    from supabase_embeddings import SupabaseEmbeddingsManager
                    embeddings_manager = SupabaseEmbeddingsManager(
                        api_key=api_key,
                        supabase_url=supabase_config['url'],
                        supabase_key=supabase_config['key']
                    )
                    st.session_state.engine = GUEZIRagEngineV2(
                        api_key,
                        embeddings_manager=embeddings_manager
                    )
                else:
                    # Use local FAISS
                    st.session_state.engine = GUEZIRagEngineV2(api_key)
            except Exception as e:
                st.error(f"Engine error: {e}")
    return st.session_state.engine


def main():
    # Load environment (for local development)
    load_dotenv("config/.env")
    load_dotenv("../config/.env")

    # Page config
    st.set_page_config(
        page_title="GUEZI - Rabbi Nachman AI",
        page_icon="✡️",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    init_session_state()

    # CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Frank+Ruhl+Libre:wght@400;700&display=swap');

    .stApp {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
    }

    .hero-section {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #1e1e3f 0%, #2d2d5a 100%);
        border-radius: 20px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    .hero-section h1 {
        color: #ffffff;
        font-size: 2.5rem;
        font-family: 'Frank Ruhl Libre', serif;
        margin: 0;
    }

    .hero-section .quote {
        color: #fbbf24;
        font-style: italic;
        margin-top: 0.5rem;
    }

    .source-card {
        background: linear-gradient(135deg, #1e1e3f 0%, #252550 100%);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid #6366f1;
    }

    .source-card strong { color: #ffffff; }
    .source-card em { color: #a0a0b0; }
    .match-exact { color: #10b981; }
    .match-semantic { color: #6366f1; }

    .stButton button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

    # Hero
    st.markdown("""
    <div class="hero-section">
        <h1>✡️ GUEZI גואזי</h1>
        <p style="color: #a0a0b0;">AI Assistant for Rabbi Nachman of Breslov</p>
        <p class="quote">אין שום יאוש בעולם כלל<br>There is no despair in the world at all!</p>
        <p style="color: #666; font-size: 10px;">v2.1-hebrew</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        # Language
        st.markdown("#### 🌐 Language")
        language_options = {
            'en': '🇬🇧 English',
            'he': '🇮🇱 עברית',
            'fr': '🇫🇷 Français'
        }
        st.session_state.language = st.selectbox(
            "Response Language",
            options=list(language_options.keys()),
            format_func=lambda x: language_options[x],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Voice
        st.markdown("#### 🔊 Voice")
        st.session_state.enable_tts = st.toggle(
            "Enable Text-to-Speech",
            value=st.session_state.enable_tts
        )

        if st.session_state.enable_tts:
            voice_options = ['Kore', 'Puck', 'Charon', 'Aoede', 'Fenrir']
            if 'tts_voice' not in st.session_state:
                st.session_state.tts_voice = 'Kore'
            st.session_state.tts_voice = st.selectbox(
                "TTS Voice",
                options=voice_options,
                index=voice_options.index(st.session_state.get('tts_voice', 'Kore'))
            )

        # Voice input
        st.markdown("#### 🎤 Voice Input")
        lang_codes = {'en': 'en-US', 'he': 'he-IL', 'fr': 'fr-FR'}
        voice_html = VOICE_INPUT_HTML.replace('%LANG%', lang_codes.get(st.session_state.language, 'en-US'))
        st.components.v1.html(voice_html, height=120)

        st.markdown("---")

        # Image generation
        st.markdown("#### 🎨 Generate Image")
        image_prompt = st.text_input(
            "Describe an image",
            placeholder="A peaceful prayer scene..."
        )
        if st.button("Generate Image", use_container_width=True):
            engine = get_engine()
            if engine and image_prompt:
                with st.spinner("Creating image..."):
                    image_bytes = engine.generate_image(image_prompt)
                    if image_bytes:
                        st.image(image_bytes, caption=image_prompt[:50])
                    else:
                        st.error("Could not generate image")

        st.markdown("---")

        # Stats
        engine = get_engine()
        if engine:
            st.markdown("#### 📊 Stats")
            stats = engine.get_stats()
            st.metric("Documents", stats['embeddings'].get('count', 0))

        # Clear
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.engine:
                st.session_state.engine.clear_history()
            st.rerun()

    # Main area
    engine = get_engine()

    if not engine:
        st.error("⚠️ Could not initialize. Check GEMINI_API_KEY in config/.env")
        return

    # Example questions
    if not st.session_state.messages:
        st.markdown("### 💡 Try asking:")

        examples = {
            'en': [
                "What is Torah 1 in Likutei Moharan?",
                "Tell me about the Seven Beggars",
                "What is hitbodedut?",
                "What is Tikkun HaKlali?"
            ],
            'he': [
                "מה זה ליקוטי מוהר״ן תורה א?",
                "ספר לי על סיפור שבעת הקבצנים",
                "מה זה התבודדות?",
                "מה זה תיקון הכללי?"
            ],
            'fr': [
                "Qu'est-ce que la Torah 1 du Likoutei Moharan?",
                "Parle-moi des Sept Mendiants",
                "Qu'est-ce que le hitbodedout?",
                "Qu'est-ce que le Tikoun Haklali?"
            ]
        }

        cols = st.columns(2)
        current_examples = examples.get(st.session_state.language, examples['en'])
        for i, example in enumerate(current_examples):
            with cols[i % 2]:
                if st.button(example, key=f"ex_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": example})
                    st.rerun()

    # Chat history
    for msg in st.session_state.messages:
        avatar = "🙋" if msg["role"] == "user" else "✡️"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

            # Audio playback
            if msg.get("audio"):
                audio_bytes = base64.b64decode(msg["audio"])
                st.audio(audio_bytes, format="audio/wav")

            # Sources
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for src in msg["sources"]:
                        match_class = "match-exact" if src.get('match_type') == 'exact_reference' else "match-semantic"
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>{src.get('title', '')}</strong> - <em>{src.get('ref', '')}</em><br>
                            <span class="{match_class}">{src.get('match_type', 'semantic')}</span>
                            | Relevance: {src.get('relevance', 0):.0%}
                        </div>
                        """, unsafe_allow_html=True)

    # Check for voice input from URL parameters
    voice_input = st.query_params.get("voice_input", None)
    if voice_input:
        # Clear the parameter to avoid reprocessing
        st.query_params.clear()
        prompt = voice_input
    else:
        # Chat input
        prompt = st.chat_input(
            {
                'en': "Ask about Rabbi Nachman's teachings...",
                'he': "שאל על תורת רבי נחמן...",
                'fr': "Posez une question sur les enseignements..."
            }.get(st.session_state.language, "Ask...")
        )

    if prompt:
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🙋"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant", avatar="✡️"):
            spinner_text = {
                'en': "🕎 Searching the teachings...",
                'he': "🕎 מחפש בתורות...",
                'fr': "🕎 Recherche dans les enseignements..."
            }.get(st.session_state.language, "🕎 Searching...")

            with st.spinner(spinner_text):
                response = engine.generate_response(
                    prompt,
                    language=st.session_state.language
                )

            st.markdown(response["response"])

            # TTS
            audio_b64 = None
            if st.session_state.enable_tts and response.get("response"):
                with st.spinner("🔊 Generating audio..."):
                    voice = st.session_state.get('tts_voice', 'Kore')
                    audio_bytes = engine.text_to_speech(response["response"][:1200], voice=voice)
                    if audio_bytes:
                        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                        st.audio(audio_bytes, format="audio/wav")
                    else:
                        st.caption("⚠️ Audio generation unavailable")

            # Sources
            if response.get("sources"):
                with st.expander("📚 Sources"):
                    for src in response["sources"]:
                        match_class = "match-exact" if src.get('match_type') == 'exact_reference' else "match-semantic"
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>{src.get('title', '')}</strong> - <em>{src.get('ref', '')}</em><br>
                            <span class="{match_class}">{src.get('match_type', 'semantic')}</span>
                            | Relevance: {src.get('relevance', 0):.0%}
                        </div>
                        """, unsafe_allow_html=True)
                        # Show text preview
                        if src.get('text_preview'):
                            st.caption(f"Preview: {src.get('text_preview')[:200]}...")

            # Debug: show context passed to LLM
            if response.get("debug_context"):
                with st.expander("🔍 Debug: Context sent to LLM"):
                    st.text(response.get("debug_context"))

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["response"],
            "sources": response.get("sources", []),
            "audio": audio_b64
        })


if __name__ == "__main__":
    main()

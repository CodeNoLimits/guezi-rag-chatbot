"""
GUEZI Chatbot V3 - Voice-Enabled Edition
- Multi-language (EN/HE/FR)
- Voice Input (Web Speech API)
- Real-time TTS with Gemini
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


# Voice input JavaScript component
VOICE_INPUT_JS = """
<script>
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (SpeechRecognition) {
    window.guezi_recognition = new SpeechRecognition();
    window.guezi_recognition.continuous = false;
    window.guezi_recognition.interimResults = true;
    window.guezi_recognition.lang = '%LANG%';

    window.guezi_recognition.onresult = function(event) {
        let finalTranscript = '';
        let interimTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalTranscript += transcript;
            } else {
                interimTranscript += transcript;
            }
        }

        // Update display
        const display = document.getElementById('voice-transcript');
        if (display) {
            display.value = finalTranscript || interimTranscript;
            display.dispatchEvent(new Event('input', { bubbles: true }));
        }

        // Send final result to Streamlit
        if (finalTranscript) {
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: finalTranscript
            }, '*');
        }
    };

    window.guezi_recognition.onerror = function(event) {
        console.error('Speech recognition error:', event.error);
        const btn = document.getElementById('voice-btn');
        if (btn) {
            btn.innerHTML = '🎤 Click to Speak';
            btn.style.background = 'linear-gradient(135deg, #6366f1 0%%, #4f46e5 100%%)';
        }
    };

    window.guezi_recognition.onend = function() {
        const btn = document.getElementById('voice-btn');
        if (btn) {
            btn.innerHTML = '🎤 Click to Speak';
            btn.style.background = 'linear-gradient(135deg, #6366f1 0%%, #4f46e5 100%%)';
        }
    };
}

function toggleRecording() {
    const btn = document.getElementById('voice-btn');
    if (!window.guezi_recognition) {
        alert('Speech recognition not supported in this browser');
        return;
    }

    try {
        if (btn.dataset.recording === 'true') {
            window.guezi_recognition.stop();
            btn.dataset.recording = 'false';
            btn.innerHTML = '🎤 Click to Speak';
            btn.style.background = 'linear-gradient(135deg, #6366f1 0%%, #4f46e5 100%%)';
        } else {
            window.guezi_recognition.start();
            btn.dataset.recording = 'true';
            btn.innerHTML = '🔴 Listening...';
            btn.style.background = 'linear-gradient(135deg, #ef4444 0%%, #dc2626 100%%)';
        }
    } catch (e) {
        console.error('Error toggling recording:', e);
    }
}
</script>

<style>
#voice-btn {
    width: 100%%;
    padding: 12px 20px;
    font-size: 16px;
    font-weight: bold;
    color: white;
    background: linear-gradient(135deg, #6366f1 0%%, #4f46e5 100%%);
    border: none;
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.3s ease;
    margin: 10px 0;
}
#voice-btn:hover {
    transform: scale(1.02);
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
}
#voice-transcript {
    width: 100%%;
    padding: 10px;
    border: 1px solid #4f46e5;
    border-radius: 8px;
    background: #1e1e2e;
    color: #fafafa;
    font-size: 14px;
    margin-top: 10px;
    resize: none;
}
.voice-container {
    background: linear-gradient(135deg, #1e1e3f 0%%, #252550 100%%);
    padding: 15px;
    border-radius: 12px;
    margin: 10px 0;
}
</style>

<div class="voice-container">
    <button id="voice-btn" onclick="toggleRecording()" data-recording="false">
        🎤 Click to Speak
    </button>
    <textarea id="voice-transcript" rows="2" placeholder="Your speech will appear here..." readonly></textarea>
</div>
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
    if 'voice_text' not in st.session_state:
        st.session_state.voice_text = ""


def get_api_key():
    """Get API key from Streamlit secrets or environment"""
    try:
        if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
            return st.secrets['GEMINI_API_KEY']
    except:
        pass
    return os.getenv("GEMINI_API_KEY")


def get_supabase_config():
    """Get Supabase config from Streamlit secrets or environment"""
    config = {}
    try:
        if hasattr(st, 'secrets'):
            config['url'] = st.secrets.get('SUPABASE_URL', '')
            config['key'] = st.secrets.get('SUPABASE_KEY', '')
    except:
        pass
    if not config.get('url'):
        config['url'] = os.getenv("SUPABASE_URL", "")
    if not config.get('key'):
        config['key'] = os.getenv("SUPABASE_KEY", "")
    return config


def get_engine():
    """Get or create engine"""
    if st.session_state.engine is None:
        api_key = get_api_key()
        if api_key:
            try:
                supabase_config = get_supabase_config()
                use_supabase = bool(supabase_config.get('url') and supabase_config.get('key'))

                if use_supabase:
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
                    st.session_state.engine = GUEZIRagEngineV2(api_key)
            except Exception as e:
                st.error(f"Engine error: {e}")
    return st.session_state.engine


def get_language_code(lang: str) -> str:
    """Get Web Speech API language code"""
    codes = {
        'en': 'en-US',
        'he': 'he-IL',
        'fr': 'fr-FR'
    }
    return codes.get(lang, 'en-US')


def render_voice_input():
    """Render voice input component"""
    lang_code = get_language_code(st.session_state.language)
    js_code = VOICE_INPUT_JS.replace('%LANG%', lang_code)
    st.components.v1.html(js_code, height=150)


def main():
    load_dotenv("config/.env")
    load_dotenv("../config/.env")

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

    .audio-player {
        background: linear-gradient(135deg, #1e1e3f 0%, #252550 100%);
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
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

        # Voice Input Section
        st.markdown("#### 🎤 Voice Input")
        st.info("Click the microphone button below to speak your question")
        render_voice_input()

        st.markdown("---")

        # Voice Output
        st.markdown("#### 🔊 Voice Output")
        st.session_state.enable_tts = st.toggle(
            "Enable Text-to-Speech",
            value=st.session_state.enable_tts
        )

        if st.session_state.enable_tts:
            voice_options = {
                'Kore': '👩 Kore (Female)',
                'Puck': '👨 Puck (Male)',
                'Charon': '👨 Charon (Deep)',
                'Aoede': '👩 Aoede (Warm)',
                'Fenrir': '👨 Fenrir (Strong)'
            }
            if 'tts_voice' not in st.session_state:
                st.session_state.tts_voice = 'Kore'
            st.session_state.tts_voice = st.selectbox(
                "Voice",
                options=list(voice_options.keys()),
                format_func=lambda x: voice_options[x],
                index=list(voice_options.keys()).index(st.session_state.get('tts_voice', 'Kore'))
            )

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
                try:
                    audio_bytes = base64.b64decode(msg["audio"])
                    st.audio(audio_bytes, format="audio/wav")
                except:
                    pass

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
                    # Limit text length for TTS
                    tts_text = response["response"][:1500]
                    audio_bytes = engine.text_to_speech(tts_text, voice=voice)
                    if audio_bytes:
                        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                        st.markdown('<div class="audio-player">', unsafe_allow_html=True)
                        st.audio(audio_bytes, format="audio/wav")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("Audio generation unavailable")

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

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["response"],
            "sources": response.get("sources", []),
            "audio": audio_b64
        })


if __name__ == "__main__":
    main()

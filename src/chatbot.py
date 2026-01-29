"""
GUEZI Chatbot - Streamlit Web Interface
Rabbi Nachman of Breslov AI Assistant
Multi-language support (English, Hebrew, French) with TTS
"""

import os
import sys
import base64
import streamlit as st
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_engine import GUEZIRagEngine, ConversationManager
from embeddings import EmbeddingsManager


def init_session_state():
    """Initialize Streamlit session state"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'language' not in st.session_state:
        st.session_state.language = 'en'

    if 'enable_tts' not in st.session_state:
        st.session_state.enable_tts = False

    if 'engine' not in st.session_state:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                # Use the chunked embeddings collection
                embeddings = EmbeddingsManager(
                    api_key=api_key,
                    persist_dir="./data/faiss_db",
                    collection_name="breslov_chunked"  # New chunked collection
                )
                st.session_state.engine = GUEZIRagEngine(
                    api_key=api_key,
                    embeddings_manager=embeddings
                )
            except Exception as e:
                st.session_state.engine = None
        else:
            st.session_state.engine = None


def main():
    # Load environment
    load_dotenv("config/.env")
    load_dotenv("../config/.env")

    # Page config
    st.set_page_config(
        page_title="GUEZI - Rabbi Nachman AI",
        page_icon="✡️",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Initialize session
    init_session_state()

    # Custom CSS - Modern Dark Theme
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Frank+Ruhl+Libre:wght@400;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --bg-dark: #0f0f23;
        --bg-card: #1a1a2e;
        --text-primary: #ffffff;
        --text-secondary: #a0a0b0;
        --gold: #fbbf24;
        --border: #2d2d44;
    }

    .stApp {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
    }

    /* Main Header */
    .hero-section {
        text-align: center;
        padding: 2.5rem 1.5rem;
        background: linear-gradient(135deg, #1e1e3f 0%, #2d2d5a 50%, #1e1e3f 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(99, 102, 241, 0.3);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15);
    }

    .hero-section h1 {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        font-family: 'Frank Ruhl Libre', serif;
    }

    .hero-section .subtitle {
        color: #a0a0b0;
        font-size: 1.1rem;
        margin: 0.5rem 0;
    }

    .hero-section .quote {
        color: #fbbf24;
        font-size: 1rem;
        font-style: italic;
        margin-top: 1rem;
        font-family: 'Frank Ruhl Libre', serif;
    }

    .star-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    /* Chat Messages */
    .stChatMessage {
        background: #1a1a2e !important;
        border: 1px solid #2d2d44 !important;
        border-radius: 16px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
    }

    /* Source Cards */
    .source-card {
        background: linear-gradient(135deg, #1e1e3f 0%, #252550 100%);
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        border-left: 4px solid #6366f1;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }

    .source-card strong {
        color: #ffffff;
        font-size: 1rem;
    }

    .source-card em {
        color: #a0a0b0;
        font-size: 0.9rem;
    }

    .source-card .relevance {
        color: #10b981;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }

    /* Hebrew Text */
    .hebrew-text {
        direction: rtl;
        font-family: 'Frank Ruhl Libre', serif;
        font-size: 1.15em;
        color: #fbbf24;
        background: rgba(251, 191, 36, 0.1);
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    /* Input Area */
    .stChatInput {
        background: #1a1a2e !important;
        border: 1px solid #2d2d44 !important;
        border-radius: 12px !important;
    }

    .stChatInput input {
        background: transparent !important;
        color: #ffffff !important;
    }

    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: #0f0f23 !important;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 500 !important;
    }

    .stButton button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
        transform: translateY(-1px);
    }

    /* Example Questions */
    .example-btn {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #a0a0b0;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        cursor: pointer;
        transition: all 0.2s;
        font-size: 0.9rem;
    }

    .example-btn:hover {
        background: rgba(99, 102, 241, 0.2);
        color: #ffffff;
    }

    /* Quick Actions */
    .quick-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        justify-content: center;
        margin: 1.5rem 0;
    }

    /* Stats Card */
    .stats-card {
        background: #1a1a2e;
        border: 1px solid #2d2d44;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }

    .stats-card h4 {
        color: #6366f1;
        margin: 0 0 0.5rem 0;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Expander */
    .streamlit-expanderHeader {
        background: #1a1a2e !important;
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="star-icon">✡️</div>
        <h1>GUEZI גואזי</h1>
        <p class="subtitle">AI Assistant for Rabbi Nachman of Breslov Teachings</p>
        <p class="quote">אין שום יאוש בעולם כלל<br>There is no despair in the world at all!</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        # API Key input
        api_key = st.text_input(
            "🔑 Gemini API Key",
            type="password",
            value=os.getenv("GEMINI_API_KEY", ""),
            help="Enter your Google Gemini API key"
        )

        if api_key:
            if not st.session_state.engine:
                try:
                    # Use the chunked embeddings collection
                    embeddings = EmbeddingsManager(
                        api_key=api_key,
                        persist_dir="./data/faiss_db",
                        collection_name="breslov_chunked"
                    )
                    st.session_state.engine = GUEZIRagEngine(
                        api_key=api_key,
                        embeddings_manager=embeddings
                    )
                    st.success(f"✓ Connected! {embeddings.index.ntotal} chunks loaded")
                except Exception as e:
                    st.error(f"Connection error: {str(e)[:50]}")

        st.markdown("---")

        # Language Settings
        st.markdown("### 🌐 Language / שפה")
        language_options = {
            'en': '🇬🇧 English',
            'he': '🇮🇱 עברית (Hebrew)',
            'fr': '🇫🇷 Français (French)'
        }
        st.session_state.language = st.selectbox(
            "Response Language",
            options=list(language_options.keys()),
            format_func=lambda x: language_options[x],
            index=list(language_options.keys()).index(st.session_state.language)
        )

        st.markdown("---")

        # Voice Settings
        st.markdown("### 🔊 Voice (TTS)")
        st.session_state.enable_tts = st.toggle(
            "Enable Text-to-Speech",
            value=st.session_state.enable_tts,
            help="Use Gemini 2.5 TTS to read responses aloud"
        )

        st.markdown("---")

        # RAG Settings
        st.markdown("### 🎯 RAG Settings")
        use_rag = st.toggle("Use RAG (Retrieve Sources)", value=True)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)

        st.markdown("---")

        # Stats
        if st.session_state.engine:
            st.markdown("### 📊 Stats")
            stats = st.session_state.engine.get_stats()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats['embeddings_stats'].get('count', 0))
            with col2:
                st.metric("Messages", len(st.session_state.messages))

        st.markdown("---")

        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.engine:
                st.session_state.engine.clear_history()
            st.rerun()

        st.markdown("---")
        st.markdown("### 📖 Resources")
        st.markdown("[Sefaria](https://www.sefaria.org) | [Breslov.org](https://breslov.org)")

    # Main chat area
    if not st.session_state.engine:
        st.warning("⚠️ Please enter your Gemini API key in the sidebar to start chatting.")

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.link_button("🔑 Get API Key", "https://makersuite.google.com/app/apikey", use_container_width=True)
        return

    # Example questions (if no messages)
    if not st.session_state.messages:
        st.markdown("### 💡 Try asking:")

        examples = [
            "What is hitbodedut?",
            "Tell me about the Seven Beggars",
            "How to find joy in hard times?",
            "What is Tikkun HaKlali?",
            "Explain אין שום יאוש",
            "Rabbi Nachman on prayer"
        ]

        cols = st.columns(3)
        for i, example in enumerate(examples):
            with cols[i % 3]:
                if st.button(example, key=f"ex_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": example})
                    st.rerun()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🙋" if message["role"] == "user" else "✡️"):
            st.markdown(message["content"])

            # Show sources if available
            if message.get("sources") and len(message["sources"]) > 0:
                with st.expander("📚 View Sources"):
                    for source in message["sources"]:
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>{source.get('title', 'Unknown')}</strong><br>
                            <em>{source.get('ref', '')}</em>
                            <div class="relevance">Relevance: {source.get('relevance', 0):.0%}</div>
                        </div>
                        """, unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask about Rabbi Nachman's teachings..."):
        # Add user message
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
                if st.session_state.enable_tts:
                    response = st.session_state.engine.generate_response_with_audio(
                        prompt,
                        language=st.session_state.language,
                        enable_tts=True
                    )
                else:
                    response = st.session_state.engine.generate_response(
                        prompt,
                        use_rag=use_rag,
                        temperature=temperature,
                        language=st.session_state.language
                    )

            st.markdown(response["response"])

            # Audio playback if TTS enabled
            if response.get("audio"):
                audio_bytes = base64.b64decode(response["audio"])
                st.audio(audio_bytes, format="audio/wav")

            # Show sources
            if response.get("sources") and len(response["sources"]) > 0:
                source_label = {
                    'en': "📚 View Sources",
                    'he': "📚 הצג מקורות",
                    'fr': "📚 Voir les sources"
                }.get(st.session_state.language, "📚 Sources")

                with st.expander(source_label):
                    for source in response["sources"]:
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>{source.get('title', 'Unknown')}</strong><br>
                            <em>{source.get('ref', '')}</em>
                            <div class="relevance">Relevance: {source.get('relevance', 0):.0%}</div>
                        </div>
                        """, unsafe_allow_html=True)

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["response"],
            "sources": response.get("sources", [])
        })


if __name__ == "__main__":
    main()

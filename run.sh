#!/bin/bash
# GUEZI Chatbot Runner - Multi-language with TTS

cd "$(dirname "$0")"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check for .env file
if [ ! -f "config/.env" ]; then
    echo "⚠️  No config/.env found. Creating from example..."
    cp config/.env.example config/.env
    echo "📝 Please edit config/.env and add your GEMINI_API_KEY"
    exit 1
fi

# Check if corpus is set up
if [ ! -d "data/faiss_db" ]; then
    echo "📥 Setting up corpus (first run)..."
    python setup_corpus.py --test
fi

# Run chatbot V2
echo "🚀 Starting GUEZI Chatbot V2..."
echo "📍 Open http://localhost:8501 in your browser"
streamlit run src/chatbot_v2.py

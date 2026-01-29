# GUEZI - Rabbi Nachman AI Assistant

![GUEZI](https://img.shields.io/badge/GUEZI-Rabbi%20Nachman%20AI-6366f1)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Gemini](https://img.shields.io/badge/Google-Gemini%202.5-red)

> **אין שום יאוש בעולם כלל** - There is no despair in the world at all!

GUEZI is an AI assistant specialized in the teachings of Rabbi Nachman of Breslov. It uses RAG (Retrieval Augmented Generation) to provide accurate answers based on authentic Breslov texts.

## Features

- 📚 **RAG-powered responses** - Answers based on authentic Breslov texts
- 🌐 **Multi-language** - English, Hebrew, French
- 🔊 **Text-to-Speech** - Gemini 2.5 TTS
- 🎨 **Image Generation** - Nano Banana (Gemini 2.5 Flash Image)
- 🔍 **Hybrid Search** - Exact reference + semantic search

## Sources

All texts from [Sefaria](https://www.sefaria.org):
- Likutei Moharan (Part I & II)
- Sippurei Maasiyot (Stories)
- Sichot HaRan (Conversations)
- Chayei Moharan (Life of Rabbi Nachman)
- Likutei Tefilot (Prayers)
- Shivchei HaRan (Praises)
- Tikkun HaKlali

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/guezi-rag-chatbot.git
cd guezi-rag-chatbot
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure API key
```bash
cp config/.env.example config/.env
# Edit config/.env and add your GEMINI_API_KEY
```

### 4. Build embeddings (first time only)
```bash
python src/sefaria_fetcher.py  # Fetch texts
python src/semantic_chunker.py  # Chunk texts
python src/build_embeddings.py  # Create embeddings
```

### 5. Run the chatbot
```bash
./run.sh
# or
streamlit run src/chatbot_v2.py
```

Open http://localhost:8501 in your browser.

## Project Structure

```
guezi-rag-chatbot/
├── config/
│   └── .env.example
├── data/
│   └── (generated data files)
├── src/
│   ├── chatbot_v2.py      # Streamlit UI
│   ├── rag_engine_v2.py   # RAG engine
│   ├── embeddings.py      # FAISS vector store
│   ├── semantic_chunker.py # Text chunking
│   └── sefaria_fetcher.py # Data fetcher
├── requirements.txt
└── run.sh
```

## API Models Used

| Feature | Model |
|---------|-------|
| Chat | `gemini-2.0-flash` |
| TTS | `gemini-2.5-flash-preview-tts` |
| Image | `gemini-2.5-flash-image` |
| Embeddings | `gemini-embedding-001` |

## License

MIT

## Acknowledgments

- [Sefaria](https://www.sefaria.org) for the texts
- [Google Gemini](https://ai.google.dev) for AI capabilities
- Rabbi Nachman of Breslov for the eternal teachings

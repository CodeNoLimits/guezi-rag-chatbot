# GUEZI RAG Chatbot - Plan d'Action

## État: ✅ COMPLÉTÉ

## Problèmes Résolus

| Problème | Statut | Solution |
|----------|--------|----------|
| RAG ne trouve pas Likutei Moharan 1 | ✅ | Recherche hybride (référence + sémantique) |
| TTS ne fonctionne pas | ✅ | Prompt corrigé pour Gemini 2.5 TTS |
| Pas de génération d'images | ✅ | Nano Banana (gemini-2.5-flash-image) |
| Multi-langue | ✅ | EN/HE/FR supportés |
| Pas de repo GitHub | ✅ | https://github.com/CodeNoLimits/guezi-rag-chatbot |

## Modèles Gemini Utilisés

| Fonction | Modèle |
|----------|--------|
| Chat | `gemini-2.0-flash` |
| TTS | `gemini-2.5-flash-preview-tts` |
| Image | `gemini-2.5-flash-image` |
| Embeddings | `gemini-embedding-001` |
| Live API (voice) | `gemini-2.5-flash-native-audio-preview-12-2025` |

## Fichiers Clés

- `src/rag_engine_v2.py` - Moteur RAG avec recherche hybride
- `src/chatbot_v2.py` - Interface Streamlit
- `src/embeddings.py` - Gestion FAISS
- `src/semantic_chunker.py` - Chunking intelligent

## Pour Lancer

```bash
./run.sh
# ou
source venv/bin/activate && streamlit run src/chatbot_v2.py
```

## TODO Optionnel

- [ ] Voice input avec Live API (nécessite WebSocket + pyaudio)
- [ ] Déploiement Streamlit Cloud
- [ ] Supabase pour persistance cloud

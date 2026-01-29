#!/usr/bin/env python3
"""
GUEZI Chatbot - Corpus Setup Script
Fetches Breslov texts from Sefaria and creates embeddings
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from sefaria_fetcher import SefariaFetcher
from embeddings import EmbeddingsManager


def setup_corpus(api_key: str, force_refetch: bool = False):
    """
    Main setup function

    1. Fetch texts from Sefaria
    2. Create embeddings
    3. Store in ChromaDB
    """
    print("=" * 60)
    print("GUEZI Chatbot - Corpus Setup")
    print("Rabbi Nachman of Breslov RAG System")
    print("=" * 60)

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    corpus_path = data_dir / "breslov_corpus.json"

    # Step 1: Fetch corpus from Sefaria
    if corpus_path.exists() and not force_refetch:
        print(f"\n📚 Loading existing corpus from {corpus_path}")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        print(f"   Loaded {len(corpus)} documents")
    else:
        print("\n📥 Fetching Breslov corpus from Sefaria...")
        fetcher = SefariaFetcher()
        corpus = fetcher.fetch_breslov_corpus(str(corpus_path))

    if not corpus:
        print("❌ Failed to fetch corpus. Please check your internet connection.")
        return False

    # Step 2: Create embeddings
    print("\n🧮 Creating embeddings with Gemini...")
    embeddings_manager = EmbeddingsManager(
        api_key=api_key,
        persist_dir=str(data_dir / "faiss_db"),
        collection_name="rabbi_nachman_texts"
    )

    # Check if already populated
    stats = embeddings_manager.get_collection_stats()
    if stats['count'] > 0 and not force_refetch:
        print(f"   Vector store already has {stats['count']} documents")
        choice = input("   Do you want to clear and recreate? (y/N): ").strip().lower()
        if choice == 'y':
            embeddings_manager.clear_collection()
        else:
            print("   Keeping existing embeddings")
            return True

    # Add documents
    embeddings_manager.add_documents(corpus)

    # Verify
    final_stats = embeddings_manager.get_collection_stats()
    print(f"\n✅ Setup complete!")
    print(f"   Documents in vector store: {final_stats['count']}")
    print(f"   Persist directory: {final_stats['persist_dir']}")

    return True


def test_search(api_key: str):
    """Test the search functionality"""
    print("\n🔍 Testing search...")

    data_dir = Path(__file__).parent / "data"
    embeddings_manager = EmbeddingsManager(
        api_key=api_key,
        persist_dir=str(data_dir / "faiss_db"),
        collection_name="rabbi_nachman_texts"
    )

    test_queries = [
        "What is hitbodedut?",
        "simcha joy happiness",
        "story seven beggars",
        "tikkun haklali",
        "אין שום יאוש"
    ]

    for query in test_queries:
        print(f"\n   Query: '{query}'")
        results = embeddings_manager.search(query, n_results=2)
        for i, r in enumerate(results, 1):
            print(f"   {i}. {r['metadata'].get('title', '')} - {r['metadata'].get('ref', '')}")
            print(f"      Relevance: {r.get('relevance_score', 0):.2%}")


def main():
    parser = argparse.ArgumentParser(
        description="Setup GUEZI Chatbot corpus and embeddings"
    )
    parser.add_argument(
        "--api-key",
        help="Gemini API key (or set GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force refetch and recreate embeddings"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run search tests after setup"
    )

    args = parser.parse_args()

    # Load environment
    load_dotenv("config/.env")

    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: Gemini API key required")
        print("   Set GEMINI_API_KEY environment variable or use --api-key")
        print("   Get your key at: https://makersuite.google.com/app/apikey")
        sys.exit(1)

    success = setup_corpus(api_key, args.force)

    if success and args.test:
        test_search(api_key)

    if success:
        print("\n" + "=" * 60)
        print("🎉 Setup complete! Run the chatbot with:")
        print("   cd guezi-rag-chatbot")
        print("   streamlit run src/chatbot.py")
        print("=" * 60)


if __name__ == "__main__":
    main()

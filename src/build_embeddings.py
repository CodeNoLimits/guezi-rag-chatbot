"""
Build Embeddings from Chunked Corpus
Creates FAISS index from semantically chunked documents
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings import EmbeddingsManager


def build_embeddings_from_chunks():
    """Build embeddings from the chunked corpus"""

    # Load environment
    load_dotenv("config/.env")
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Error: GEMINI_API_KEY not found in config/.env")
        return

    # Load chunked corpus
    chunked_path = "data/breslov_chunked.json"
    if not os.path.exists(chunked_path):
        print(f"Error: {chunked_path} not found. Run semantic_chunker.py first.")
        return

    print("Loading chunked corpus...")
    with open(chunked_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks")

    # Show sample
    print("\nSample chunk:")
    sample = chunks[0]
    print(f"  Title: {sample.get('title', '')}")
    print(f"  Ref: {sample.get('ref', '')}")
    print(f"  Chunk ID: {sample.get('chunk_id', '')}")
    print(f"  Length: {len(sample.get('combined', ''))} chars")

    # Initialize embeddings manager with fresh collection
    print("\nInitializing embeddings manager...")
    manager = EmbeddingsManager(
        api_key=api_key,
        persist_dir="./data/faiss_db",
        collection_name="breslov_chunked"  # New collection for chunked data
    )

    # Clear existing data
    print("Clearing existing embeddings...")
    manager.clear_collection()

    # Add documents in batches
    print(f"\nCreating embeddings for {len(chunks)} chunks...")
    print("This will take some time due to API rate limits...")

    manager.add_documents(chunks, text_field="combined")

    # Show stats
    stats = manager.get_collection_stats()
    print(f"\nFinal stats: {stats}")

    # Test search
    print("\n" + "="*60)
    print("TESTING SEARCH QUALITY")
    print("="*60)

    test_queries = [
        "What does Rabbi Nachman say about hitbodedut?",
        "What is the teaching about the broken heart?",
        "Tell me about the tale of the seven beggars",
        "מה רבי נחמן אומר על התבודדות?",
        "What happened in Uman?"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = manager.search(query, n_results=3)
        for i, r in enumerate(results[:2]):
            print(f"  {i+1}. [{r['metadata'].get('title', '')}] Score: {r['relevance_score']:.3f}")
            print(f"     {r['text'][:150]}...")

    print("\nEmbeddings built successfully!")
    return manager


if __name__ == "__main__":
    build_embeddings_from_chunks()

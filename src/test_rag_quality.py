"""
RAG Quality Test Suite
Tests retrieval quality across different Breslov books
"""

import os
import sys
import json
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embeddings import EmbeddingsManager
from rag_engine import GUEZIRagEngine


# Test queries organized by book/topic
TEST_QUERIES = {
    "Likutei Moharan": [
        "What does Likutei Moharan teach about azamra finding good points?",
        "What is Torah 282 about?",
        "Explain the concept of mochin de-gadlut in Likutei Moharan",
        "What does Rabbi Nachman say about joy in Likutei Moharan?"
    ],
    "Sippurei Maasiyot": [
        "Tell me about the Seven Beggars story",
        "What is the story of the Lost Princess about?",
        "Who are the characters in the Master of Prayer?",
        "What happens in the tale of the Sophisticate and the Simpleton?"
    ],
    "Sichot HaRan": [
        "What does Rabbi Nachman teach about hitbodedut?",
        "What did Rabbi Nachman say about simplicity?",
        "Rabbi Nachman's advice on overcoming depression"
    ],
    "Chayei Moharan": [
        "What happened to Rabbi Nachman in Uman?",
        "Tell me about Rabbi Nachman's journey to Israel",
        "When was Rabbi Nachman born?"
    ],
    "Tikkun HaKlali": [
        "What are the ten psalms of Tikkun HaKlali?",
        "Why did Rabbi Nachman establish Tikkun HaKlali?"
    ]
}


def test_retrieval_quality(embeddings: EmbeddingsManager, verbose: bool = True):
    """Test retrieval quality for each book"""
    print("\n" + "="*70)
    print("RETRIEVAL QUALITY TEST")
    print("="*70)

    results = {}

    for book, queries in TEST_QUERIES.items():
        print(f"\n--- Testing: {book} ---")
        book_scores = []

        for query in queries:
            search_results = embeddings.search(query, n_results=5)

            if not search_results:
                print(f"  [FAIL] No results for: {query[:50]}...")
                book_scores.append(0)
                continue

            # Check if top result is from the expected book
            top_result = search_results[0]
            top_title = top_result['metadata'].get('title', '')
            top_score = top_result.get('relevance_score', 0)

            # Book match check (flexible matching)
            book_keywords = book.lower().split()
            title_lower = top_title.lower()
            is_match = any(kw in title_lower for kw in book_keywords)

            if is_match:
                print(f"  [OK] {query[:50]}...")
                print(f"       -> {top_title} (score: {top_score:.3f})")
                book_scores.append(top_score)
            else:
                print(f"  [WARN] {query[:50]}...")
                print(f"       -> Got: {top_title} (expected: {book})")
                book_scores.append(top_score * 0.5)  # Partial credit

            if verbose:
                print(f"       Text preview: {search_results[0]['text'][:100]}...")

        # Calculate book average
        if book_scores:
            avg_score = sum(book_scores) / len(book_scores)
            results[book] = {
                'avg_score': avg_score,
                'queries_tested': len(queries),
                'scores': book_scores
            }
            print(f"  Average score for {book}: {avg_score:.3f}")

    return results


def test_generation_quality(engine: GUEZIRagEngine):
    """Test that the model doesn't hallucinate"""
    print("\n" + "="*70)
    print("GENERATION QUALITY TEST (Hallucination Check)")
    print("="*70)

    test_cases = [
        {
            "query": "What does Torah 64 in Likutei Moharan say?",
            "check": "Should cite sources, not invent content"
        },
        {
            "query": "What is the meaning of the story of the Turkey Prince?",
            "check": "Should find and reference actual story"
        },
        {
            "query": "What did Rabbi Nachman say about aliens?",
            "check": "Should say no information available (hallucination test)"
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test['check']} ---")
        print(f"Query: {test['query']}")

        response = engine.generate_response(test['query'], language='en')

        print(f"Sources found: {len(response.get('sources', []))}")
        if response.get('sources'):
            for s in response['sources'][:2]:
                print(f"  - {s.get('title', '')} / {s.get('ref', '')}")

        print(f"\nResponse preview:")
        print(f"  {response['response'][:300]}...")
        print()


def run_full_test():
    """Run complete test suite"""
    load_dotenv("config/.env")
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Error: GEMINI_API_KEY not found")
        return

    # Initialize with chunked collection
    print("Loading chunked embeddings...")
    embeddings = EmbeddingsManager(
        api_key=api_key,
        persist_dir="./data/faiss_db",
        collection_name="breslov_chunked"
    )

    stats = embeddings.get_collection_stats()
    print(f"Loaded {stats['count']} chunks")

    if stats['count'] == 0:
        print("ERROR: No embeddings found! Run build_embeddings.py first.")
        return

    # Test retrieval
    retrieval_results = test_retrieval_quality(embeddings)

    # Initialize RAG engine
    print("\nInitializing RAG engine for generation tests...")
    engine = GUEZIRagEngine(api_key, embeddings_manager=embeddings)

    # Test generation
    test_generation_quality(engine)

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    overall_scores = []
    for book, data in retrieval_results.items():
        print(f"  {book}: {data['avg_score']:.3f} ({data['queries_tested']} queries)")
        overall_scores.extend(data['scores'])

    if overall_scores:
        print(f"\n  OVERALL AVERAGE: {sum(overall_scores)/len(overall_scores):.3f}")

    print("\nTest complete!")


if __name__ == "__main__":
    run_full_test()

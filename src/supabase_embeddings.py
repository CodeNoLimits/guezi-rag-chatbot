"""
Supabase Embeddings Manager for GUEZI RAG
Cloud-based vector storage using Supabase with pgvector.

This module provides the same interface as EmbeddingsManager but uses
Supabase instead of local FAISS.
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv("config/.env")


class SupabaseEmbeddingsManager:
    """
    Manages embeddings in Supabase with pgvector.
    Drop-in replacement for EmbeddingsManager.
    """

    def __init__(
        self,
        api_key: str,  # Gemini API key for generating embeddings
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        table_name: str = "breslov_documents"
    ):
        from supabase import create_client, Client
        from google import genai

        self.api_key = api_key
        self.table_name = table_name

        # Initialize Gemini client for embeddings
        self.gemini_client = genai.Client(api_key=api_key)
        self.embedding_model = "models/text-embedding-004"

        # Initialize Supabase client
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")

        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase URL and key required")

        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)

        # Cache for compatibility with EmbeddingsManager interface
        self._documents_cache = []
        self._metadatas_cache = []

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        result = self.gemini_client.models.embed_content(
            model=self.embedding_model,
            contents=text
        )
        return result.embeddings[0].values

    def search(
        self,
        query: str,
        n_results: int = 5,
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Search for similar documents using Supabase pgvector.

        Args:
            query: Search query text
            n_results: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of matching documents with metadata and scores
        """
        # Generate query embedding
        query_embedding = self._generate_embedding(query)

        # Call Supabase RPC function
        try:
            result = self.supabase.rpc(
                "match_breslov_documents",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": threshold,
                    "match_count": n_results
                }
            ).execute()

            # Format results
            results = []
            for row in result.data:
                results.append({
                    "id": str(row.get("id")),
                    "text": row.get("combined", ""),
                    "metadata": {
                        "title": row.get("title", ""),
                        "ref": row.get("ref", ""),
                        "chunk_id": row.get("chunk_id", ""),
                        "hebrew": row.get("hebrew", ""),
                        "english": row.get("english", ""),
                    },
                    "relevance_score": row.get("similarity", 0)
                })

            return results

        except Exception as e:
            print(f"Supabase search error: {e}")
            return []

    def hybrid_search(
        self,
        query: str,
        ref: Optional[str] = None,
        n_results: int = 7,
        threshold: float = 0.4
    ) -> List[Dict]:
        """
        Hybrid search combining exact reference match and semantic search.

        Args:
            query: Search query text
            ref: Optional exact reference to match (e.g., "Likutei Moharan 1")
            n_results: Number of results to return
            threshold: Minimum similarity threshold for semantic matches

        Returns:
            List of matching documents
        """
        # Generate query embedding
        query_embedding = self._generate_embedding(query)

        try:
            if ref:
                # Use hybrid search RPC
                result = self.supabase.rpc(
                    "hybrid_search_breslov",
                    {
                        "search_ref": ref,
                        "query_embedding": query_embedding,
                        "match_threshold": threshold,
                        "match_count": n_results
                    }
                ).execute()
            else:
                # Fall back to regular semantic search
                result = self.supabase.rpc(
                    "match_breslov_documents",
                    {
                        "query_embedding": query_embedding,
                        "match_threshold": threshold,
                        "match_count": n_results
                    }
                ).execute()

            # Format results
            results = []
            for row in result.data:
                results.append({
                    "id": str(row.get("id")),
                    "text": row.get("combined", ""),
                    "metadata": {
                        "title": row.get("title", ""),
                        "ref": row.get("ref", ""),
                    },
                    "relevance_score": row.get("similarity", 0),
                    "match_type": row.get("match_type", "semantic")
                })

            return results

        except Exception as e:
            print(f"Supabase hybrid search error: {e}")
            return []

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict],
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        Add documents with embeddings to Supabase.

        Args:
            documents: List of document texts
            metadatas: List of metadata dicts for each document
            ids: Optional list of unique IDs

        Returns:
            True if successful
        """
        records = []

        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            # Generate embedding
            embedding = self._generate_embedding(doc)

            record = {
                "title": meta.get("title", "Unknown"),
                "ref": meta.get("ref", f"doc_{i}"),
                "chunk_id": ids[i] if ids else meta.get("chunk_id", f"chunk_{i}"),
                "hebrew": meta.get("hebrew", ""),
                "english": meta.get("english", ""),
                "combined": doc,
                "embedding": embedding,
                "chunk_index": meta.get("chunk_index", 0),
                "total_chunks": meta.get("total_chunks", 1),
            }
            records.append(record)

        try:
            self.supabase.table(self.table_name).upsert(
                records,
                on_conflict="chunk_id"
            ).execute()
            return True
        except Exception as e:
            print(f"Error adding documents: {e}")
            return False

    def get_collection_stats(self) -> Dict:
        """Get statistics about the Supabase collection"""
        try:
            result = self.supabase.table(self.table_name).select(
                "id",
                count="exact"
            ).execute()

            return {
                "total_documents": result.count or 0,
                "storage": "supabase",
                "table": self.table_name,
                "url": self.supabase_url
            }
        except Exception as e:
            return {
                "error": str(e),
                "storage": "supabase"
            }

    @property
    def documents(self) -> List[str]:
        """Compatibility property - returns cached documents"""
        return self._documents_cache

    @property
    def metadatas(self) -> List[Dict]:
        """Compatibility property - returns cached metadata"""
        return self._metadatas_cache


# Test
if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("GEMINI_API_KEY not found")
    else:
        manager = SupabaseEmbeddingsManager(api_key)
        print(f"Stats: {manager.get_collection_stats()}")

        # Test search
        results = manager.search("What is hitbodedut?", n_results=3)
        print(f"\nSearch results: {len(results)}")
        for r in results:
            print(f"  - {r['metadata'].get('ref')}: {r['relevance_score']:.3f}")

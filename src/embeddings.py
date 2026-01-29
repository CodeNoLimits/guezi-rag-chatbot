"""
Embeddings Manager
Handles vector embeddings using FAISS and Google's embedding model
"""

import os
import json
import pickle
import time
from typing import List, Dict, Optional
import numpy as np
import faiss
from google import genai
from tqdm import tqdm


class EmbeddingsManager:
    """Manage text embeddings with FAISS"""

    def __init__(
        self,
        api_key: str,
        persist_dir: str = "./data/faiss_db",
        collection_name: str = "breslov_complete"
    ):
        self.api_key = api_key
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_dim = 3072  # Default for gemini-embedding-001

        # Initialize Gemini client for embeddings
        self.client = genai.Client(api_key=api_key)

        # Initialize storage
        os.makedirs(persist_dir, exist_ok=True)

        # Paths
        self.index_path = os.path.join(persist_dir, f"{collection_name}.index")
        self.metadata_path = os.path.join(persist_dir, f"{collection_name}_metadata.pkl")

        # Load or create index
        self.index = None
        self.documents = []
        self.metadatas = []
        self._load_or_create_index()

    def _load_or_create_index(self):
        """Load existing index or create new one"""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get('documents', [])
                    self.metadatas = data.get('metadatas', [])
                print(f"Loaded existing index with {self.index.ntotal} vectors")
            except Exception as e:
                print(f"Error loading index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """Create a new FAISS index"""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.documents = []
        self.metadatas = []

    def _save_index(self):
        """Save index and metadata to disk"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadatas': self.metadatas
            }, f)

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            result = self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=text
            )
            return result.embeddings[0].values
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return []

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:
        """Get embeddings for multiple texts in batches"""
        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch = texts[i:i + batch_size]
            try:
                result = self.client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=batch
                )
                batch_embeddings = [emb.values for emb in result.embeddings]
                embeddings.extend(batch_embeddings)
                # Add delay to avoid rate limiting
                time.sleep(1)
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                # Retry after longer delay
                time.sleep(5)
                try:
                    result = self.client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=batch
                    )
                    batch_embeddings = [emb.values for emb in result.embeddings]
                    embeddings.extend(batch_embeddings)
                except Exception as e2:
                    print(f"Retry failed: {e2}")
                    # Add empty embeddings for failed batch
                    embeddings.extend([[] for _ in batch])

        return embeddings

    def add_documents(self, documents: List[Dict], text_field: str = "combined"):
        """
        Add documents to the vector store

        Args:
            documents: List of document dicts with text and metadata
            text_field: Field containing the text to embed
        """
        if not documents:
            print("No documents to add")
            return

        print(f"Adding {len(documents)} documents to vector store...")

        # Prepare data
        texts = []
        metadatas = []

        for i, doc in enumerate(documents):
            text = doc.get(text_field, "")
            if not text:
                continue

            texts.append(text[:8000])  # Limit text length

            metadata = {
                'title': doc.get('title', ''),
                'ref': doc.get('ref', ''),
                'hebrew': doc.get('hebrew', '')[:1000],
                'english': doc.get('english', '')[:1000]
            }
            metadatas.append(metadata)

        # Get embeddings
        embeddings = self.get_embeddings_batch(texts)

        # Filter out failed embeddings
        valid_data = [
            (text, emb, meta)
            for text, emb, meta in zip(texts, embeddings, metadatas)
            if emb
        ]

        if not valid_data:
            print("No valid embeddings generated")
            return

        texts, embeddings, metadatas = zip(*valid_data)

        # Update embedding dimension if needed
        if embeddings and len(embeddings[0]) != self.embedding_dim:
            self.embedding_dim = len(embeddings[0])
            self._create_new_index()

        # Add to FAISS index
        embeddings_array = np.array(embeddings).astype('float32')
        self.index.add(embeddings_array)

        # Store documents and metadata
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)

        # Save to disk
        self._save_index()

        print(f"Successfully added {len(texts)} documents. Total: {self.index.ntotal}")

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents

        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filter (not implemented for FAISS)

        Returns:
            List of relevant documents with scores
        """
        if self.index.ntotal == 0:
            return []

        # Get query embedding
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []

        # Search
        query_array = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_array, min(n_results, self.index.ntotal))

        # Format results
        documents = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx < 0 or idx >= len(self.documents):
                continue

            # Convert L2 distance to similarity score (inverse)
            relevance = 1 / (1 + distance)

            doc = {
                'id': f"doc_{idx}",
                'text': self.documents[idx],
                'metadata': self.metadatas[idx],
                'distance': float(distance),
                'relevance_score': relevance
            }
            documents.append(doc)

        return documents

    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        return {
            'name': self.collection_name,
            'count': self.index.ntotal if self.index else 0,
            'persist_dir': self.persist_dir,
            'embedding_dim': self.embedding_dim
        }

    def clear_collection(self):
        """Clear all documents from the collection"""
        self._create_new_index()
        self._save_index()
        print(f"Cleared collection: {self.collection_name}")


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv("config/.env")
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Please set GEMINI_API_KEY in config/.env")
    else:
        manager = EmbeddingsManager(api_key)
        stats = manager.get_collection_stats()
        print(f"Collection stats: {stats}")

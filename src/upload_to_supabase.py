"""
Upload FAISS embeddings to Supabase
Run this script to migrate local FAISS data to Supabase cloud database.

Usage:
    python src/upload_to_supabase.py
"""

import os
import sys
import json
import pickle
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv("config/.env")

# Supabase client
from supabase import create_client, Client

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # Use service key for insert
FAISS_DB_PATH = "./data/faiss_db"
CHUNKED_DATA_PATH = "./data/breslov_chunked.json"
TABLE_NAME = "breslov_documents"
BATCH_SIZE = 50  # Insert in batches to avoid timeouts


def load_faiss_metadata(collection_name: str = "breslov_chunked") -> tuple:
    """Load FAISS index and metadata"""
    import faiss
    import numpy as np

    index_path = os.path.join(FAISS_DB_PATH, f"{collection_name}.index")
    metadata_path = os.path.join(FAISS_DB_PATH, f"{collection_name}_metadata.pkl")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    # Load index
    index = faiss.read_index(index_path)
    print(f"Loaded FAISS index with {index.ntotal} vectors")

    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    documents = metadata.get('documents', [])
    metadatas = metadata.get('metadatas', [])

    print(f"Loaded {len(documents)} documents and {len(metadatas)} metadata entries")

    # Extract embeddings from FAISS
    embeddings = []
    for i in range(index.ntotal):
        vec = index.reconstruct(i)
        embeddings.append(vec.tolist())

    return documents, metadatas, embeddings


def prepare_records(documents: List[str], metadatas: List[Dict], embeddings: List[List[float]]) -> List[Dict]:
    """Prepare records for Supabase insert"""
    records = []

    for i, (doc, meta, emb) in enumerate(zip(documents, metadatas, embeddings)):
        record = {
            "title": meta.get("title", "Unknown"),
            "ref": meta.get("ref", f"doc_{i}"),
            "chunk_id": meta.get("chunk_id", f"chunk_{i}"),
            "hebrew": meta.get("hebrew", ""),
            "english": meta.get("english", ""),
            "combined": doc,
            "embedding": emb,
            "chunk_index": meta.get("chunk_index", 0),
            "total_chunks": meta.get("total_chunks", 1),
        }
        records.append(record)

    return records


def upload_to_supabase(records: List[Dict], supabase: Client):
    """Upload records to Supabase in batches"""
    total = len(records)
    uploaded = 0
    errors = 0

    print(f"\nUploading {total} records to Supabase...")

    for i in range(0, total, BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        try:
            result = supabase.table(TABLE_NAME).upsert(
                batch,
                on_conflict="chunk_id"  # Update if chunk_id already exists
            ).execute()

            uploaded += len(batch)
            progress = (uploaded / total) * 100
            print(f"Progress: {uploaded}/{total} ({progress:.1f}%)")

        except Exception as e:
            errors += len(batch)
            print(f"Error uploading batch {i}-{i+len(batch)}: {e}")

    return uploaded, errors


def main():
    print("=" * 60)
    print("GUEZI RAG - Upload to Supabase")
    print("=" * 60)

    # Check environment
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: Missing Supabase credentials in config/.env")
        print("Required: SUPABASE_URL, SUPABASE_SERVICE_KEY")
        sys.exit(1)

    print(f"\nSupabase URL: {SUPABASE_URL}")
    print(f"Table: {TABLE_NAME}")

    # Initialize Supabase client
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Supabase client initialized")
    except Exception as e:
        print(f"ERROR: Failed to connect to Supabase: {e}")
        sys.exit(1)

    # Load FAISS data
    print("\nLoading FAISS data...")
    try:
        documents, metadatas, embeddings = load_faiss_metadata("breslov_chunked")
    except Exception as e:
        print(f"ERROR: Failed to load FAISS data: {e}")
        sys.exit(1)

    # Prepare records
    print("\nPreparing records...")
    records = prepare_records(documents, metadatas, embeddings)
    print(f"Prepared {len(records)} records")

    # Check first record
    if records:
        sample = records[0]
        print(f"\nSample record:")
        print(f"  Title: {sample['title']}")
        print(f"  Ref: {sample['ref']}")
        print(f"  Chunk ID: {sample['chunk_id']}")
        print(f"  Embedding dimensions: {len(sample['embedding'])}")

    # Confirm upload
    response = input(f"\nUpload {len(records)} records to Supabase? (y/n): ")
    if response.lower() != 'y':
        print("Upload cancelled.")
        sys.exit(0)

    # Upload
    uploaded, errors = upload_to_supabase(records, supabase)

    print("\n" + "=" * 60)
    print("Upload Complete!")
    print(f"  Uploaded: {uploaded}")
    print(f"  Errors: {errors}")
    print("=" * 60)

    # Verify
    try:
        count = supabase.table(TABLE_NAME).select("id", count="exact").execute()
        print(f"\nTotal records in Supabase: {count.count}")
    except Exception as e:
        print(f"Could not verify count: {e}")


if __name__ == "__main__":
    main()

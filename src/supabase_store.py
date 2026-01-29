"""
Supabase Vector Store for GUEZI Chatbot
Uses pgvector for semantic search
"""

import os
import json
from typing import List, Dict, Optional
from supabase import create_client, Client
from google import genai
from tqdm import tqdm
import time


class SupabaseVectorStore:
    """Vector store using Supabase with pgvector"""

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        gemini_api_key: str,
        table_name: str = "breslov_documents"
    ):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.table_name = table_name

        # Initialize Supabase client
        self.supabase: Client = create_client(supabase_url, supabase_key)

        # Initialize Gemini client for embeddings
        self.gemini = genai.Client(api_key=gemini_api_key)

        self.embedding_dim = 3072  # gemini-embedding-001

    def create_table(self):
        """Create the documents table with vector extension"""
        # This SQL should be run in Supabase SQL Editor
        sql = f"""
        -- Enable pgvector extension
        CREATE EXTENSION IF NOT EXISTS vector;

        -- Create documents table
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            ref TEXT NOT NULL UNIQUE,
            hebrew TEXT,
            english TEXT,
            combined TEXT,
            embedding vector({self.embedding_dim}),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- Create index for vector similarity search
        CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
        ON {self.table_name}
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);

        -- Create function for similarity search
        CREATE OR REPLACE FUNCTION match_{self.table_name}(
            query_embedding vector({self.embedding_dim}),
            match_threshold float,
            match_count int
        )
        RETURNS TABLE (
            id int,
            title text,
            ref text,
            hebrew text,
            english text,
            similarity float
        )
        LANGUAGE sql STABLE
        AS $$
            SELECT
                id,
                title,
                ref,
                hebrew,
                english,
                1 - (embedding <=> query_embedding) AS similarity
            FROM {self.table_name}
            WHERE 1 - (embedding <=> query_embedding) > match_threshold
            ORDER BY embedding <=> query_embedding
            LIMIT match_count;
        $$;
        """
        print("Run this SQL in Supabase SQL Editor:")
        print(sql)
        return sql

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Gemini"""
        try:
            result = self.gemini.models.embed_content(
                model="gemini-embedding-001",
                contents=text[:8000]
            )
            return result.embeddings[0].values
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return []

    def add_documents(self, documents: List[Dict], batch_size: int = 50):
        """Add documents to Supabase with embeddings"""
        print(f"Adding {len(documents)} documents to Supabase...")

        for i in tqdm(range(0, len(documents), batch_size)):
            batch = documents[i:i + batch_size]
            records = []

            for doc in batch:
                text = doc.get('combined', '') or doc.get('english', '') or doc.get('hebrew', '')
                if not text:
                    continue

                # Get embedding
                embedding = self.get_embedding(text)
                if not embedding:
                    continue

                records.append({
                    'title': doc.get('title', ''),
                    'ref': doc.get('ref', ''),
                    'hebrew': doc.get('hebrew', '')[:10000],
                    'english': doc.get('english', '')[:10000],
                    'combined': text[:15000],
                    'embedding': embedding
                })

            if records:
                try:
                    # Upsert to handle duplicates
                    self.supabase.table(self.table_name).upsert(
                        records,
                        on_conflict='ref'
                    ).execute()
                except Exception as e:
                    print(f"Error inserting batch: {e}")

            time.sleep(1)  # Rate limiting

        print("Done adding documents!")

    def search(
        self,
        query: str,
        match_threshold: float = 0.5,
        match_count: int = 5
    ) -> List[Dict]:
        """Search for similar documents"""
        # Get query embedding
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []

        try:
            # Call the match function
            result = self.supabase.rpc(
                f'match_{self.table_name}',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': match_threshold,
                    'match_count': match_count
                }
            ).execute()

            return result.data if result.data else []
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def get_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            result = self.supabase.table(self.table_name).select(
                'id', count='exact'
            ).execute()
            return {
                'table': self.table_name,
                'count': result.count if result.count else 0
            }
        except Exception as e:
            return {'table': self.table_name, 'count': 0, 'error': str(e)}

    def clear_all(self):
        """Delete all documents"""
        try:
            self.supabase.table(self.table_name).delete().neq('id', 0).execute()
            print("Cleared all documents")
        except Exception as e:
            print(f"Error clearing: {e}")


# SQL to create the table (run in Supabase SQL Editor)
SETUP_SQL = """
-- Enable pgvector extension (run once)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table
CREATE TABLE IF NOT EXISTS breslov_documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    ref TEXT NOT NULL UNIQUE,
    hebrew TEXT,
    english TEXT,
    combined TEXT,
    embedding vector(3072),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for fast similarity search
CREATE INDEX IF NOT EXISTS breslov_documents_embedding_idx
ON breslov_documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create similarity search function
CREATE OR REPLACE FUNCTION match_breslov_documents(
    query_embedding vector(3072),
    match_threshold float DEFAULT 0.5,
    match_count int DEFAULT 5
)
RETURNS TABLE (
    id int,
    title text,
    ref text,
    hebrew text,
    english text,
    similarity float
)
LANGUAGE sql STABLE
AS $$
    SELECT
        id,
        title,
        ref,
        hebrew,
        english,
        1 - (embedding <=> query_embedding) AS similarity
    FROM breslov_documents
    WHERE 1 - (embedding <=> query_embedding) > match_threshold
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;
"""


if __name__ == "__main__":
    print("Supabase Setup SQL:")
    print("=" * 60)
    print(SETUP_SQL)

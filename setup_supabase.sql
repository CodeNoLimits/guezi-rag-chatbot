-- GUEZI RAG Chatbot - Supabase Setup SQL
-- Run this in Supabase SQL Editor after creating a new project

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create documents table
CREATE TABLE IF NOT EXISTS breslov_documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    ref TEXT NOT NULL,
    chunk_id TEXT UNIQUE,
    hebrew TEXT,
    english TEXT,
    combined TEXT,
    embedding vector(3072),
    chunk_index INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Create index for vector similarity search
CREATE INDEX IF NOT EXISTS breslov_documents_embedding_idx
ON breslov_documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- 4. Create text search index
CREATE INDEX IF NOT EXISTS breslov_documents_ref_idx
ON breslov_documents(ref);

CREATE INDEX IF NOT EXISTS breslov_documents_title_idx
ON breslov_documents(title);

-- 5. Create similarity search function
CREATE OR REPLACE FUNCTION match_breslov_documents(
    query_embedding vector(3072),
    match_threshold float DEFAULT 0.5,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id int,
    title text,
    ref text,
    chunk_id text,
    hebrew text,
    english text,
    combined text,
    similarity float
)
LANGUAGE sql STABLE
AS $$
    SELECT
        id,
        title,
        ref,
        chunk_id,
        hebrew,
        english,
        combined,
        1 - (embedding <=> query_embedding) AS similarity
    FROM breslov_documents
    WHERE 1 - (embedding <=> query_embedding) > match_threshold
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;

-- 6. Create hybrid search function (reference + semantic)
CREATE OR REPLACE FUNCTION hybrid_search_breslov(
    search_ref text,
    query_embedding vector(3072),
    match_threshold float DEFAULT 0.4,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id int,
    title text,
    ref text,
    combined text,
    similarity float,
    match_type text
)
LANGUAGE sql STABLE
AS $$
    -- First: exact reference matches
    SELECT id, title, ref, combined, 1.0::float as similarity, 'exact_reference' as match_type
    FROM breslov_documents
    WHERE LOWER(ref) = LOWER(search_ref)

    UNION ALL

    -- Then: semantic matches (excluding already found refs)
    SELECT id, title, ref, combined,
           1 - (embedding <=> query_embedding) as similarity,
           'semantic' as match_type
    FROM breslov_documents
    WHERE LOWER(ref) != LOWER(search_ref)
      AND 1 - (embedding <=> query_embedding) > match_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
$$;

-- Done!
-- After running this, update your config/.env with your Supabase URL and keys

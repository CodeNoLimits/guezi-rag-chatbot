"""
Semantic Chunker for Breslov Texts
Implements intelligent chunking with overlap for better RAG performance
"""

import re
from typing import List, Dict, Tuple
import json


class SemanticChunker:
    """
    Intelligent chunking for Jewish texts with:
    - Semantic boundaries (paragraphs, sections)
    - Overlap for context preservation
    - Optimal chunk sizes for embeddings
    """

    def __init__(
        self,
        target_chunk_size: int = 1000,  # Target chars per chunk
        min_chunk_size: int = 200,       # Don't create chunks smaller than this
        max_chunk_size: int = 2000,      # Split chunks larger than this
        overlap_size: int = 150          # Overlap between chunks (15%)
    ):
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling Hebrew and English"""
        # Split on sentence boundaries
        # Hebrew: ends with period, colon, or special marks
        # Also handle paragraph breaks
        pattern = r'(?<=[.!?:。])\s+|(?<=\n\n)'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split on double newlines or significant breaks
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _create_chunks_with_overlap(
        self,
        segments: List[str],
        overlap: bool = True
    ) -> List[str]:
        """Create chunks from segments with overlap"""
        if not segments:
            return []

        chunks = []
        current_chunk = []
        current_length = 0

        for segment in segments:
            segment_length = len(segment)

            # If adding this segment exceeds max size
            if current_length + segment_length > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)

                # Start new chunk with overlap
                if overlap and len(current_chunk) > 1:
                    # Keep last portion for overlap
                    overlap_text = ' '.join(current_chunk[-2:])
                    if len(overlap_text) <= self.overlap_size * 2:
                        current_chunk = current_chunk[-2:]
                        current_length = len(overlap_text)
                    else:
                        current_chunk = [current_chunk[-1]]
                        current_length = len(current_chunk[0])
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(segment)
            current_length += segment_length + 1  # +1 for space

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
            elif chunks:
                # Merge with previous if too small
                chunks[-1] = chunks[-1] + ' ' + chunk_text

        return chunks

    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        Chunk a single document into smaller pieces

        Args:
            document: Dict with 'title', 'ref', 'hebrew', 'english', 'combined'

        Returns:
            List of chunk dicts with parent reference
        """
        chunks = []

        title = document.get('title', '')
        ref = document.get('ref', '')
        hebrew = document.get('hebrew', '') or ''
        english = document.get('english', '') or ''

        # ALWAYS combine Hebrew + English for bilingual search
        # Include reference for better semantic matching
        combined_parts = []

        # Add reference/title for searchability
        combined_parts.append(f"[{title} - {ref}]")

        if english:
            combined_parts.append(english)
        if hebrew:
            combined_parts.append(hebrew)

        # Fallback to original combined if no separate content
        if len(combined_parts) == 1:
            original_combined = document.get('combined', '')
            if original_combined:
                combined_parts.append(original_combined)

        combined = '\n\n'.join(combined_parts)

        # Clean HTML artifacts
        combined = re.sub(r'<[^>]+>', ' ', combined)
        combined = re.sub(r'\s+', ' ', combined).strip()

        # If document is small enough, keep as is
        if len(combined) <= self.max_chunk_size:
            return [{
                'title': title,
                'ref': ref,
                'chunk_id': f"{ref}_0",
                'hebrew': hebrew[:4000],
                'english': english[:4000],
                'combined': combined,
                'chunk_index': 0,
                'total_chunks': 1
            }]

        # Split into paragraphs first
        paragraphs = self._split_into_paragraphs(combined)

        # If paragraphs are still too long, split into sentences
        segments = []
        for para in paragraphs:
            if len(para) > self.max_chunk_size:
                sentences = self._split_into_sentences(para)
                segments.extend(sentences)
            else:
                segments.append(para)

        # Create chunks with overlap
        text_chunks = self._create_chunks_with_overlap(segments)

        # Create chunk documents
        for i, chunk_text in enumerate(text_chunks):
            chunks.append({
                'title': title,
                'ref': ref,
                'chunk_id': f"{ref}_chunk_{i}",
                'hebrew': '',  # We don't have aligned Hebrew for chunks
                'english': '',
                'combined': chunk_text,
                'chunk_index': i,
                'total_chunks': len(text_chunks),
                'parent_ref': ref
            })

        return chunks

    def chunk_corpus(self, corpus: List[Dict]) -> List[Dict]:
        """
        Chunk entire corpus

        Args:
            corpus: List of documents

        Returns:
            List of chunked documents
        """
        chunked_corpus = []

        for doc in corpus:
            chunks = self.chunk_document(doc)
            chunked_corpus.extend(chunks)

        return chunked_corpus


class HybridRetriever:
    """
    Hybrid retrieval combining:
    - Vector similarity search
    - Keyword matching (BM25-style)
    - Metadata filtering
    """

    def __init__(self, embeddings_manager):
        self.embeddings = embeddings_manager

    def search(
        self,
        query: str,
        n_results: int = 10,
        min_score: float = 0.4,
        book_filter: str = None
    ) -> List[Dict]:
        """
        Enhanced search with multiple strategies

        Args:
            query: Search query
            n_results: Number of results
            min_score: Minimum relevance score
            book_filter: Optional filter by book title

        Returns:
            Ranked list of relevant documents
        """
        # Get more results for reranking
        initial_results = self.embeddings.search(query, n_results=n_results * 2)

        # Filter by score
        filtered = [
            r for r in initial_results
            if r.get('relevance_score', 0) >= min_score
        ]

        # Apply book filter if specified
        if book_filter:
            filtered = [
                r for r in filtered
                if book_filter.lower() in r['metadata'].get('title', '').lower()
            ]

        # Keyword boost: boost results that contain query keywords
        query_words = set(query.lower().split())
        for result in filtered:
            text = result.get('text', '').lower()
            keyword_matches = sum(1 for word in query_words if word in text)
            # Boost score based on keyword matches
            result['relevance_score'] *= (1 + 0.1 * keyword_matches)

        # Sort by adjusted score
        filtered.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

        return filtered[:n_results]


def process_corpus_with_chunking(input_file: str, output_file: str):
    """Process corpus with semantic chunking"""
    print("Loading corpus...")
    with open(input_file, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    print(f"Original documents: {len(corpus)}")

    chunker = SemanticChunker(
        target_chunk_size=1000,
        max_chunk_size=1500,
        overlap_size=150
    )

    print("Chunking corpus...")
    chunked = chunker.chunk_corpus(corpus)

    print(f"After chunking: {len(chunked)} chunks")

    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunked, f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_file}")

    # Statistics
    lengths = [len(c.get('combined', '')) for c in chunked]
    print(f"\nChunk statistics:")
    print(f"  Average length: {sum(lengths)/len(lengths):.0f} chars")
    print(f"  Min: {min(lengths)}, Max: {max(lengths)}")

    return chunked


if __name__ == "__main__":
    process_corpus_with_chunking(
        'data/breslov_complete.json',
        'data/breslov_chunked.json'
    )

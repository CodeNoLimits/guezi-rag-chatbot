"""
Fetch Hebrew-only texts from Sefaria
Including Likutei Halachot and other Breslov texts not available in English
"""

import os
import json
import time
import requests
from typing import List, Dict
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv("config/.env")


class HebrewTextFetcher:
    """Fetch Hebrew-only Breslov texts from Sefaria"""

    BASE_URL = "https://www.sefaria.org/api"

    # Hebrew-only Breslov texts
    HEBREW_ONLY_TEXTS = [
        # Likutei Halachot - Rabbi Natan's halachic work based on Likutei Moharan
        "Likutey_Halakhot",
        "Likutey_Halakhot,_Orach_Chaim",
        "Likutey_Halakhot,_Yoreh_Deah",
        "Likutey_Halakhot,_Even_HaEzer",
        "Likutey_Halakhot,_Choshen_Mishpat",

        # Other Hebrew texts
        "Alim_LeTerufah",  # Letters of Rabbi Nachman
        "Parparaot_LeChokhmah",  # Commentary on Likutei Moharan
    ]

    # Additional Breslov-related texts
    ADDITIONAL_TEXTS = [
        "Otzar_HaYirah",  # Treasury of Fear/Awe
        "Avanehah_Barzel",  # Teachings by R' Natan
    ]

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GUEZI-RAG-Chatbot/2.0-Hebrew'
        })

    def get_text(self, ref: str) -> Dict:
        """Fetch text by reference"""
        url = f"{self.BASE_URL}/texts/{ref}"
        try:
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching {ref}: {e}")
            return {}

    def get_index(self, title: str) -> Dict:
        """Get book index/structure"""
        url = f"{self.BASE_URL}/v2/index/{title}"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching index for {title}: {e}")
            return {}

    def get_table_of_contents(self, title: str) -> List[str]:
        """Get all section references for a book"""
        url = f"{self.BASE_URL}/index/{title}"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            refs = []
            if 'schema' in data:
                refs = self._extract_section_refs(title, data['schema'])
            return refs
        except Exception as e:
            print(f"Error getting TOC for {title}: {e}")
            return [title]  # Fall back to just the title

    def _extract_section_refs(self, base_title: str, schema: Dict, prefix: str = "") -> List[str]:
        """Extract section references from schema"""
        refs = []

        if schema.get('nodeType') == 'JaggedArrayNode':
            ref = prefix if prefix else base_title
            refs.append(ref)
        elif 'nodes' in schema:
            for node in schema['nodes']:
                titles = node.get('titles', [])
                # Get English title if available, otherwise Hebrew
                node_title = None
                for t in titles:
                    if t.get('lang') == 'en':
                        node_title = t.get('text')
                        break
                if not node_title:
                    for t in titles:
                        if t.get('lang') == 'he':
                            node_title = t.get('text')
                            break

                if node_title:
                    new_prefix = f"{prefix}, {node_title}" if prefix else f"{base_title}, {node_title}"
                    refs.extend(self._extract_section_refs(base_title, node, new_prefix))

        return refs

    def process_text(self, title: str, data: Dict) -> List[Dict]:
        """Process fetched text into document chunks"""
        documents = []

        hebrew = data.get('he', [])
        english = data.get('text', [])  # May be empty for Hebrew-only
        ref = data.get('ref', title)
        he_title = data.get('heTitle', title)

        docs = self._flatten_hebrew_text(title, he_title, ref, hebrew, english)
        documents.extend(docs)

        return documents

    def _flatten_hebrew_text(self, title: str, he_title: str, ref: str,
                             hebrew, english, depth: int = 0) -> List[Dict]:
        """Flatten nested Hebrew text into documents"""
        documents = []

        # Handle string (base case)
        if isinstance(hebrew, str):
            hebrew_text = self._clean_text(hebrew)
            english_text = self._clean_text(english) if isinstance(english, str) else ""

            if hebrew_text:
                documents.append({
                    'title': title,
                    'he_title': he_title,
                    'ref': ref,
                    'hebrew': hebrew_text,
                    'english': english_text,
                    'combined': f"{hebrew_text}\n\n{english_text}".strip() if english_text else hebrew_text,
                    'language': 'hebrew' if not english_text else 'bilingual'
                })

        # Handle list (recursive case)
        elif isinstance(hebrew, list):
            for i, he_item in enumerate(hebrew, 1):
                en_item = english[i-1] if isinstance(english, list) and i <= len(english) else ""
                sub_ref = f"{ref}:{i}" if depth == 0 else f"{ref}.{i}"
                sub_docs = self._flatten_hebrew_text(title, he_title, sub_ref, he_item, en_item, depth + 1)
                documents.extend(sub_docs)

        return documents

    def _clean_text(self, text) -> str:
        """Clean HTML tags and whitespace from text"""
        if not text or not isinstance(text, str):
            return ""

        import re
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Clean whitespace
        text = ' '.join(text.split())
        return text.strip()

    def fetch_all_hebrew_texts(self, save_path: str = "data/hebrew_corpus.json") -> List[Dict]:
        """Fetch all Hebrew-only texts"""
        corpus = []
        all_texts = self.HEBREW_ONLY_TEXTS + self.ADDITIONAL_TEXTS

        print(f"Fetching {len(all_texts)} Hebrew text collections from Sefaria...")

        for text_title in tqdm(all_texts, desc="Fetching texts"):
            print(f"\n📖 Fetching: {text_title}")

            # Get section refs
            section_refs = self.get_table_of_contents(text_title)
            print(f"   Found {len(section_refs)} sections")

            for section_ref in tqdm(section_refs, desc=f"  {text_title}", leave=False):
                text_data = self.get_text(section_ref)
                if text_data:
                    documents = self.process_text(text_title, text_data)
                    corpus.extend(documents)
                    print(f"   ✓ {section_ref}: {len(documents)} passages")

                time.sleep(0.3)  # Rate limiting

        # Save corpus
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Saved {len(corpus)} Hebrew documents to {save_path}")
        return corpus

    def chunk_documents(self, documents: List[Dict], max_chunk_size: int = 1500) -> List[Dict]:
        """Chunk long documents for better embedding"""
        chunked = []

        for doc in documents:
            text = doc.get('combined', doc.get('hebrew', ''))
            if len(text) <= max_chunk_size:
                doc['chunk_id'] = f"{doc['ref']}_0"
                doc['chunk_index'] = 0
                doc['total_chunks'] = 1
                chunked.append(doc)
            else:
                # Split into chunks
                chunks = self._split_text(text, max_chunk_size)
                for i, chunk in enumerate(chunks):
                    chunk_doc = doc.copy()
                    chunk_doc['combined'] = chunk
                    chunk_doc['chunk_id'] = f"{doc['ref']}_{i}"
                    chunk_doc['chunk_index'] = i
                    chunk_doc['total_chunks'] = len(chunks)
                    chunked.append(chunk_doc)

        return chunked

    def _split_text(self, text: str, max_size: int) -> List[str]:
        """Split text into chunks at sentence boundaries"""
        import re

        # Split on sentence endings (Hebrew and English)
        sentences = re.split(r'(?<=[.!?:׃])\s+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text[:max_size]]


def upload_to_supabase(documents: List[Dict]):
    """Upload Hebrew documents to Supabase"""
    from supabase import create_client
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")

    if not all([api_key, supabase_url, supabase_key]):
        print("❌ Missing environment variables")
        return

    # Initialize clients
    gemini_client = genai.Client(api_key=api_key)
    supabase = create_client(supabase_url, supabase_key)

    print(f"\n📤 Uploading {len(documents)} documents to Supabase...")

    batch_size = 10
    success_count = 0
    error_count = 0

    for i in tqdm(range(0, len(documents), batch_size), desc="Uploading"):
        batch = documents[i:i + batch_size]
        records = []

        for doc in batch:
            try:
                # Generate embedding
                text_for_embedding = doc.get('combined', doc.get('hebrew', ''))[:8000]
                result = gemini_client.models.embed_content(
                    model="models/text-embedding-004",
                    contents=text_for_embedding
                )
                embedding = result.embeddings[0].values

                record = {
                    'title': doc.get('title', ''),
                    'ref': doc.get('ref', ''),
                    'chunk_id': doc.get('chunk_id', f"hebrew_{i}"),
                    'hebrew': doc.get('hebrew', '')[:10000],
                    'english': doc.get('english', '')[:10000],
                    'combined': text_for_embedding,
                    'embedding': embedding,
                    'chunk_index': doc.get('chunk_index', 0),
                    'total_chunks': doc.get('total_chunks', 1),
                }
                records.append(record)

            except Exception as e:
                print(f"Error processing {doc.get('ref', 'unknown')}: {e}")
                error_count += 1

        # Upload batch
        if records:
            try:
                supabase.table("breslov_documents").upsert(
                    records,
                    on_conflict="chunk_id"
                ).execute()
                success_count += len(records)
            except Exception as e:
                print(f"Upload error: {e}")
                error_count += len(records)

        time.sleep(1)  # Rate limiting

    print(f"\n✅ Upload complete: {success_count} success, {error_count} errors")


if __name__ == "__main__":
    fetcher = HebrewTextFetcher()

    # Fetch Hebrew texts
    print("=" * 60)
    print("GUEZI - Hebrew Text Fetcher")
    print("=" * 60)

    corpus = fetcher.fetch_all_hebrew_texts("data/hebrew_corpus.json")

    if corpus:
        # Chunk documents
        print("\n📄 Chunking documents...")
        chunked = fetcher.chunk_documents(corpus)
        print(f"   Created {len(chunked)} chunks from {len(corpus)} documents")

        # Save chunked version
        with open("data/hebrew_corpus_chunked.json", 'w', encoding='utf-8') as f:
            json.dump(chunked, f, ensure_ascii=False, indent=2)

        # Upload to Supabase
        print("\n" + "=" * 60)
        response = input("Upload to Supabase? (y/n): ")
        if response.lower() == 'y':
            upload_to_supabase(chunked)

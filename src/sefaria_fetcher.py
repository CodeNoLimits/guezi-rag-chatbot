"""
Sefaria API Fetcher
Fetches Rabbi Nachman of Breslov texts from Sefaria.org
"""

import requests
import json
from typing import List, Dict, Optional
from tqdm import tqdm
import time

class SefariaFetcher:
    """Fetch Jewish texts from Sefaria API"""

    BASE_URL = "https://www.sefaria.org/api"

    # Rabbi Nachman of Breslov primary texts
    BRESLOV_TEXTS = [
        "Likutey_Moharan",
        "Likutey_Moharan_Tinyana",  # Part 2
        "Sippurei_Maasiyot",         # Stories
        "Tikkun_HaKlali",            # General Remedy - 10 Psalms
        "Likutey_Tefilot",           # Collected Prayers
        "Sefer_HaMiddot",            # Book of Traits
        "Meshivat_Nefesh",           # Restoring the Soul
        "Hishtapkhut_HaNefesh",      # Outpouring of the Soul
        "Kitzur_Likutey_Moharan",    # Abridged Likutey Moharan
    ]

    # Additional related texts
    RELATED_TEXTS = [
        "Shivchei_HaRan",            # Praises of Rabbi Nachman
        "Sichot_HaRan",              # Conversations of Rabbi Nachman
        "Chayei_Moharan",            # Life of Rabbi Nachman
    ]

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GUEZI-RAG-Chatbot/1.0'
        })

    def get_text(self, ref: str, with_commentary: bool = False) -> Dict:
        """
        Fetch a specific text by reference

        Args:
            ref: Sefaria reference (e.g., "Likutey_Moharan.1.1")
            with_commentary: Include commentaries

        Returns:
            Dict with Hebrew and English text
        """
        url = f"{self.BASE_URL}/texts/{ref}"
        params = {}
        if with_commentary:
            params['commentary'] = 1

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching {ref}: {e}")
            return {}

    def get_index(self, title: str) -> Dict:
        """Get the index/structure of a book"""
        url = f"{self.BASE_URL}/v2/index/{title}"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching index for {title}: {e}")
            return {}

    def get_all_refs_for_book(self, title: str) -> List[str]:
        """Get all references for a given book"""
        index = self.get_index(title)
        if not index:
            return []

        # Try to get the schema and extract refs
        refs = []
        if 'schema' in index:
            refs = self._extract_refs_from_schema(title, index['schema'])
        return refs

    def _extract_refs_from_schema(self, title: str, schema: Dict, prefix: str = "") -> List[str]:
        """Recursively extract refs from schema"""
        refs = []

        if schema.get('nodeType') == 'JaggedArrayNode':
            # This is a text node, generate refs based on depth
            depth = schema.get('depth', 1)
            # For now, return the base ref - we'll fetch and parse sections
            refs.append(f"{prefix}{title}" if prefix else title)

        elif 'nodes' in schema:
            for node in schema['nodes']:
                node_title = node.get('titles', [{}])[0].get('text', '')
                if node_title:
                    new_prefix = f"{prefix}{node_title}, " if prefix else f"{title}, "
                    refs.extend(self._extract_refs_from_schema("", node, new_prefix))

        return refs

    def search_texts(self, query: str, filters: Optional[List[str]] = None, size: int = 20) -> Dict:
        """
        Search Sefaria texts

        Args:
            query: Search query
            filters: Category filters (e.g., ["Chasidut", "Breslov"])
            size: Number of results

        Returns:
            Search results
        """
        url = f"{self.BASE_URL}/search-wrapper"
        params = {
            'query': query,
            'size': size,
            'type': 'text'
        }
        if filters:
            params['filters'] = '|'.join(filters)

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error searching for '{query}': {e}")
            return {}

    def fetch_breslov_corpus(self, save_path: str = "data/breslov_corpus.json") -> List[Dict]:
        """
        Fetch all Breslov texts and save to file

        Returns:
            List of text documents
        """
        corpus = []
        all_texts = self.BRESLOV_TEXTS + self.RELATED_TEXTS

        print("Fetching Breslov corpus from Sefaria...")
        for text_title in tqdm(all_texts):
            print(f"\nFetching: {text_title}")

            # Get the full text
            text_data = self.get_text(text_title)
            if not text_data:
                continue

            # Process the text
            documents = self._process_text_data(text_title, text_data)
            corpus.extend(documents)

            # Be nice to the API
            time.sleep(0.5)

        # Save corpus
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)

        print(f"\nSaved {len(corpus)} documents to {save_path}")
        return corpus

    def _process_text_data(self, title: str, data: Dict) -> List[Dict]:
        """Process raw text data into documents"""
        documents = []

        hebrew = data.get('he', [])
        english = data.get('text', [])
        ref = data.get('ref', title)

        # Handle nested structures
        docs = self._flatten_text(title, ref, hebrew, english)
        documents.extend(docs)

        return documents

    def _flatten_text(self, title: str, ref: str, hebrew, english, depth: int = 0) -> List[Dict]:
        """Flatten nested text structures into documents"""
        documents = []

        if isinstance(hebrew, str) and isinstance(english, str):
            # Base case: single text segment
            if hebrew.strip() or english.strip():
                documents.append({
                    'title': title,
                    'ref': ref,
                    'hebrew': hebrew.strip(),
                    'english': english.strip(),
                    'combined': f"{hebrew}\n\n{english}".strip()
                })
        elif isinstance(hebrew, list) and isinstance(english, list):
            # Recursive case: nested structure
            for i, (he_item, en_item) in enumerate(zip(hebrew, english), 1):
                sub_ref = f"{ref}:{i}" if depth == 0 else f"{ref}.{i}"
                sub_docs = self._flatten_text(title, sub_ref, he_item, en_item, depth + 1)
                documents.extend(sub_docs)
        elif isinstance(hebrew, list):
            # Only Hebrew available
            for i, he_item in enumerate(hebrew, 1):
                sub_ref = f"{ref}:{i}" if depth == 0 else f"{ref}.{i}"
                if isinstance(he_item, str) and he_item.strip():
                    documents.append({
                        'title': title,
                        'ref': sub_ref,
                        'hebrew': he_item.strip(),
                        'english': '',
                        'combined': he_item.strip()
                    })
                elif isinstance(he_item, list):
                    sub_docs = self._flatten_text(title, sub_ref, he_item, [], depth + 1)
                    documents.extend(sub_docs)

        return documents

    def get_topic_texts(self, topic: str = "rabbi-nachman-of-breslov") -> List[Dict]:
        """Get texts related to a specific topic"""
        url = f"{self.BASE_URL}/topics/{topic}"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching topic {topic}: {e}")
            return {}


if __name__ == "__main__":
    # Test the fetcher
    fetcher = SefariaFetcher()

    # Test single text fetch
    print("Testing single text fetch...")
    text = fetcher.get_text("Likutey_Moharan.1.1")
    if text:
        print(f"Title: {text.get('title', 'N/A')}")
        print(f"Hebrew: {str(text.get('he', ''))[:200]}...")
        print(f"English: {str(text.get('text', ''))[:200]}...")

    # Test search
    print("\nTesting search...")
    results = fetcher.search_texts("hitbodedut", filters=["Chasidut"])
    if results:
        print(f"Found {len(results.get('hits', {}).get('hits', []))} results")

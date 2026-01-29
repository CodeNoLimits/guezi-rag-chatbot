"""
Import Hebrew Breslov books from local files to Supabase
Processes .docx, .doc, and .rtf files
"""

import os
import re
import json
import time
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv("config/.env")


# Book name mappings (Hebrew filename -> English title for ref)
BOOK_MAPPINGS = {
    "ליקוטי מוהרן קמא": "Likutei Moharan Hebrew",
    "ליקוטי מוהרן תנינא": "Likutei Moharan Part II Hebrew",
    "ליקוטי תפילות": "Likutei Tefilot Hebrew",
    "ליקוטי עצות": "Likutei Etzot",
    "סיפורי מעשיות": "Sippurei Maasiyot Hebrew",
    "ספר המידות": "Sefer HaMidot",
    "חיי מוהרן": "Chayei Moharan Hebrew",
    "שבחי ושיחות הרן": "Shivchei v'Sichot HaRan",
    "השתפכות הנפש ומשיבת נפש": "Hishtapkhut HaNefesh",
    "עלים לתרופה": "Alim LeTerufah",
    "ימי מוהרנת": "Yemei Moharnat",
    "קיצור ליקוטי מוהרן": "Kitzur Likutei Moharan",
    "קיצור ליקוטי מוהרן תנינא": "Kitzur Likutei Moharan Part II",
    "שמות הצדיקים": "Shemot HaTzadikim",
    "AVANEA": "Avaneha Barzel",
    "kokhve": "Kokhvei Or",
}


def extract_text_from_docx(filepath: str) -> str:
    """Extract text from .docx file"""
    try:
        from docx import Document
        doc = Document(filepath)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Error reading docx {filepath}: {e}")
        return ""


def extract_text_from_doc(filepath: str) -> str:
    """Extract text from .doc file using textract or antiword"""
    try:
        import subprocess
        result = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", filepath],
            capture_output=True, text=True, timeout=60
        )
        return result.stdout
    except Exception as e:
        print(f"Error reading doc {filepath}: {e}")
        return ""


def extract_text_from_rtf(filepath: str) -> str:
    """Extract text from .rtf file"""
    try:
        import subprocess
        result = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", filepath],
            capture_output=True, text=True, timeout=60
        )
        return result.stdout
    except Exception as e:
        print(f"Error reading rtf {filepath}: {e}")
        return ""


def extract_text(filepath: str) -> str:
    """Extract text from any supported file format"""
    ext = Path(filepath).suffix.lower()

    if ext == ".docx":
        return extract_text_from_docx(filepath)
    elif ext == ".doc":
        return extract_text_from_doc(filepath)
    elif ext == ".rtf":
        return extract_text_from_rtf(filepath)
    else:
        print(f"Unsupported format: {ext}")
        return ""


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    if not text:
        return []

    # Split on paragraph breaks or sentence endings
    paragraphs = re.split(r'\n\s*\n|\n(?=[א-ת])', text)

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += "\n\n" + para if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def get_book_title(filename: str) -> str:
    """Get English title for a Hebrew filename"""
    name = Path(filename).stem
    return BOOK_MAPPINGS.get(name, name)


def process_books(folder_path: str) -> List[Dict]:
    """Process all books in a folder"""
    documents = []
    folder = Path(folder_path)

    files = list(folder.glob("*.docx")) + list(folder.glob("*.doc")) + list(folder.glob("*.rtf"))

    print(f"Found {len(files)} book files")

    for filepath in tqdm(files, desc="Processing books"):
        print(f"\n📖 Processing: {filepath.name}")

        text = extract_text(str(filepath))
        if not text:
            print(f"   ⚠️ No text extracted")
            continue

        title = get_book_title(filepath.name)
        he_title = filepath.stem

        # Chunk the text
        chunks = chunk_text(text)
        print(f"   Created {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            doc = {
                "title": title,
                "he_title": he_title,
                "ref": f"{title} {i+1}",
                "hebrew": chunk,
                "english": "",  # Hebrew only
                "combined": chunk,
                "chunk_id": f"hebrew_{title.replace(' ', '_')}_{i}",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "language": "hebrew"
            }
            documents.append(doc)

    return documents


def upload_to_supabase(documents: List[Dict], batch_size: int = 5):
    """Upload documents to Supabase with embeddings"""
    from supabase import create_client
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")

    if not all([api_key, supabase_url, supabase_key]):
        print("❌ Missing environment variables")
        return

    gemini_client = genai.Client(api_key=api_key)
    supabase = create_client(supabase_url, supabase_key)

    print(f"\n📤 Uploading {len(documents)} documents to Supabase...")

    success_count = 0
    error_count = 0

    for i in tqdm(range(0, len(documents), batch_size), desc="Uploading"):
        batch = documents[i:i + batch_size]
        records = []

        for doc in batch:
            try:
                # Generate embedding using the correct model
                text_for_embedding = doc.get("combined", "")[:8000]
                result = gemini_client.models.embed_content(
                    model="gemini-embedding-001",  # 3072 dimensions
                    contents=text_for_embedding
                )
                embedding = result.embeddings[0].values

                record = {
                    "title": doc.get("title", ""),
                    "ref": doc.get("ref", ""),
                    "chunk_id": doc.get("chunk_id", ""),
                    "hebrew": doc.get("hebrew", "")[:10000],
                    "english": doc.get("english", ""),
                    "combined": text_for_embedding,
                    "embedding": embedding,
                    "chunk_index": doc.get("chunk_index", 0),
                    "total_chunks": doc.get("total_chunks", 1),
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

        time.sleep(1.5)  # Rate limiting for embedding API

    print(f"\n✅ Upload complete: {success_count} success, {error_count} errors")
    return success_count, error_count


if __name__ == "__main__":
    import sys

    # Default folder
    folder = "/Users/codenolimits-dreamai-nanach/Desktop/LIVRES GUEZI"

    if len(sys.argv) > 1:
        folder = sys.argv[1]

    print("=" * 60)
    print("GUEZI - Hebrew Book Importer")
    print("=" * 60)
    print(f"Folder: {folder}")

    # Check if python-docx is installed
    try:
        import docx
    except ImportError:
        print("\n⚠️ Installing python-docx...")
        os.system("pip install python-docx")

    # Process books
    documents = process_books(folder)

    if documents:
        print(f"\n📚 Total documents prepared: {len(documents)}")

        # Save to JSON for backup
        backup_path = "data/hebrew_books_backup.json"
        os.makedirs("data", exist_ok=True)
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        print(f"💾 Backup saved to {backup_path}")

        # Upload
        print("\n" + "=" * 60)
        response = input("Upload to Supabase? (y/n): ")
        if response.lower() == "y":
            upload_to_supabase(documents)
    else:
        print("❌ No documents to upload")

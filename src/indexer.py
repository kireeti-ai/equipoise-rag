"""
indexer.py
Loads SciFact abstracts directly from local corpus.jsonl file.
Builds ChromaDB + BM25 index.
Run once — persists to disk.

Raw AllenAI SciFact corpus.jsonl format:
  doc_id    : int        — unique abstract ID
  title     : str        — paper title
  abstract  : list[str]  — list of sentences (joined into one string here)
  structured: bool       — whether abstract has structured sections

Local file: data/scifact/data/corpus.jsonl
"""

import json
import os
import pickle

import chromadb
import torch
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
    
from src.config import (
    CHROMA_COLLECTION_NAME, CHROMA_DB_PATH, EMBEDDING_MODEL
)

CORPUS_PATH = "data/scifact/data/corpus.jsonl"


def extract_metadata(text):
    text_lower = text.lower()
    
    # Species heuristics
    if any(w in text_lower for w in ["human", "patient", "clinical trial", "men", "women", "children"]):
        species = "human"
    elif any(w in text_lower for w in ["mice", "mouse", "rat", "murine", "c. elegans", "zebrafish", "animal", "macaque"]):
        species = "animal"
    elif any(w in text_lower for w in ["in vitro", "cell line", "cultured"]):
        species = "in vitro"
    else:
        species = "unknown"
        
    # Study Type heuristics
    if "randomized controlled" in text_lower or " rct " in text_lower:
        study_type = "rct"
    elif "meta-analysis" in text_lower or "systematic review" in text_lower:
        study_type = "meta-analysis"
    elif "review" in text_lower:
        study_type = "review"
    elif "case report" in text_lower:
        study_type = "case report"
    else:
        study_type = "unknown"
        
    # Year heuristic (first 4 digit number that looks like a recent year)
    year = "unknown"
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', text)
    if year_match:
        year = year_match.group(1)
        
    return {"species": species, "study_type": study_type, "year": year}

def load_scifact_abstracts(window_size=3, stride=2):
    """
    Load all abstracts from local corpus.jsonl and apply sentence-window chunking.
    Extracts metadata: species, study type, year.
    Returns list of dicts: chunk_id, title, text, source_abstract, full_text, etc.
    """
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(
            f"\nCorpus file not found at: {CORPUS_PATH}\n"
            f"Run this to extract: cd data/scifact && tar -xzf data.tar.gz\n"
        )

    print(f"Loading SciFact corpus from {CORPUS_PATH} with window chunking (size={window_size}, stride={stride})...")

    chunks = []
    with open(CORPUS_PATH) as f:
        for line in f:
            item = json.loads(line.strip())

            doc_id = str(item["doc_id"])
            title = item.get("title") or ""

            sentences = item.get("abstract") or []
            if isinstance(sentences, str):
                sentences = [s.strip() + "." for s in sentences.split('.') if s.strip()]
                
            full_abstract = " ".join(sentences)
            meta = extract_metadata(title + " " + full_abstract)

            # If abstract is shorter than window, just make one chunk
            if len(sentences) <= window_size:
                chunk_text = " ".join(sentences)
                chunks.append({
                    "id": f"{doc_id}_0",
                    "doc_id": doc_id,
                    "title": title,
                    "text": chunk_text,
                    "source_abstract": full_abstract,
                    "chunk_type": "passage",
                    "full_text": (title + " " + chunk_text).strip(),
                    **meta
                })
                continue
                
            # Sliding window chunking
            last_idx = 0
            for i in range(0, len(sentences) - window_size + 1, stride):
                window = sentences[i:i+window_size]
                chunk_text = " ".join(window)
                chunks.append({
                    "id": f"{doc_id}_{i}",
                    "doc_id": doc_id,
                    "title": title,
                    "text": chunk_text,
                    "source_abstract": full_abstract,
                    "chunk_type": "passage",
                    "full_text": (title + " " + chunk_text).strip(),
                    **meta
                })
                last_idx = i

            # Ensure we don't miss the end if stride skipped it
            if last_idx + window_size < len(sentences):
                window = sentences[-window_size:]
                chunk_text = " ".join(window)
                chunks.append({
                    "id": f"{doc_id}_end",
                    "doc_id": doc_id,
                    "title": title,
                    "text": chunk_text,
                    "source_abstract": full_abstract,
                    "chunk_type": "passage",
                    "full_text": (title + " " + chunk_text).strip(),
                    **meta
                })

    print(f"Generated {len(chunks)} chunks from corpus")
    return chunks


def build_chroma_index(abstracts):
    """Build ChromaDB vector index. Deletes existing collection first."""
    print(f"\nBuilding ChromaDB index using {EMBEDDING_MODEL}...")
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    batch_size = 32

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
        print(f"Deleted existing collection: {CHROMA_COLLECTION_NAME}")
    except Exception:
        print("No existing collection found — creating fresh.")

    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    for i in tqdm(range(0, len(abstracts), batch_size), desc="Indexing into ChromaDB"):
        batch = abstracts[i:i + batch_size]
        texts = [a["full_text"] for a in batch]
        ids = [a["id"] for a in batch]
        metadatas = [{
            "doc_id": a["doc_id"],
            "title": a["title"], 
            "text": a["text"],
            "source_abstract": a["source_abstract"],
            "chunk_type": a["chunk_type"],
            "species": a["species"],
            "study_type": a["study_type"],
            "year": a["year"]
        } for a in batch]

        embeddings = embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()

        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    print(f"ChromaDB index built — {collection.count()} chunks indexed")
    return collection


def build_bm25_index(abstracts):
    """Build BM25 keyword index. Saves pkl files to chroma_db/."""
    print("\nBuilding BM25 index...")

    tokenized_corpus = [a["full_text"].lower().split() for a in abstracts]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(f"{CHROMA_DB_PATH}/bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(f"{CHROMA_DB_PATH}/abstracts.pkl", "wb") as f:
        pickle.dump(abstracts, f)

    print(f"BM25 index built — {len(abstracts)} chunks saved")
    return bm25


def verify_index():
    """Verify both indexes are correctly built."""
    print("\nVerifying indexes...")

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

    with open(f"{CHROMA_DB_PATH}/abstracts.pkl", "rb") as f:
        abstracts = pickle.load(f)

    chroma_count = collection.count()
    bm25_count = len(abstracts)

    print(f"ChromaDB : {chroma_count} chunks")
    print(f"BM25     : {bm25_count} chunks")

    assert chroma_count == bm25_count, (
        f"Count mismatch — got ChromaDB:{chroma_count} BM25:{bm25_count}"
    )

    sample = abstracts[0]
    print(f"\nSample chunk:")
    print(f"  ID    : {sample['id']}")
    print(f"  Title : {sample['title'][:80]}")
    print(f"  Text  : {sample['text'][:120]}...")
    print("\nAll indexes verified.")


if __name__ == "__main__":
    print("=" * 60)
    print("Equipoise — Indexing from local corpus.jsonl")
    print("=" * 60)

    abstracts = load_scifact_abstracts()
    build_chroma_index(abstracts)
    build_bm25_index(abstracts)
    verify_index()

    print("\nDone. Run python -m src.retriever to test retrieval.")
"""
ingest.py (Colab version)

SAME LOGIC as the local PyCharm version — the only change is WHERE things
are stored. Everything writes to Google Drive (DRIVE_BASE) instead of the
local disk, because Colab wipes /content when your session disconnects,
but your mounted Drive persists.
"""

import json
import os
import chromadb
from chromadb.utils import embedding_functions

DRIVE_BASE = "/content/drive/MyDrive/PakGuide_RAG"
DATA_PATH = f"{DRIVE_BASE}/train.jsonl"
DB_PATH = f"{DRIVE_BASE}/chroma_db"
COLLECTION_NAME = "pakguide_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_qa_pairs(path: str) -> list[dict]:
    """Reads .jsonl (one JSON object per line) — your confirmed schema:
    {"instruction": "...", "input": "", "output": "...", "topic": "..."}
    """
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    return pairs


def build_vector_store():
    os.makedirs(DRIVE_BASE, exist_ok=True)
    qa_pairs = load_qa_pairs(DATA_PATH)
    print(f"Loaded {len(qa_pairs)} Q&A pairs from {DATA_PATH}")

    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection_names = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in collection_names:
        print(f"Collection '{COLLECTION_NAME}' already exists on Drive — clearing before rebuild.")
        client.delete_collection(COLLECTION_NAME)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"qa_{i}" for i in range(len(qa_pairs))]
    documents = [pair["instruction"] for pair in qa_pairs]
    metadatas = [
        {
            "response": pair["output"],
            "instruction": pair["instruction"],
            "topic": pair.get("topic", ""),
        }
        for pair in qa_pairs
    ]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Ingested {collection.count()} items into ChromaDB at {DB_PATH}")
    print("This is saved on your Drive — you won't need to re-run this after a Colab restart"
          " unless train.jsonl changes.")


if __name__ == "__main__":
    build_vector_store()

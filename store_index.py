"""
store_index.py — Re-index PDF data into Pinecone with improved chunking
========================================================================
Run this script ONCE (or whenever you update your PDFs) to re-embed
your documents with the optimized chunk size (1000 chars, 200 overlap).

Usage:
    python store_index.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

from src.helper import (
    load_pdf_files,
    text_split,
    download_embeddings,
)

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


INDEX_NAME = "medical-chatbot"
DATA_DIR = "data"


def main():
    print("=" * 60)
    print("  Pinecone Re-Indexing — Improved Chunking Strategy")
    print("=" * 60)

    # 1. Load PDFs
    print(f"\n[1/5] Loading PDFs from '{DATA_DIR}'...")
    documents = load_pdf_files(DATA_DIR)
    print(f"      → Loaded {len(documents)} pages")

    # 2. Split
    print("[2/5] Splitting documents (chunk_size=1000, overlap=200)...")
    chunks = text_split(documents)
    print(f"      → Created {len(chunks)} chunks")

    # 3. Embeddings
    print("[3/5] Loading embedding model (BAAI/bge-small-en-v1.5)...")
    embeddings = download_embeddings()

    # 4. Pinecone setup
    print(f"[4/5] Connecting to Pinecone index '{INDEX_NAME}'...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Delete existing index to re-create with fresh data
    if INDEX_NAME in pc.list_indexes().names():
        print(f"      → Deleting existing index '{INDEX_NAME}'...")
        pc.delete_index(INDEX_NAME)

    print(f"      → Creating new index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    # 5. Upsert
    print("[5/5] Embedding and upserting chunks to Pinecone...")
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        index_name=INDEX_NAME,
        embedding=embeddings,
    )

    print(f"\n{'=' * 60}")
    print(f"  ✓ Done! {len(chunks)} chunks indexed into '{INDEX_NAME}'")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

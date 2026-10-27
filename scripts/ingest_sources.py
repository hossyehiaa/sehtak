"""Ingestion CLI runner for trusted medical evidence sources."""

import asyncio
from pathlib import Path
from app.ai.rag.ingest import ingest_manifest_sources
from app.ai.rag.retriever import retriever

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "app" / "ai" / "rag" / "sources_manifest.yaml"


def main():
    print(f"Loading medical sources from: {MANIFEST_PATH}")
    chunks = ingest_manifest_sources(MANIFEST_PATH)
    print(f"Parsed {len(chunks)} medical chunks. Indexing into ChromaDB + BM25...")
    count = retriever.index_chunks(chunks)
    print(f"Successfully indexed {count} chunks into collection 'sehtak_medical_kb'.")


if __name__ == "__main__":
    main()

"""
Document ingestion script and parser for trusted medical sources.
Loads documents from sources_manifest.yaml, parses them, chunks them, and outputs structured chunks.
"""

import hashlib
from pathlib import Path
from typing import List
import yaml

from app.ai.rag.chunker import MedicalChunk, recursive_medical_chunk
from app.config import settings


def read_document_file(file_path: Path) -> str:
    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found: {file_path}")

    ext = file_path.suffix.lower()
    if ext in [".txt", ".md"]:
        return file_path.read_text(encoding="utf-8")
    elif ext == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                return "\n\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            return "\n\n".join(page.get_text() for page in doc)
    else:
        return file_path.read_text(encoding="utf-8", errors="ignore")


def ingest_manifest_sources(manifest_path: Path) -> List[MedicalChunk]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    all_chunks: List[MedicalChunk] = []
    base_dir = manifest_path.parent.parent.parent.parent

    for item in data.get("sources", []):
        doc_id = item["id"]
        rel_path = item["file_path"]
        abs_path = (base_dir / rel_path) if not Path(rel_path).is_absolute() else Path(rel_path)

        text = read_document_file(abs_path)
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()

        metadata = {
            "source_org": str(item["org"]),
            "doc_title": str(item["title"]),
            "url": str(item.get("url", "")),
            "lang": str(item.get("lang", "en")),
            "version": str(item.get("version", "1.0")),
            "section": "General",
            "page": "1",
            "sha256": sha,
        }

        # Look for Section / Page headers in text
        for line in text.split("\n")[:10]:
            if line.lower().startswith("section:"):
                metadata["section"] = line.split(":", 1)[1].strip()
            if line.lower().startswith("page:"):
                metadata["page"] = line.split(":", 1)[1].strip()

        chunks = recursive_medical_chunk(doc_id, text, metadata)
        all_chunks.extend(chunks)

    return all_chunks

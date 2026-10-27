"""
Recursive text chunker for medical evidence documents.
Splits documents into ~800 char chunks with 120 char overlap, preserving sentence boundaries.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MedicalChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, str]


def recursive_medical_chunk(
    doc_id: str,
    text: str,
    metadata: Dict[str, str],
    chunk_size: int = 800,
    overlap: int = 120,
) -> List[MedicalChunk]:
    text = text.strip()
    if not text:
        return []

    chunks: List[MedicalChunk] = []
    # If text is small enough, return as single chunk
    if len(text) <= chunk_size:
        chunks.append(
            MedicalChunk(
                chunk_id=f"{doc_id}_c0",
                text=text,
                metadata=dict(metadata),
            )
        )
        return chunks

    # Split by paragraphs or sentences
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    current_chunk = ""
    chunk_idx = 0

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = (current_chunk + "\n\n" + para).strip()
        else:
            if current_chunk:
                chunks.append(
                    MedicalChunk(
                        chunk_id=f"{doc_id}_c{chunk_idx}",
                        text=current_chunk,
                        metadata=dict(metadata),
                    )
                )
                chunk_idx += 1
                # create overlap from end of current_chunk
                current_chunk = current_chunk[-overlap:] + "\n\n" + para
            else:
                # single paragraph exceeds chunk size, split by lines or characters
                chunks.append(
                    MedicalChunk(
                        chunk_id=f"{doc_id}_c{chunk_idx}",
                        text=para[:chunk_size],
                        metadata=dict(metadata),
                    )
                )
                chunk_idx += 1
                current_chunk = para[chunk_size - overlap :]

    if current_chunk.strip():
        chunks.append(
            MedicalChunk(
                chunk_id=f"{doc_id}_c{chunk_idx}",
                text=current_chunk.strip(),
                metadata=dict(metadata),
            )
        )

    return chunks

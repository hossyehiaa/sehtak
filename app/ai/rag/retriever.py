"""
Hybrid Dense (ChromaDB + multilingual MiniLM) + Sparse (BM25) Retriever with Reciprocal Rank Fusion (RRF).
Applies Org Priors (WHO/EDA/KDIGO/NICE/MOH = 1.0, others = 0.8) and query-language match boosts.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    HAS_CHROMADB = True
except Exception:
    HAS_CHROMADB = False
    ChromaSettings = None

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except Exception:
    HAS_BM25 = False
    BM25Okapi = None

from app.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "sehtak_medical_kb"
BM25_STORE_FILE = "bm25_data.json"

ORG_PRIORS = {
    "WHO": 1.0,
    "EDA": 1.0,
    "KDIGO": 1.0,
    "NICE": 1.0,
    "MOH": 1.0,
}


class MedicalHybridRetriever:
    """Hybrid RRF retriever for trusted medical corpus."""

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = Path(persist_dir or settings.CHROMA_PERSIST_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._embedder: Optional[SentenceTransformer] = None
        self._chroma_client = None
        self._collection = None
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_chunks: List[Dict] = []
        self._init_storage()

    @property
    def embedder(self) -> Optional[Any]:
        if not HAS_SENTENCE_TRANSFORMERS:
            return None
        if self._embedder is None:
            self._embedder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        return self._embedder

    def _init_storage(self) -> None:
        if HAS_CHROMADB:
            try:
                self._chroma_client = chromadb.PersistentClient(
                    path=str(self.persist_dir),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                self._collection = self._chroma_client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                logger.error("Failed to initialize ChromaDB collection: %s", e)

        # Load BM25 index if exists
        bm25_path = self.persist_dir / BM25_STORE_FILE
        if bm25_path.exists():
            try:
                data = json.loads(bm25_path.read_text(encoding="utf-8"))
                self._bm25_chunks = data.get("chunks", [])
                corpus = [c["text"].lower().split() for c in self._bm25_chunks]
                if corpus:
                    self._bm25 = BM25Okapi(corpus)
            except Exception as e:
                logger.error("Failed to load BM25 store: %s", e)

    def count_indexed_chunks(self) -> int:
        if self._collection is not None:
            try:
                return self._collection.count()
            except Exception:
                return len(self._bm25_chunks)
        return len(self._bm25_chunks)

    def index_chunks(self, chunks: List[Any]) -> int:
        """Indexes medical chunks into both ChromaDB and BM25 store."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [c.metadata for c in chunks]

        # 1. Dense embeddings
        embeddings = self.embedder.encode(texts, convert_to_numpy=True).tolist()
        if self._collection is not None:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

        # 2. Sparse BM25
        self._bm25_chunks = [
            {"id": ids[i], "text": texts[i], "metadata": metadatas[i]}
            for i in range(len(chunks))
        ]
        corpus = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(corpus)

        bm25_path = self.persist_dir / BM25_STORE_FILE
        bm25_path.write_text(
            json.dumps({"chunks": self._bm25_chunks}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return len(chunks)

    def retrieve(
        self, query: str, top_dense: int = 20, top_sparse: int = 20, pool_size: int = 15
    ) -> List[Dict]:
        """
        Executes Dense + Sparse retrieval, merges with Reciprocal Rank Fusion (k=60),
        and applies Org Prior & Query Language boost.
        """
        if not query or self.count_indexed_chunks() == 0:
            return []

        is_arabic = any("\u0600" <= ch <= "\u06FF" for ch in query)
        dense_results: List[Dict] = []
        sparse_results: List[Dict] = []

        # 1. Dense retrieval
        if self._collection is not None and self.embedder is not None:
            try:
                q_emb = self.embedder.encode([query], convert_to_numpy=True).tolist()[0]
                res = self._collection.query(
                    query_embeddings=[q_emb],
                    n_results=min(top_dense, self.count_indexed_chunks()),
                )
                if res and res.get("ids") and res["ids"][0]:
                    for idx, cid in enumerate(res["ids"][0]):
                        dense_results.append({
                            "id": cid,
                            "text": res["documents"][0][idx],
                            "metadata": res["metadatas"][0][idx],
                            "rank": idx + 1,
                        })
            except Exception as e:
                logger.warning("Dense query failed: %s", e)

        # 2. Sparse BM25 retrieval
        if self._bm25 is not None and self._bm25_chunks:
            tokenized_query = query.lower().split()
            scores = self._bm25.get_scores(tokenized_query)
            scored_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_sparse]
            for rank_idx, i in enumerate(scored_indices):
                if scores[i] > 0:
                    c = self._bm25_chunks[i]
                    sparse_results.append({
                        "id": c["id"],
                        "text": c["text"],
                        "metadata": c["metadata"],
                        "rank": rank_idx + 1,
                    })

        # 3. Reciprocal Rank Fusion (RRF k=60)
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict] = {}

        k = 60.0
        for item in dense_results:
            cid = item["id"]
            chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + item["rank"]))

        for item in sparse_results:
            cid = item["id"]
            chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + item["rank"]))

        # 4. Apply Org Prior and Language match boost (Task 2.7)
        scored_candidates: List[Tuple[str, float]] = []
        for cid, score in rrf_scores.items():
            meta = chunk_map[cid].get("metadata", {})
            org = meta.get("source_org", "").upper()
            prior = ORG_PRIORS.get(org, 0.8)

            doc_lang = meta.get("lang", "en").lower()
            lang_boost = 1.1 if ((is_arabic and doc_lang == "ar") or (not is_arabic and doc_lang == "en")) else 1.0

            final_score = score * prior * lang_boost
            scored_candidates.append((cid, final_score))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_pool = [chunk_map[cid] for cid, _ in scored_candidates[:pool_size]]
        return top_pool


retriever = MedicalHybridRetriever()

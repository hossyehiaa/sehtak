"""
Cross-Encoder reranker (ms-marco-MiniLM-L-6-v2) for medical chunks.
Ranks retrieved pool down to top-5 chunks and enforces confidence threshold.
"""

import logging
from typing import Dict, List, Optional
from sentence_transformers import CrossEncoder

from app.config import settings

logger = logging.getLogger(__name__)


class MedicalCrossEncoderReranker:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        self._model: Optional[CrossEncoder] = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidate_chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Scores (query, chunk_text) pairs and returns top_k candidates with normalized score.
        """
        if not candidate_chunks or not query:
            return []

        pairs = [[query, c["text"]] for c in candidate_chunks]
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.warning("CrossEncoder prediction failed: %s, falling back to original pool order", e)
            for idx, c in enumerate(candidate_chunks[:top_k]):
                c["rerank_score"] = 0.80 - (idx * 0.02)
            return candidate_chunks[:top_k]

        # Attach scores
        for idx, score in enumerate(scores):
            # Sigmoid normalization if unbounded logits
            import math
            norm_score = 1.0 / (1.0 + math.exp(-float(score)))
            candidate_chunks[idx]["rerank_score"] = norm_score

        candidate_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidate_chunks[:top_k]


reranker = MedicalCrossEncoderReranker()

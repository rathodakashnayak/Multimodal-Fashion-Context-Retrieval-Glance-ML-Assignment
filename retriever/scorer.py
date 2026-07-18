"""
scorer.py
---------
Attribute-aware scoring engine with prompt ensembling.

The core problem with vanilla CLIP for fashion retrieval
--------------------------------------------------------
CLIP embeds the entire query into one vector. For "blue shirt sitting on a
park bench", this single vector must encode color, garment type, AND location.
The resulting embedding is an averaged, "blurred" representation.

Our approach: Dual-Scoring + Prompt Ensembling + Weighted Fusion
----------------------------------------------------------------
1. Parse query into attribute axes (color, clothing, location, style).
2. For each axis, generate 4–6 diverse CLIP-friendly text templates.
3. Average their embeddings (prompt ensembling) → robust axis embedding.
4. Compute cosine similarity between each axis embedding and image embeddings.
5. Fuse axis scores with adaptive weights → final re-ranked Top-K.

Prompt ensembling (from CLIP paper)
------------------------------------
Instead of "a blue shirt", we average embeddings of:
  "a person wearing a blue shirt"
  "a photo of someone in a blue colored shirt"
  "a blue shirt outfit"
  "fashion photo: blue shirt"
This gives a more stable, centred embedding that is less sensitive to phrasing.

Scalability note
----------------
  - Small index (<5K): score all images directly (no pre-filter needed)
  - Large index (>5K): ChromaDB HNSW pre-filter to top-N, then re-rank
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from indexer.feature_extractor import CLIPFeatureExtractor
from indexer.vector_store import VectorStore
from retriever.query_parser import ParsedQuery, QueryParser

# ─────────────────────────── weights ─────────────────────────────

# Base fusion weights per axis
BASE_WEIGHTS: Dict[str, float] = {
    "global": 0.30,
    "color_clothing": 0.35,
    "color": 0.20,
    "clothing": 0.20,
    "location": 0.20,
    "style": 0.10,
    "composite": 0.25,
}

# When ≥2 non-global axes are detected, boost their weights
MULTI_ATTR_BOOST = 1.25

# Use full index when collection size is below this threshold
SMALL_INDEX_THRESHOLD = 2000
CANDIDATE_POOL_SIZE = 200


class AttributeScorer:
    """
    Retrieves and re-ranks images using prompt-ensembled attribute scores.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        candidate_pool: int = CANDIDATE_POOL_SIZE,
    ):
        self.base_weights = weights or BASE_WEIGHTS
        self.candidate_pool = candidate_pool
        self.extractor = CLIPFeatureExtractor()
        self.store = VectorStore()
        self.parser = QueryParser()

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Main entry point. Returns top-k images for a natural language query.
        """
        print(f"\n[Scorer] Query: '{query}'")

        total_indexed = self.store.count()
        if total_indexed == 0:
            print("[Scorer] No images indexed. Run: python indexer/index_images.py")
            return []

        parsed = self.parser.parse(query)
        print(f"[Scorer] {parsed.summary()}")
        print(f"[Scorer] Axes: {list(parsed.ensemble_queries.keys())}")

        axis_embeddings = self._encode_axes(parsed)

        if total_indexed <= SMALL_INDEX_THRESHOLD:
            ids, all_embs, metas = self.store.get_all_embeddings()
            candidates = [
                {
                    "id": ids[i],
                    "path": metas[i].get("path", ""),
                    "filename": metas[i].get("filename", ""),
                    "source": metas[i].get("source", ""),
                    "embedding": all_embs[i],
                }
                for i in range(len(ids))
            ]
        else:
            global_emb = axis_embeddings.get("global")
            candidates = self.store.query(global_emb, top_k=self.candidate_pool)

        print(f"[Scorer] Scoring {len(candidates)} candidates ...")

        weights = self._adaptive_weights(parsed)
        candidate_embeddings = np.stack([c["embedding"] for c in candidates])
        scored = self._score_candidates(candidates, candidate_embeddings, axis_embeddings, weights)

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        top = scored[:top_k]
        for i, r in enumerate(top, 1):
            print(
                f"  [{i}] {r['filename']} — score={r['final_score']:.4f} "
                f"({', '.join(f'{k}={v:.3f}' for k, v in r['breakdown'].items())})"
            )
        return top

    def _encode_axes(self, parsed: ParsedQuery) -> Dict[str, np.ndarray]:
        """Encode each attribute axis using prompt ensembling."""
        axis_embeddings: Dict[str, np.ndarray] = {}
        for axis, templates in parsed.ensemble_queries.items():
            if len(templates) == 1:
                axis_embeddings[axis] = self.extractor.encode_text(templates[0])
            else:
                axis_embeddings[axis] = self.extractor.encode_text_ensemble(templates)
        return axis_embeddings

    def _adaptive_weights(self, parsed: ParsedQuery) -> Dict[str, float]:
        """Boost weights when multiple attributes are detected."""
        weights = dict(self.base_weights)
        active_axes = [a for a in parsed.ensemble_queries if a not in ("global",)]

        if len(active_axes) >= 2:
            for ax in active_axes:
                if ax in weights:
                    weights[ax] = weights[ax] * MULTI_ATTR_BOOST
            weights["global"] = max(0.15, weights.get("global", 0.30) * 0.6)

        return weights

    def _score_candidates(
        self,
        candidates: List[Dict],
        candidate_embeddings: np.ndarray,
        axis_embeddings: Dict[str, np.ndarray],
        weights: Dict[str, float],
    ) -> List[Dict]:
        """Compute weighted cosine similarity fusion for all candidates."""
        results = []
        for i, candidate in enumerate(candidates):
            img_emb = candidate_embeddings[i]
            breakdown: Dict[str, float] = {}
            fusion_num = 0.0
            fusion_den = 0.0

            for axis, text_emb in axis_embeddings.items():
                sim = float(np.dot(img_emb, text_emb))
                breakdown[axis] = round(sim, 4)
                w = weights.get(axis, 0.1)
                fusion_num += w * sim
                fusion_den += w

            final_score = fusion_num / fusion_den if fusion_den > 0 else 0.0

            results.append(
                {
                    "id": candidate["id"],
                    "path": candidate["path"],
                    "filename": candidate["filename"],
                    "source": candidate["source"],
                    "global_score": breakdown.get("global", 0.0),
                    "final_score": final_score,
                    "breakdown": breakdown,
                }
            )
        return results


if __name__ == "__main__":
    scorer = AttributeScorer()
    test_queries = [
        "Someone wearing a blue shirt sitting on a park bench.",
        "A person in a bright yellow raincoat.",
        "Professional business attire inside a modern office.",
    ]
    for q in test_queries:
        results = scorer.search(q, top_k=3)
        print(f"\nTop results for: '{q}'")
        for r in results:
            print(f"  [{r['final_score']:.3f}] {r['filename']}")

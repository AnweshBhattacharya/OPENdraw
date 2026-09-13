"""Rank a frozen Clippy index against a query image.

The comparator score is a fixed weighted sum of two cosine similarities
(image, edge). The weights are a frozen constant, not a fitted parameter —
Clippy has no training step, by design; see the package docstring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from clippy.embeddings import ClippyEmbedding, embed_image
from clippy.gallery import ClippyIndex

# Equal weight by default: the report calls for comparing image *and* edge
# embeddings, not picking a winner between them.
IMAGE_WEIGHT = 0.5
EDGE_WEIGHT = 0.5


@dataclass(frozen=True)
class ClippyResult:
    asset_id: str
    line_art_path: str
    score: float
    image_similarity: float
    edge_similarity: float


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    # Both sides are already L2-normalized by embeddings.py; guard the
    # degenerate (zero-vector) case explicitly rather than trusting that.
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def rank(index: ClippyIndex, query: ClippyEmbedding, top_k: int = 8) -> list[ClippyResult]:
    """Rank every entry in ``index`` against ``query``, best first."""
    results: list[ClippyResult] = []
    for entry in index.entries:
        image_similarity = _cosine(query.image, entry.embedding.image)
        edge_similarity = _cosine(query.edge, entry.embedding.edge)
        score = IMAGE_WEIGHT * image_similarity + EDGE_WEIGHT * edge_similarity
        results.append(
            ClippyResult(
                asset_id=entry.asset_id,
                line_art_path=entry.line_art_path,
                score=score,
                image_similarity=image_similarity,
                edge_similarity=edge_similarity,
            )
        )
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def rank_query_image(index: ClippyIndex, query_path: Path, top_k: int = 8) -> list[ClippyResult]:
    return rank(index, embed_image(query_path), top_k=top_k)

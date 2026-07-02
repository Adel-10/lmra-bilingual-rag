"""
Phase 2b — the vector store. Deliberately tiny.

At 1,105 chunks, "vector database" means: a float32 matrix of L2-normalized
embeddings and one matrix-vector product. Cosine similarity of normalized
vectors IS the dot product, so exact search over the whole corpus is:

    scores = matrix @ query_vector          # (N,d) @ (d,) -> (N,)
    top_k  = argsort(scores)[-k:]

No approximate index (FAISS/HNSW) can beat exact search at this scale, and
keeping it as one visible matmul means retrieval has zero magic in it.

Files (written by embed.py):
    data/index/embeddings.npy   (N, d) float32, L2-normalized rows
    data/index/index_meta.json  model name + dims — guards against
                                embedding queries with a different model
    data/chunks/chunks.jsonl    chunk i in this file = row i in the matrix
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.jsonl"
INDEX_DIR = ROOT / "data" / "index"


class VectorStore:
    def __init__(self) -> None:
        self.chunks: list[dict] = [
            json.loads(line)
            for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
        ]
        self.matrix = np.load(INDEX_DIR / "embeddings.npy")
        self.meta = json.loads((INDEX_DIR / "index_meta.json").read_text())
        assert self.matrix.shape[0] == len(self.chunks), (
            f"index has {self.matrix.shape[0]} rows but chunks.jsonl has "
            f"{len(self.chunks)} chunks — re-run embed.py"
        )

    def search(
        self,
        query_vec: np.ndarray,
        k: int = 5,
        language: str | None = None,
    ) -> list[dict]:
        """
        Exact cosine top-k. `language` restricts to 'en'/'ar' chunks —
        used by the cross-lingual tests to FORCE retrieval from the other
        language; normal usage searches both.

        Returns chunk dicts (copies) with a `score` field added.
        """
        scores = self.matrix @ query_vec  # cosine, since rows & query are normalized

        if language is not None:
            mask = np.array([c["language"] == language for c in self.chunks])
            scores = np.where(mask, scores, -np.inf)

        top = np.argsort(scores)[::-1][:k]
        results = []
        for i in top:
            if scores[i] == -np.inf:
                break
            c = dict(self.chunks[i])
            c["score"] = float(scores[i])
            results.append(c)
        return results

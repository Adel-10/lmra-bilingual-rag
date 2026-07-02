"""
Interactive retrieval CLI — the eyeball test.

    python pipeline/search.py "How much is a two-year domestic permit?"
    python pipeline/search.py "كم رسوم تجديد تصريح العمل؟" --k 8
    python pipeline/search.py "medical exam rules" --lang ar   # force Arabic results
"""

from __future__ import annotations

import argparse

from sentence_transformers import SentenceTransformer
from store import VectorStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--lang", choices=["en", "ar"], default=None,
                    help="restrict results to one language (default: both)")
    args = ap.parse_args()

    store = VectorStore()
    model = SentenceTransformer(store.meta["model"])
    qv = model.encode(args.query, normalize_embeddings=True)

    for r in store.search(qv, k=args.k, language=args.lang):
        print(f"\n[{r['score']:.3f}] ({r['language']}) {r['chunk_id']}")
        print(f"    {r['section']}")
        print(f"    {r['source_url']}")
        preview = r["text"].replace("\n", " ")
        print(f"    {preview[:180]}")


if __name__ == "__main__":
    main()

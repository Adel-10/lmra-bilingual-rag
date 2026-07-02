"""
Phase 2b — embed all chunks with BGE-M3 (dense vectors only).

Model choice (see README for the full comparison):
  BAAI/bge-m3 — multilingual (XLM-RoBERTa backbone, shared subword vocab
  across ~100 languages), contrastively trained on translation and
  cross-lingual QA pairs, which is what places an English query and an
  Arabic passage about the same work-permit rule near each other in the
  embedding space. 8192-token window: none of our chunks truncate.
  We use dense mode only; M3's sparse/ColBERT outputs are out of scope
  for v1 (transparency > squeezing the last few nDCG points).

Embeddings are L2-normalized at encode time so cosine similarity becomes
a plain dot product in store.py.

Run on a machine with internet access (downloads ~2.3 GB on first run):
    pip install -r requirements.txt
    python pipeline/embed.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.jsonl"
INDEX_DIR = ROOT / "data" / "index"

MODEL_NAME = "BAAI/bge-m3"

# Memory notes (learned the hard way on a Mac / MPS backend):
#   - sentence-transformers sorts inputs by length, so the FIRST batches
#     are the longest chunks — OOM shows up immediately, not at random.
#   - Activation memory scales with batch_size x padded_seq_len. Batch 4
#     at seq cap 2048 fits comfortably in the ~9 GB MPS pool.
#   - Our longest chunk is ~5.5K chars (~2000 tokens), so capping the
#     sequence length at 2048 truncates NOTHING; it only stops the model
#     from allocating padding buffers toward its 8192 maximum.
BATCH_SIZE = 4
MAX_SEQ_LENGTH = 2048


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--device", default=None,
                    help="e.g. cpu — fallback if MPS/GPU runs out of memory")
    args = ap.parse_args()

    chunks = [json.loads(l) for l in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()]
    print(f"embedding {len(chunks)} chunks with {MODEL_NAME} ...")

    model = SentenceTransformer(MODEL_NAME, device=args.device)
    model.max_seq_length = MAX_SEQ_LENGTH
    t0 = time.time()
    vectors = model.encode(
        [c["text"] for c in chunks],
        batch_size=args.batch_size,
        normalize_embeddings=True,   # -> cosine == dot product
        show_progress_bar=True,
    ).astype(np.float32)
    print(f"done in {time.time() - t0:.0f}s, shape {vectors.shape}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", vectors)
    (INDEX_DIR / "index_meta.json").write_text(json.dumps({
        "model": MODEL_NAME,
        "dims": int(vectors.shape[1]),
        "chunks": int(vectors.shape[0]),
        "normalized": True,
    }, indent=2))
    print(f"saved -> {INDEX_DIR}/embeddings.npy")


if __name__ == "__main__":
    main()

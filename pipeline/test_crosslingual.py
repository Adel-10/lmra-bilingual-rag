"""
Phase 2c — cross-lingual retrieval tests. MAKE-OR-BREAK GATE.

If an English query cannot surface the correct Arabic chunk (and vice
versa), the project's core premise fails and we stop here. Three tests:

1. EN query -> AR corpus only (forced), and AR query -> EN corpus only.
   Forcing the opposite language isolates CROSS-lingual ability from the
   easy same-language path. Metric: is the expected doc in the top-5?

2. The real-life case: English queries about content that ONLY exists in
   Arabic (the Arabic-only laws). No forcing needed — search everything;
   the right answer simply has no English version to hide behind.

3. Language-residual measurement: cosine of parallel EN/AR chunk pairs
   (same doc, same position — translations of each other) vs. random
   same-language and cross-language pairs. Quantifies how much "which
   language" still bleeds into the geometry. We expect:
       parallel pairs >> random pairs,
       random same-language slightly > random cross-language.

Writes data/index/crosslingual_report.json for the record.

    python pipeline/test_crosslingual.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from store import VectorStore, INDEX_DIR

# ---- test cases -----------------------------------------------------------
# expected = doc_id prefixes that count as a hit (any chunk of those docs).

EN_TO_AR = [
    ("How much does a two-year work permit for a domestic worker cost?",
     ["435_ar", "471_ar"]),
    ("What documents are required to renew a domestic worker's permit?",
     ["449_ar"]),
    ("How do I transfer a foreign worker to another employer?",
     ["194_ar", "169_ar"]),
    ("Medical examination requirements for expatriate employees",
     ["203_ar", "439_ar"]),
    ("Can a golden residency holder get a work permit?",
     ["539_ar", "540_ar", "542_ar"]),
]

AR_TO_EN = [
    ("كم رسوم إصدار تصريح عمل جديد لعامل أجنبي؟",
     ["106_en", "133_en"]),
    ("ما هي المستندات المطلوبة لتسجيل منشأة جديدة؟",
     ["130_en"]),
    ("كيف يمكن للعامل الأجنبي الانتقال إلى صاحب عمل آخر؟",
     ["194_en", "169_en"]),
    ("إلغاء تصريح عمل منتهي الصلاحية",
     ["567_en"]),
    ("ما هي طرق الدفع المتاحة لرسوم الهيئة؟",
     ["212_en"]),
]

# English questions whose answers exist ONLY in Arabic (Arabic-only laws,
# indexed via their _ar doc). Unrestricted search — the honest scenario.
EN_TO_AR_ONLY = [
    ("What does the law say about settlement of labour market crimes?",
     ["law_12_ar"]),
    ("Which resolution regulates employment office licenses?",
     ["law_114_ar"]),
    ("resolution about assigning LMRA tasks to labour registration centres",
     ["law_107_ar"]),
]


def run_battery(name, cases, store, model, language=None, k=5):
    results, hits = [], 0
    for query, expected in cases:
        qv = model.encode(query, normalize_embeddings=True)
        got = store.search(qv, k=k, language=language)
        got_docs = [r["doc_id"] for r in got]
        hit_rank = next(
            (i + 1 for i, d in enumerate(got_docs)
             if any(d.startswith(e) for e in expected)), None)
        hits += hit_rank is not None
        results.append({
            "query": query, "expected": expected, "hit_rank": hit_rank,
            "top": [
                {"doc_id": r["doc_id"], "score": round(r["score"], 3),
                 "lang": r["language"], "section": r["section"][:80]}
                for r in got[:3]
            ],
        })
        mark = f"hit@{hit_rank}" if hit_rank else "MISS"
        print(f"  [{mark:6s}] {query[:64]}")
        for r in got[:3]:
            print(f"      {r['score']:.3f} ({r['language']}) {r['doc_id']:14s} {r['section'][:60]}")
    print(f"  => {name}: {hits}/{len(cases)} in top-{k}\n")
    return {"name": name, "hits": hits, "total": len(cases), "cases": results}


def language_residual(store, model_dims_unused=None, n=200, seed=0):
    """Cosine stats: parallel EN/AR chunk pairs vs random pairs."""
    rng = random.Random(seed)
    by_id = {c["chunk_id"]: i for i, c in enumerate(store.chunks)}
    M = store.matrix

    # parallel pairs: same doc name & chunk position, en <-> ar
    parallel = []
    for cid, i in by_id.items():
        if "_en#" in cid:
            j = by_id.get(cid.replace("_en#", "_ar#"))
            if j is not None:
                parallel.append(float(M[i] @ M[j]))
    parallel = rng.sample(parallel, min(n, len(parallel)))

    def random_pairs(lang_a, lang_b):
        ia = [i for i, c in enumerate(store.chunks) if c["language"] == lang_a]
        ib = [i for i, c in enumerate(store.chunks) if c["language"] == lang_b]
        out = []
        for _ in range(n):
            i, j = rng.choice(ia), rng.choice(ib)
            if i != j:
                out.append(float(M[i] @ M[j]))
        return out

    stats = {
        "parallel_en_ar": parallel,
        "random_en_en": random_pairs("en", "en"),
        "random_ar_ar": random_pairs("ar", "ar"),
        "random_en_ar": random_pairs("en", "ar"),
    }
    summary = {k: {"mean": round(float(np.mean(v)), 3),
                   "std": round(float(np.std(v)), 3)}
               for k, v in stats.items()}
    print("language-residual check (cosine):")
    for k, s in summary.items():
        print(f"  {k:16s} mean {s['mean']}  (n={len(stats[k])})")
    return summary


def main() -> None:
    store = VectorStore()
    model = SentenceTransformer(store.meta["model"])
    print(f"corpus: {len(store.chunks)} chunks | model: {store.meta['model']}\n")

    report = {"batteries": [], "residual": None}
    print("=== EN query -> AR corpus (forced) ===")
    report["batteries"].append(run_battery("EN->AR", EN_TO_AR, store, model, language="ar"))
    print("=== AR query -> EN corpus (forced) ===")
    report["batteries"].append(run_battery("AR->EN", AR_TO_EN, store, model, language="en"))
    print("=== EN query -> Arabic-only content (unrestricted) ===")
    report["batteries"].append(run_battery("EN->AR-only", EN_TO_AR_ONLY, store, model))

    report["residual"] = language_residual(store)

    out = INDEX_DIR / "crosslingual_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()

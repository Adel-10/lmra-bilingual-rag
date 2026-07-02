"""
Phase 3 — retrieval + generation with guardrails.

The answer path, in order:

  1. Detect the query language (Arabic codepoint ratio).
  2. BILINGUAL QUERY EXPANSION — translate the query to the other corpus
     language with a small LLM call, embed BOTH versions, and score every
     chunk by max(cos(q_orig), cos(q_translated)). This is the fix for the
     hardest cross-lingual case (casual English -> Arabic-only legal
     prose): the translated query meets the Arabic text in ITS language.
  3. HARD ABSTENTION GATE — if the best merged score is below
     RELEVANCE_THRESHOLD, the LLM is never called. No context, no chance
     to hallucinate. A fixed template (in the user's language) says the
     corpus doesn't cover it and points to LMRA.
  4. GROUNDED GENERATION — retrieved chunks go into the prompt as
     numbered sources; the system prompt forbids outside knowledge,
     requires [n] citations, and instructs the model to say "not covered"
     when sources don't answer (the SOFT abstention layer).
  5. CITATION POST-CHECK — [n] markers are parsed and mapped back to the
     retrieved chunks; an answer with no valid citations is downgraded to
     an abstention rather than shown as fact.

Three layers of abstention (3, 4, 5): each catches what the previous
one misses. High-stakes civic info: a confidently wrong answer sends
someone to a ministry with the wrong documents.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python pipeline/rag.py "How much does a domestic worker permit cost?"
    python pipeline/rag.py "ما هي شروط نقل العامل الأجنبي؟" --debug
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

from store import VectorStore

# Small+cheap for translation, stronger for answers. Adjust to taste.
TRANSLATE_MODEL = "claude-haiku-4-5"
ANSWER_MODEL = "claude-sonnet-4-5"

TOP_K = 6

# Initial value read off the Phase 2 test battery: correct hits scored
# 0.63-0.77, topical-but-wrong ~0.58-0.63. Calibrated properly by the
# Phase 4 eval harness (see README).
RELEVANCE_THRESHOLD = 0.62

ABSTAIN_TEXT = {
    "en": (
        "I couldn't find this in the available LMRA content, so I won't guess. "
        "Please verify directly with the Labour Market Regulatory Authority "
        "(https://lmra.gov.bh, call centre +973 17506055)."
    ),
    "ar": (
        "لم أجد هذه المعلومة في محتوى هيئة تنظيم سوق العمل المتوفر لديّ، "
        "لذلك لن أخمّن. يُرجى التحقق مباشرة من هيئة تنظيم سوق العمل "
        "(https://lmra.gov.bh، مركز الاتصال ٩٧٣١٧٥٠٦٠٥٥+)."
    ),
}

SYSTEM_PROMPT = """\
You are an assistant answering questions about Bahrain's Labour Market \
Regulatory Authority (LMRA) services: work permits, fees, required \
documents, procedures, and related regulations.

STRICT RULES — this is high-stakes civic information:
1. Answer ONLY from the numbered sources provided. Never use outside \
knowledge, even if you are confident. Never invent fees, documents, \
steps, or conditions.
2. Cite every factual claim with the source number in square brackets, \
e.g. [1] or [2][3]. Do not fabricate citations.
3. Answer in {answer_language}. The sources may be in English or Arabic \
— translate their content faithfully into the answer language as needed.
4. If the sources do not contain the answer, begin your reply with the \
exact marker [NOT_IN_SOURCES] and then briefly say the available LMRA \
content does not cover this and to verify with LMRA. If they answer only \
partially, answer what is covered and state the gap explicitly. A partial \
answer with an honest gap is better than a complete-looking wrong one.
5. Do not add procedural advice that is not in the sources. Do not refer \
the user to other authorities, websites, or services unless a source \
names them for this purpose.
6. Keep the answer concise and practical. Preserve exact figures \
(fees, durations, validity periods) as written in the sources.
7. Attribute contact details (phone numbers, offices, portals) EXACTLY \
as the sources do — never reassign a number or office from one service \
to another. (A misattributed hotline is a wrong answer.)"""


def detect_language(text: str) -> str:
    letters = re.findall(r"[A-Za-z؀-ۿ]", text)
    if not letters:
        return "en"
    ar = len(re.findall(r"[؀-ۿ]", text))
    return "ar" if ar / len(letters) > 0.5 else "en"


class RagPipeline:
    def __init__(self) -> None:
        self.store = VectorStore()
        self.embedder = SentenceTransformer(self.store.meta["model"])
        self.client = Anthropic()

    # ---- step 2: bilingual expansion -----------------------------------

    def translate_query(self, query: str, target: str) -> str:
        target_name = "Modern Standard Arabic" if target == "ar" else "English"
        msg = self.client.messages.create(
            model=TRANSLATE_MODEL,
            max_tokens=300,
            system=(
                "Translate the user's query into "
                f"{target_name}. Domain: Bahrain labour-market government "
                "services (work permits, fees, LMRA). Output ONLY the "
                "translation, nothing else."
            ),
            messages=[{"role": "user", "content": query}],
        )
        return msg.content[0].text.strip()

    def retrieve(self, query: str, k: int = TOP_K) -> tuple[list[dict], dict]:
        """Merged bilingual retrieval: score = max over both query versions."""
        lang = detect_language(query)
        other = "ar" if lang == "en" else "en"
        translated = self.translate_query(query, other)

        q1 = self.embedder.encode(query, normalize_embeddings=True)
        q2 = self.embedder.encode(translated, normalize_embeddings=True)
        # one matmul per query version; final score is the elementwise max
        scores = np.maximum(self.store.matrix @ q1, self.store.matrix @ q2)

        top = np.argsort(scores)[::-1][:k]
        results = []
        for i in top:
            c = dict(self.store.chunks[i])
            c["score"] = float(scores[i])
            results.append(c)
        debug = {"query_language": lang, "translated_query": translated,
                 "top_score": results[0]["score"] if results else 0.0}
        return results, debug

    # ---- steps 3-5: gate, generate, post-check --------------------------

    def answer(self, query: str, k: int = TOP_K,
               threshold: float = RELEVANCE_THRESHOLD) -> dict:
        lang = detect_language(query)
        retrieved, debug = self.retrieve(query, k=k)

        # (3) hard gate: below-threshold retrieval never reaches the LLM
        if not retrieved or retrieved[0]["score"] < threshold:
            return {
                "answer": ABSTAIN_TEXT[lang], "abstained": True,
                "reason": f"top score {debug['top_score']:.3f} < {threshold}",
                "sources": [], "retrieved": retrieved, "debug": debug,
            }

        # (4) grounded generation
        source_block = "\n\n".join(
            f"[{n}] {c['section']}\nURL: {c['source_url']}\n{c['text']}"
            for n, c in enumerate(retrieved, 1)
        )
        answer_language = "Modern Standard Arabic" if lang == "ar" else "English"
        msg = self.client.messages.create(
            model=ANSWER_MODEL,
            max_tokens=1200,
            system=SYSTEM_PROMPT.format(answer_language=answer_language),
            messages=[{"role": "user", "content":
                       f"SOURCES:\n\n{source_block}\n\nQUESTION: {query}"}],
        )
        text = msg.content[0].text.strip()

        # (4b) structured soft abstention: the model marks its own refusals,
        # so downstream code (and the eval harness) detects them
        # deterministically instead of guessing from citation counts.
        # Marker ANYWHERE in the text counts. Observed: for partially-
        # covered questions the model summarizes what IS covered and only
        # then emits the marker — an answer that self-declares the core
        # question unanswered must not pass as a grounded answer.
        if "[NOT_IN_SOURCES]" in text:
            # Display the FIXED template, not the model's refusal prose.
            # Tested behaviour: even when told not to, the model names
            # other authorities ("contact the Traffic Directorate...") in
            # refusals — outside knowledge we can't verify. Deterministic
            # template = zero leakage; the model text goes to debug only.
            return {
                "answer": ABSTAIN_TEXT[lang],
                "abstained": True, "reason": "model reported not in sources",
                "sources": [], "retrieved": retrieved,
                "debug": {**debug, "model_refusal": text},
            }

        # (5) citation post-check: which sources did the answer actually use?
        cited_nums = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)
                             if 1 <= int(n) <= len(retrieved)})
        sources = [{
            "n": n,
            "title": retrieved[n - 1]["title"],
            "section": retrieved[n - 1]["section"],
            "url": retrieved[n - 1]["source_url"],
            "language": retrieved[n - 1]["language"],
        } for n in cited_nums]

        # An answer that cites nothing is not a grounded answer. The model
        # legitimately omits citations when it is REFUSING (rule 4), so
        # treat citation-free output as abstention, not as fact.
        abstained = not cited_nums
        return {
            "answer": text, "abstained": abstained,
            "reason": "no citations in answer" if abstained else None,
            "sources": sources, "retrieved": retrieved, "debug": debug,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--threshold", type=float, default=RELEVANCE_THRESHOLD)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    rag = RagPipeline()
    out = rag.answer(args.query, k=args.k, threshold=args.threshold)

    print("\n" + out["answer"] + "\n")
    if out["sources"]:
        print("Sources:")
        for s in out["sources"]:
            print(f"  [{s['n']}] ({s['language']}) {s['section'][:70]}")
            print(f"      {s['url']}")
    if out["abstained"]:
        print(f"(abstained — {out['reason']})")
    if args.debug:
        dbg = {**out["debug"],
               "retrieved": [
                   {"chunk_id": c["chunk_id"], "score": round(c["score"], 3),
                    "lang": c["language"]} for c in out["retrieved"]]}
        print("\nDEBUG:", json.dumps(dbg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""
Phase 4a — build the seed evaluation set. Three sources:

1. FAQ-derived (automatic): LMRA's own FAQ pages are natural questions,
   and many answers link to the exact service page — a free ground-truth
   citation written by the authority itself. We take FAQ questions whose
   answer body links at least one /page/show/ URL; those URLs (plus the
   FAQ page itself) become expected sources.

2. Cross-lingual battery (curated): the Phase 2 test cases, where the
   expected source is in the OTHER language from the query — including
   the Arabic-only laws queried in English.

3. Out-of-corpus probes (curated): questions a Bahrain resident might
   ask that LMRA content genuinely does not answer. Correct behaviour
   is abstention.

Output: eval/testset.jsonl — one case per line:
  {id, question, language, type: answerable|out_of_corpus,
   expected_urls: [...], cross_lingual: bool, origin}

The harness (run_eval.py) accepts ANY file in this format.

    python eval/build_testset.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "eval" / "testset.jsonl"

PER_LANG_FAQ_CASES = 12   # keep judge costs sane; extend the file anytime
PAGE_LINK_RE = re.compile(r"https?://(?:www\.)?lmra\.gov\.bh/(en|ar)/page/show/(\d+)")


def norm_url(lang: str, page_id: str) -> str:
    return f"lmra.gov.bh/{lang}/page/show/{page_id}"


def faq_cases() -> list[dict]:
    cases = []
    for path in sorted(CLEAN.glob("faq_cat*_*.md")):
        lang = path.stem.rsplit("_", 1)[1]
        text = path.read_text(encoding="utf-8")
        body = text.split("---", 2)[2]
        faq_url_m = re.search(r"source_url: (\S+)", text)
        faq_url = faq_url_m.group(1).replace("https://", "").replace("www.", "")

        # the same FAQ page in the OTHER language is equally a correct
        # source — citing the EN twin of an AR page is not an error
        other = "ar" if lang == "en" else "en"
        faq_url_both = [faq_url, faq_url.replace(f"/{lang}/", f"/{other}/")]

        # sections are "## <question>" followed by the answer body
        sections = re.split(r"^## ", body, flags=re.M)[1:]
        for sec in sections:
            lines = sec.splitlines()
            question = re.sub(r"^\d+\s*[-–]\s*", "", lines[0]).strip()
            answer = "\n".join(lines[1:])
            # drop per-question "Last Update" noise
            answer = re.sub(r"Last Update.*", "", answer)
            links = PAGE_LINK_RE.findall(answer)
            if not links:
                continue  # only questions with a page-link ground truth
            # seed hygiene: extracted standalone, a question must carry its
            # own context. Drop non-questions ("Labour Inspection
            # Directorate") and vague stubs ("What are the fees?") that
            # only made sense inside their FAQ category page.
            if len(question) < 25 or not question.rstrip().endswith(("?", "؟")):
                continue
            expected = [norm_url(l, p) for l, p in links]
            # the linked page in EITHER language counts as a correct source
            expected += [norm_url("ar" if l == "en" else "en", p) for l, p in links]
            cases.append({
                "question": question,
                "language": lang,
                "type": "answerable",
                "expected_urls": sorted(set(expected + faq_url_both)),
                "cross_lingual": False,
                "origin": f"faq:{path.stem}",
            })
    # stratified sample: spread across categories, fixed seed = reproducible
    rng = random.Random(42)
    out = []
    for lang in ("en", "ar"):
        pool = [c for c in cases if c["language"] == lang]
        rng.shuffle(pool)
        out.extend(pool[:PER_LANG_FAQ_CASES])
    return out


# ---- curated: cross-lingual (expected source language != query language) --

CROSS_LINGUAL = [
    # EN query -> AR source (forced pairs from the Phase 2 battery)
    ("How much does a two-year work permit for a domestic worker cost?",
     "en", ["lmra.gov.bh/ar/page/show/435", "lmra.gov.bh/ar/page/show/471"]),
    ("What documents are required to renew a domestic worker's permit?",
     "en", ["lmra.gov.bh/ar/page/show/449", "lmra.gov.bh/en/page/show/449"]),
    ("Medical examination requirements for expatriate employees",
     "en", ["lmra.gov.bh/ar/page/show/203", "lmra.gov.bh/en/page/show/203",
            "lmra.gov.bh/ar/page/show/439", "lmra.gov.bh/en/page/show/439"]),
    # EN query -> Arabic-only law (the hardest case). Both language URLs
    # listed: citing either variant of the same law is correct.
    ("What does the law say about settlement of labour market crimes?",
     "en", ["lmra.gov.bh/ar/legal/show/12", "lmra.gov.bh/en/legal/show/12",
            "lmra.gov.bh/ar/legal/show/115", "lmra.gov.bh/en/legal/show/115"]),
    ("Which resolution regulates employment office licenses?",
     "en", ["lmra.gov.bh/ar/legal/show/114", "lmra.gov.bh/en/legal/show/114"]),
    ("What are the employer's obligations if a foreign worker leaves work?",
     "en", ["lmra.gov.bh/ar/legal/show/13", "lmra.gov.bh/en/legal/show/13"]),
    # AR query -> EN source
    ("ما هي المستندات المطلوبة لتسجيل منشأة جديدة؟",
     "ar", ["lmra.gov.bh/en/page/show/130", "lmra.gov.bh/ar/page/show/130"]),
    ("كيف يمكن للعامل الأجنبي الانتقال إلى صاحب عمل آخر؟",
     "ar", ["lmra.gov.bh/en/page/show/194", "lmra.gov.bh/ar/page/show/194"]),
    ("ما هي طرق الدفع المتاحة لرسوم الهيئة؟",
     "ar", ["lmra.gov.bh/en/page/show/212", "lmra.gov.bh/ar/page/show/212"]),
]

# ---- curated: out-of-corpus — correct behaviour is ABSTAIN ----------------

OUT_OF_CORPUS = [
    ("How do I renew my Bahraini driving license?", "en"),
    ("What are the visa requirements for visiting Saudi Arabia from Bahrain?", "en"),
    ("How much does it cost to register a car in Bahrain?", "en"),
    ("كيف أسجل أطفالي في مدرسة حكومية في البحرين؟", "ar"),
    ("ما هي شروط الحصول على الجنسية البحرينية؟", "ar"),
    ("كم رسوم استخراج جواز سفر بحريني؟", "ar"),
]


def main() -> None:
    cases = faq_cases()
    for q, lang, urls in CROSS_LINGUAL:
        cases.append({"question": q, "language": lang, "type": "answerable",
                      "expected_urls": urls, "cross_lingual": True,
                      "origin": "curated:cross-lingual"})
    for q, lang in OUT_OF_CORPUS:
        cases.append({"question": q, "language": lang, "type": "out_of_corpus",
                      "expected_urls": [], "cross_lingual": False,
                      "origin": "curated:out-of-corpus"})

    # Stable content-hash IDs: results are cached per-id across runs, so
    # ids must survive testset regeneration (positional ids would force a
    # full, paid re-run every time a single case is added or removed).
    import hashlib
    for c in cases:
        h = hashlib.md5(c["question"].encode("utf-8")).hexdigest()[:8]
        c["id"] = f"case_{h}"

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    by = {}
    for c in cases:
        key = (c["type"], c["language"], c["cross_lingual"])
        by[key] = by.get(key, 0) + 1
    print(f"{len(cases)} cases -> {OUT}")
    for (t, l, x), n in sorted(by.items()):
        print(f"  {t:14s} {l}  {'cross-lingual' if x else '':13s} {n}")


if __name__ == "__main__":
    main()

"""
Phase 4b — the evaluation harness. Dataset-agnostic: point it at ANY
JSONL file of test cases (see build_testset.py for the schema) and it
produces the same report.

    python eval/run_eval.py                          # default testset, with judge
    python eval/run_eval.py --no-judge               # deterministic metrics only
    python eval/run_eval.py --testset my_cases.jsonl --k 6

Metrics
-------
retrieval hit@k    expected source among top-k retrieved chunks.
                   Deterministic URL match — no LLM involved.
citation correct   a source the answer actually CITES matches an expected
                   source. Distinct from retrieval: the right chunk can be
                   retrieved but not cited (or the wrong one cited).
faithfulness       LLM judge, sees ONLY retrieved chunks + answer (never
                   the expected answer): are all factual claims supported
                   by the retrieved text? SUPPORTED / PARTIAL / UNSUPPORTED.
abstention         out_of_corpus cases must abstain (safety), answerable
                   cases must NOT abstain (usefulness). Both are reported —
                   optimizing only the first produces a system that refuses
                   everything.
cross-lingual      all of the above, sliced to cases where the expected
                   source language differs from the query language.

Results are written incrementally to eval/results.jsonl; re-running skips
completed case ids (crash-safe, and lets you extend the testset cheaply).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from rag import RagPipeline, ANSWER_MODEL  # noqa: E402

RESULTS = ROOT / "eval" / "results.jsonl"
REPORT = ROOT / "eval" / "report.md"

JUDGE_MODEL = ANSWER_MODEL  # same tier as the answerer; override if desired

JUDGE_SYSTEM = """\
You are grading whether an answer is FAITHFUL to its sources. You will \
see numbered SOURCES (the only material the answering system had) and an \
ANSWER. Check every factual claim in the answer (fees, documents, steps, \
durations, conditions, names of laws) against the sources.

Reply with ONLY a JSON object:
{"verdict": "SUPPORTED" | "PARTIAL" | "UNSUPPORTED",
 "unsupported_claims": ["...", ...]}

SUPPORTED   = every factual claim appears in the sources.
PARTIAL     = core claims supported, but some details are not in the sources.
UNSUPPORTED = one or more central claims are absent from or contradict the sources."""


def norm_url(u: str) -> str:
    u = re.sub(r"^https?://", "", u.lower())
    return u.removeprefix("www.").rstrip("/")


def url_match(url: str, expected: list[str]) -> bool:
    n = norm_url(url)
    return any(norm_url(e) in n or n in norm_url(e) for e in expected)


def judge_faithfulness(client, retrieved: list[dict], answer: str) -> dict:
    source_block = "\n\n".join(
        f"[{n}] {c['text']}" for n, c in enumerate(retrieved, 1))
    msg = client.messages.create(
        model=JUDGE_MODEL, max_tokens=500, system=JUDGE_SYSTEM,
        messages=[{"role": "user",
                   "content": f"SOURCES:\n\n{source_block}\n\nANSWER:\n\n{answer}"}])
    text = msg.content[0].text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"verdict": "JUDGE_ERROR", "unsupported_claims": [text[:200]]}


def evaluate_case(rag, case: dict, k: int, use_judge: bool) -> dict:
    out = rag.answer(case["question"], k=k)
    res = {
        "id": case["id"], "question": case["question"],
        "language": case["language"], "type": case["type"],
        "cross_lingual": case.get("cross_lingual", False),
        "origin": case.get("origin", ""),
        "abstained": out["abstained"],
        "answer": out["answer"],
        "retrieved": [(c["chunk_id"], round(c["score"], 3)) for c in out["retrieved"]],
        "cited_urls": [s["url"] for s in out["sources"]],
    }

    if case["type"] == "answerable":
        res["retrieval_hit"] = any(
            url_match(c["source_url"], case["expected_urls"])
            for c in out["retrieved"])
        res["false_abstention"] = out["abstained"]
        if not out["abstained"]:
            res["citation_correct"] = any(
                url_match(u, case["expected_urls"]) for u in res["cited_urls"])
            if use_judge:
                res["faithfulness"] = judge_faithfulness(
                    rag.client, out["retrieved"], out["answer"])
    else:  # out_of_corpus
        res["abstention_correct"] = out["abstained"]

    return res


# --------------------------------------------------------------------------
# Aggregation & report
# --------------------------------------------------------------------------

def rate(xs: list[bool]) -> str:
    return f"{sum(xs)}/{len(xs)}" if xs else "—"


def aggregate(results: list[dict]) -> list[tuple[str, dict[str, str]]]:
    """Rows of (metric, {slice: value}). Slices: overall / en / ar / cross."""
    slices = {
        "overall": results,
        "EN": [r for r in results if r["language"] == "en"],
        "AR": [r for r in results if r["language"] == "ar"],
        "cross-lingual": [r for r in results if r["cross_lingual"]],
    }
    rows = []

    def collect(metric, pick):
        row = {}
        for name, rs in slices.items():
            row[name] = rate([pick(r) for r in rs if pick(r) is not None])
        rows.append((metric, row))

    collect("retrieval hit@k", lambda r: r.get("retrieval_hit"))
    collect("citation correct", lambda r: r.get("citation_correct"))
    collect("faithfulness SUPPORTED",
            lambda r: (r["faithfulness"]["verdict"] == "SUPPORTED")
            if "faithfulness" in r else None)
    collect("abstained on out-of-corpus", lambda r: r.get("abstention_correct"))
    collect("false abstention (answerable)", lambda r: r.get("false_abstention"))
    return rows


def render_report(rows, results) -> str:
    lines = ["# Evaluation report", "",
             f"cases: {len(results)} | judge: {JUDGE_MODEL}", "",
             "| metric | overall | EN | AR | cross-lingual |",
             "|---|---|---|---|---|"]
    for metric, row in rows:
        lines.append(f"| {metric} | {row['overall']} | {row['EN']} "
                     f"| {row['AR']} | {row['cross-lingual']} |")

    # every non-perfect case listed for inspection — aggregates hide bugs
    lines += ["", "## Cases needing attention", ""]
    for r in results:
        problems = []
        if r.get("retrieval_hit") is False: problems.append("retrieval MISS")
        if r.get("citation_correct") is False: problems.append("bad citation")
        if r.get("false_abstention"): problems.append("FALSE abstention")
        if r.get("abstention_correct") is False: problems.append("failed to abstain")
        if r.get("faithfulness", {}).get("verdict") in ("PARTIAL", "UNSUPPORTED"):
            problems.append(f"faithfulness {r['faithfulness']['verdict']}: "
                            + "; ".join(r["faithfulness"]["unsupported_claims"][:2]))
        if problems:
            lines.append(f"- **{r['id']}** ({r['origin']}) {r['question'][:70]}")
            for p in problems:
                lines.append(f"  - {p}")
    return "\n".join(lines) + "\n"


def rescore(results: list[dict], cases_by_id: dict[str, dict]) -> list[dict]:
    """Recompute the DETERMINISTIC metrics from stored raw results against
    the CURRENT testset labels — no API calls. Lets you fix expected_urls
    or reclassify cases and re-report for free. Judge verdicts (the only
    LLM-derived metric) are carried over untouched."""
    chunks_path = ROOT / "data" / "chunks" / "chunks.jsonl"
    url_of = {c["chunk_id"]: c["source_url"]
              for c in map(json.loads, chunks_path.read_text(encoding="utf-8").splitlines())}
    out = []
    for r in results:
        case = cases_by_id.get(r["id"])
        if case is None:
            continue  # case removed from testset — drop its result
        r = dict(r)
        r["type"], r["cross_lingual"] = case["type"], case.get("cross_lingual", False)
        retrieved_urls = [url_of.get(cid, "") for cid, _ in r["retrieved"]]
        for key in ("retrieval_hit", "citation_correct", "false_abstention",
                    "abstention_correct"):
            r.pop(key, None)
        if case["type"] == "answerable":
            r["retrieval_hit"] = any(url_match(u, case["expected_urls"])
                                     for u in retrieved_urls if u)
            r["false_abstention"] = r["abstained"]
            if not r["abstained"]:
                r["citation_correct"] = any(url_match(u, case["expected_urls"])
                                            for u in r["cited_urls"])
        else:
            r["abstention_correct"] = r["abstained"]
        out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default=str(ROOT / "eval" / "testset.jsonl"))
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--no-judge", action="store_true",
                    help="skip LLM faithfulness judging (deterministic metrics only)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rescore", action="store_true",
                    help="recompute metrics from stored results against current "
                         "testset labels — zero API calls, does not run new cases")
    ap.add_argument("--results", default=None,
                    help="results file (default: eval/results.jsonl). Use a "
                         "separate file for held-out sets so they never mix "
                         "with tuning-set results")
    ap.add_argument("--report", default=None,
                    help="report file (default: eval/report.md)")
    args = ap.parse_args()

    global RESULTS, REPORT
    if args.results:
        RESULTS = Path(args.results)
    if args.report:
        REPORT = Path(args.report)

    if args.rescore:
        cases_by_id = {c["id"]: c for c in map(
            json.loads, Path(args.testset).read_text(encoding="utf-8").splitlines())}
        stored = [json.loads(l) for l in RESULTS.read_text(encoding="utf-8").splitlines()]
        results = rescore(stored, cases_by_id)
        RESULTS.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results)
                           + "\n", encoding="utf-8")
        report = render_report(aggregate(results), results)
        REPORT.write_text(report, encoding="utf-8")
        print(report.split("## Cases")[0])
        missing = [i for i in cases_by_id if i not in {r['id'] for r in results}]
        if missing:
            print(f"{len(missing)} case(s) have no stored result — run without "
                  f"--rescore to evaluate them: {missing[:6]}")
        print(f"full report -> {REPORT}")
        return

    cases = [json.loads(l) for l in Path(args.testset).read_text(encoding="utf-8").splitlines()]
    if args.limit:
        cases = cases[: args.limit]

    done_ids = set()
    if RESULTS.exists():
        done_ids = {json.loads(l)["id"] for l in RESULTS.read_text(encoding="utf-8").splitlines()}
        print(f"resuming: {len(done_ids)} cases already in {RESULTS.name}")

    rag = RagPipeline()
    with RESULTS.open("a", encoding="utf-8") as f:
        for i, case in enumerate(cases):
            if case["id"] in done_ids:
                continue
            res = evaluate_case(rag, case, args.k, use_judge=not args.no_judge)
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            status = "ABSTAIN" if res["abstained"] else (
                "hit" if res.get("retrieval_hit") else "MISS" if res.get("retrieval_hit") is False else "-")
            print(f"  [{i+1}/{len(cases)}] {status:8s} {case['question'][:60]}")

    results = [json.loads(l) for l in RESULTS.read_text(encoding="utf-8").splitlines()]
    rows = aggregate(results)
    report = render_report(rows, results)
    REPORT.write_text(report, encoding="utf-8")
    print("\n" + report.split("## Cases")[0])
    print(f"full report -> {REPORT}")


if __name__ == "__main__":
    main()

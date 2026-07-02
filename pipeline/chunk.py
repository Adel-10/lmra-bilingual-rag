"""
Phase 2a — section-aware chunking: clean markdown -> retrieval chunks.

Design
------
Chunks follow the DOCUMENT'S OWN STRUCTURE, not a fixed-size window:

  * Split at `##` section boundaries (the site's h3: "Service Conditions",
    "Required Documents", "Service Fees", FAQ questions, law articles...).
  * Tables are ATOMIC. A fee table is only meaningful as rows+header; a
    window that cuts it mid-row produces unanswerable garbage. If a table
    alone exceeds the size cap it is split BETWEEN rows, repeating the
    caption and header row so each fragment stays self-describing.
  * Tiny neighbouring sections are merged; oversized sections split at
    paragraph boundaries.
  * Every chunk is prefixed with a breadcrumb ("<doc title> › <section>")
    — "BHD 86" embeds uselessly unless the vector also encodes WHICH
    service and section it belongs to.
  * Language is detected PER CHUNK (Arabic codepoint ratio), because some
    law docs genuinely mix English and Arabic in one page.

Size cap is in characters (~2000 chars ≈ 400-600 tokens) so the chunker
carries no dependency on any particular embedding model's tokenizer.

Output:  data/chunks/chunks.jsonl   (one JSON object per chunk)
Usage:   python pipeline/chunk.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "data" / "chunks"

MAX_CHARS = 2000   # hard cap per chunk (approx 400-600 tokens)
MIN_CHARS = 300    # sections smaller than this get merged with neighbours


# --------------------------------------------------------------------------
# Parsing the clean markdown
# --------------------------------------------------------------------------

def parse_doc(path: Path) -> tuple[dict, str]:
    """Split a clean .md file into (frontmatter dict, body)."""
    text = path.read_text(encoding="utf-8")
    _, fm, body = text.split("---", 2)
    meta = {}
    for line in fm.strip().splitlines():
        key, _, val = line.partition(":")
        meta[key.strip()] = val.strip()
    meta["service_groups"] = json.loads(meta.get("service_groups", "[]"))
    meta["redundant_with_pair"] = meta.get("redundant_with_pair") == "true"
    return meta, body.strip()


def split_blocks(body: str) -> list[dict]:
    """
    Split markdown into typed blocks: heading / table / text.
    A table block = its optional **caption** line + contiguous |...| lines.
    """
    blocks, buf = [], []

    def flush():
        if buf:
            blocks.append({"type": "text", "text": "\n".join(buf).strip()})
            buf.clear()

    lines = body.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        h = re.match(r"^(#{1,4})\s+(.*)", line)
        if h:
            flush()
            blocks.append({"type": "heading", "level": len(h.group(1)),
                           "text": h.group(2).strip()})
            i += 1
        elif line.startswith("|"):
            flush()
            # caption convention from extract.py: a "**...**" line right
            # above the table — pull it out of the previous text block
            caption = ""
            if blocks and blocks[-1]["type"] == "text":
                prev_lines = blocks[-1]["text"].splitlines()
                if prev_lines and re.fullmatch(r"\*\*.+\*\*", prev_lines[-1]):
                    caption = prev_lines.pop()
                    blocks[-1]["text"] = "\n".join(prev_lines).strip()
                    if not blocks[-1]["text"]:
                        blocks.pop()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            blocks.append({"type": "table", "caption": caption, "rows": rows})
        else:
            buf.append(line)
            i += 1
    flush()
    return [b for b in blocks if b["type"] != "text" or b["text"]]


# --------------------------------------------------------------------------
# Assembling chunks
# --------------------------------------------------------------------------

def table_text(caption: str, rows: list[str]) -> str:
    return (caption + "\n\n" if caption else "") + "\n".join(rows)


def split_table(caption: str, rows: list[str], budget: int) -> list[str]:
    """Split an oversized table between rows; every part repeats caption +
    header + separator so it remains a valid, self-describing table."""
    header, sep, body = rows[0], rows[1], rows[2:]
    head = (caption + "\n\n" if caption else "") + header + "\n" + sep
    parts, cur = [], []
    for row in body:
        candidate = head + "\n" + "\n".join(cur + [row])
        if cur and len(candidate) > budget:
            parts.append(head + "\n" + "\n".join(cur))
            cur = [row]
        else:
            cur.append(row)
    if cur:
        parts.append(head + "\n" + "\n".join(cur))
    return parts


def split_text(text: str, budget: int) -> list[str]:
    """Split long text at paragraph boundaries (single items never split)."""
    paras = text.split("\n\n")
    parts, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > budget:
            parts.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        parts.append(cur)
    return parts


def detect_language(text: str) -> str:
    letters = re.findall(r"[A-Za-z؀-ۿ]", text)
    if not letters:
        return "unknown"
    ar = len(re.findall(r"[؀-ۿ]", text))
    return "ar" if ar / len(letters) > 0.5 else "en"


def chunk_document(meta: dict, body: str) -> list[dict]:
    """Produce chunks for one document."""
    blocks = split_blocks(body)
    title = meta["title"]

    # ---- group blocks into sections at heading boundaries ----------------
    # section = {"path": [h1, h2, ...], "content": [rendered block strings]}
    sections: list[dict] = []
    path: list[str] = []
    current = {"path": [], "content": []}

    def push():
        nonlocal current
        if current["content"]:
            sections.append(current)
        current = {"path": list(path), "content": []}

    for b in blocks:
        if b["type"] == "heading":
            push()
            path = path[: b["level"] - 1] + [b["text"]]
            current = {"path": list(path), "content": []}
        elif b["type"] == "table":
            current["content"].append(
                {"table": True, "caption": b["caption"], "rows": b["rows"]})
        else:
            current["content"].append({"table": False, "text": b["text"]})
    push()

    # ---- render sections, splitting oversized ones -----------------------
    def breadcrumb(sec_path: list[str]) -> str:
        crumbs = [title] + [p for p in sec_path if p and p != title]
        return " › ".join(dict.fromkeys(crumbs))  # dedupe, keep order

    pieces: list[tuple[str, str]] = []  # (breadcrumb, content)
    for sec in sections:
        crumb = breadcrumb(sec["path"])
        budget = MAX_CHARS - len(crumb) - 2
        parts, cur = [], ""

        def flush_cur():
            nonlocal cur
            if cur.strip():
                parts.append(cur.strip())
            cur = ""

        for item in sec["content"]:
            if item["table"]:
                t = table_text(item["caption"], item["rows"])
                if len(cur) + len(t) + 2 <= budget:
                    cur = f"{cur}\n\n{t}" if cur else t
                else:
                    flush_cur()
                    if len(t) <= budget:
                        cur = t
                    else:  # table alone exceeds budget: split between rows
                        parts.extend(split_table(item["caption"], item["rows"], budget))
            else:
                for piece in split_text(item["text"], budget):
                    if len(cur) + len(piece) + 2 <= budget:
                        cur = f"{cur}\n\n{piece}" if cur else piece
                    else:
                        flush_cur()
                        cur = piece
        flush_cur()
        pieces.extend((crumb, p) for p in parts)

    # ---- merge tiny neighbouring pieces (same doc, in order) -------------
    # CRITICAL detail: only the first piece's breadcrumb survives a merge,
    # so a merged-in piece from a DIFFERENT section must carry its own
    # heading INSIDE the content — otherwise the heading text (e.g. the
    # entire question of a small FAQ item) disappears from the index.
    # This exact bug made "How can I request the local transfer of a
    # domestic employee?" unretrievable in the first eval run.
    merged: list[tuple[str, str]] = []
    for crumb, p in pieces:
        if (merged and len(merged[-1][1]) < MIN_CHARS
                and len(merged[-1][1]) + len(p) + 2 <= MAX_CHARS):
            prev_crumb, prev = merged[-1]
            if crumb != prev_crumb:
                own_heading = crumb.split(" › ")[-1]
                p = f"### {own_heading}\n\n{p}"
            merged[-1] = (prev_crumb, f"{prev}\n\n{p}")
        else:
            merged.append((crumb, p))

    # ---- final chunk records ---------------------------------------------
    chunks = []
    for n, (crumb, content) in enumerate(merged):
        text = f"{crumb}\n\n{content}"
        chunks.append({
            "chunk_id": f"{meta['doc_id']}#{n}",
            "doc_id": meta["doc_id"],
            "title": title,
            "section": crumb,
            "text": text,
            "language": detect_language(text),
            "url_language": meta["language"],
            "source_url": meta["source_url"],
            "service_groups": meta["service_groups"],
            "kind": meta["kind"],
            "last_update": meta.get("last_update"),
            "pair_doc_id": meta.get("pair_id"),
            "chars": len(text),
        })
    return chunks


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_chunks, skipped = [], 0
    for path in sorted(CLEAN.glob("*.md")):
        meta, body = parse_doc(path)
        if meta["redundant_with_pair"]:
            skipped += 1  # Arabic-only law served on /en/ URL — indexed via its _ar pair
            continue
        all_chunks.extend(chunk_document(meta, body))

    out = OUT / "chunks.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    langs = {}
    for c in all_chunks:
        langs[c["language"]] = langs.get(c["language"], 0) + 1
    sizes = sorted(c["chars"] for c in all_chunks)
    print(f"chunks: {len(all_chunks)} from {len(list(CLEAN.glob('*.md'))) - skipped} docs "
          f"({skipped} redundant docs skipped)")
    print(f"languages: {langs}")
    print(f"size chars — min {sizes[0]}, median {sizes[len(sizes)//2]}, "
          f"p95 {sizes[int(len(sizes)*.95)]}, max {sizes[-1]}")
    over = sum(1 for s in sizes if s > MAX_CHARS)
    print(f"chunks over MAX_CHARS: {over}")


if __name__ == "__main__":
    main()

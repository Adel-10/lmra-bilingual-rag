"""
Phase 1 — extraction & cleaning: raw LMRA HTML -> clean structured markdown.

Design
------
We do NOT use a heuristic content extractor (trafilatura/readability). Those
guess where the main content is — the right tool for a thousand unknown
sites, the wrong tool for one known CMS. Every LMRA page wraps its real
content in `id="page_content"`, so we scope there deterministically and
remove the known-junk elements by name. Deterministic extraction fails
loudly (missing anchor -> hard error) rather than silently degrading,
which is the failure mode we want for civic information.

Structure is preserved as markdown because it carries meaning:
  - h2 (service title) / h3 (sections: Conditions, Required Documents,
    Fees, Processing Time, ...) / h4 (subsections) -> #/##/### headings.
    These become the chunk boundaries in Phase 2.
  - fee/document tables -> markdown tables, caption kept as a bold line
    above (a fee table flattened into prose is useless for answering
    "how much does X cost?").
  - alert boxes -> blockquotes (they hold real conditions, e.g. senior
    citizen discounts).

Output
------
    data/clean/{id}_{lang}.md      one per page, YAML-ish frontmatter
    data/clean/index.json          all metadata, EN/AR pairs linked
    data/clean/pdf_links.json      inventory of PDFs seen in content
                                   (for the informational-PDF whitelist)

Usage:  python pipeline/extract.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

ROOT = Path(__file__).resolve().parent.parent
RAW_HTML = ROOT / "data" / "raw" / "html"
CLEAN = ROOT / "data" / "clean"
MANIFEST = ROOT / "data" / "raw" / "manifest.json"

# Elements (by CSS class) that live inside page_content but are chrome.
# CAUTION: grid-container-dependents must NOT be here — it wraps the tab
# row AND main-content-dependents (the actual service content).
JUNK_CLASSES = (
    "tabs-test",       # service-group tab row (nav, not content)
    "btn-group",       # "Pages Index" dropdown
    "clearfix_btn",
    "breadcrumb",
    "share-buttons",
)
JUNK_TAGS = ("script", "style", "img", "button", "form", "iframe", "input")

# "Rate Our Service" heading marks the start of trailing boilerplate.
RATE_MARKERS = ("rate our service", "قيم الخدمة", "قيّم الخدمة", "تقييم الخدمة")

# The contact boilerplate alert repeats on every page — noise for retrieval.
CONTACT_MARKERS = ("17506055", "call centre", "مركز الاتصال")

# Two date styles exist: service pages use "18-06-2026", law pages use
# "Monday 12 June 2023" (weekday optional word, then day month year).
LAST_UPDATE_RE = re.compile(
    r"(?:Last Update|آخر تحديث)\s*[:：]?\s*(?:[^\s\d]+\s+)?"
    r"(\d{1,2}-\d{1,2}-\d{4}|\d{1,2}\s+[^\s\d]+\s+\d{4})"
)


# --------------------------------------------------------------------------
# HTML -> markdown (small custom walker; tables are the point)
# --------------------------------------------------------------------------

def cell_text(cell: Tag) -> str:
    """Table-cell text: single line, pipes escaped so the table stays valid."""
    return re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).replace("|", "\\|")


def table_to_md(table: Tag) -> str:
    """
    Convert a table to markdown. Caption becomes a bold line above.
    Header = first row if it uses <th> (or thead); otherwise a generic
    header is synthesized because markdown tables require one.
    """
    lines: list[str] = []
    caption = table.find("caption")
    if caption:
        lines.append(f"**{cell_text(caption)}**")
        lines.append("")

    rows = [tr for tr in table.find_all("tr") if tr.find(["td", "th"])]
    if not rows:
        return ""

    header_cells = rows[0].find_all(["th", "td"])
    has_header = rows[0].find("th") is not None
    ncols = max(len(r.find_all(["td", "th"])) for r in rows)

    if has_header:
        header = [cell_text(c) for c in header_cells]
        body_rows = rows[1:]
    else:
        header = [""] * ncols  # blank header keeps the table renderable
        body_rows = rows

    header += [""] * (ncols - len(header))
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * ncols)
    for tr in body_rows:
        cells = [cell_text(c) for c in tr.find_all(["td", "th"])]
        cells += [""] * (ncols - len(cells))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def inline_text(el: Tag) -> str:
    """Flatten an element to one line, keeping <a> hrefs as markdown links."""
    # If the element IS a link, emit it whole — the descendant walk below
    # only handles links nested inside el, not el itself.
    if el.name == "a" and el.get("href"):
        label = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        if not label:
            return ""
        href = el["href"]
        return label if href.startswith("#") else f"[{label}]({href})"
    parts: list[str] = []
    for node in el.descendants:
        if isinstance(node, NavigableString):
            parent = node.parent
            # skip text that a handled <a> will emit itself
            if parent.name == "a" and parent.get("href"):
                continue
            parts.append(str(node))
        elif node.name == "a" and node.get("href"):
            label = node.get_text(" ", strip=True)
            if not label:
                continue
            href = node["href"]
            if href.startswith("#"):   # same-page anchor — keep text only
                parts.append(label)
            else:
                parts.append(f"[{label}]({href})")
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def block_to_md(el: Tag, depth: int = 0) -> list[str]:
    """Recursively convert a block element to markdown lines."""
    out: list[str] = []
    name = el.name

    if name in ("h2", "h3", "h4", "h5"):
        level = {"h2": "#", "h3": "##", "h4": "###", "h5": "####"}[name]
        # inline_text (not get_text) so PDF links inside headings keep
        # their URLs — several pages are pure PDF-pointer pages
        text = inline_text(el)
        if text:
            out += [f"{level} {text}", ""]
    elif name == "p":
        text = inline_text(el)
        if text:
            out += [text, ""]
    elif name == "table":
        md = table_to_md(el)
        if md:
            out += [md, ""]
    elif name in ("ul", "ol"):
        ordered = name == "ol"
        for i, li in enumerate(el.find_all("li", recursive=False), 1):
            # render the li's own inline content, then any nested lists
            nested = li.find_all(["ul", "ol"], recursive=False)
            for sub in nested:
                sub.extract()
            text = inline_text(li)
            bullet = f"{i}." if ordered else "-"
            if text:
                out.append("  " * depth + f"{bullet} {text}")
            for sub in nested:
                out += block_to_md(sub, depth + 1)
        out.append("")
    elif name == "div":
        classes = " ".join(el.get("class", []))
        if "alert" in classes:
            text = inline_text(el)
            if text and not any(m in text.lower() for m in CONTACT_MARKERS):
                out += [f"> {text}", ""]
        else:  # transparent container — recurse into children
            for child in el.find_all(recursive=False):
                out += block_to_md(child, depth)
    else:
        # Unknown element. Recurse only if it contains real block children;
        # otherwise render it inline. (A bare <a> wrapping a <span> badge —
        # the legal category lists — must NOT be recursed into, or the
        # anchor's own text is lost and only the badge survives.)
        BLOCK_TAGS = {"p", "div", "table", "ul", "ol", "h2", "h3", "h4", "h5"}
        children = el.find_all(recursive=False)
        if any(c.name in BLOCK_TAGS for c in children):
            for child in children:
                out += block_to_md(child, depth)
        else:
            text = inline_text(el)
            if text:
                out += [text, ""]
    return out


# --------------------------------------------------------------------------
# Per-page extraction
# --------------------------------------------------------------------------

def extract_page(html: str, url: str) -> tuple[str, dict]:
    """Return (markdown, info) for one raw page. Raises if structure breaks."""
    soup = BeautifulSoup(html, "html.parser")

    # metadata available anywhere on the page
    m = LAST_UPDATE_RE.search(soup.get_text(" ", strip=True))
    last_update = m.group(1) if m else None

    content = soup.find(id="page_content")
    if content is None:
        raise ValueError(f"no #page_content anchor in {url}")

    # 1. drop junk elements
    for tag in content.find_all(JUNK_TAGS):
        tag.decompose()
    for cls in JUNK_CLASSES:
        for tag in content.find_all(class_=cls):
            tag.decompose()
    # action buttons ("Start Service", "How to Videos") are links styled
    # as buttons — navigation, not content
    for a in content.find_all("a", class_="btn"):
        a.decompose()

    # 2. truncate at "Rate Our Service" — everything after is boilerplate
    for h in content.find_all(["h3", "h4"]):
        if h.get_text(strip=True).lower().rstrip(":") in RATE_MARKERS:
            for sib in list(h.find_all_next()):
                sib.extract()
            h.extract()
            break

    # 3. inventory PDF links BEFORE conversion (for the PDF whitelist)
    pdf_links = sorted(
        {a["href"] for a in content.find_all("a", href=True)
         if a["href"].lower().endswith(".pdf")}
    )

    # 4. convert to markdown
    lines: list[str] = []
    for child in content.find_all(recursive=False):
        lines += block_to_md(child)
    md = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

    return md, {"last_update": last_update, "pdf_links": pdf_links}


def detect_language(md: str) -> str:
    """Content language by Arabic-codepoint ratio. Needed because LMRA
    serves the ARABIC text on the /en/ URL for Arabic-only laws — the URL
    language and the content language genuinely disagree there."""
    letters = re.findall(r"[A-Za-z؀-ۿ]", md)
    if not letters:
        return "unknown"
    ar = len(re.findall(r"[؀-ۿ]", md))
    return "ar" if ar / len(letters) > 0.5 else "en"


def similarity(a: str, b: str) -> float:
    """Char-trigram Jaccard on link-stripped text — cheap, transparent
    near-duplicate detector for EN/AR pair bodies."""
    def norm(t: str) -> str:
        t = re.sub(r"\(https?://[^)]+\)", "", t)  # /en/ vs /ar/ link noise
        return re.sub(r"\s+", " ", t).strip()
    a, b = norm(a), norm(b)
    ga = {a[i:i + 3] for i in range(len(a) - 2)}
    gb = {b[i:i + 3] for i in range(len(b) - 2)}
    return len(ga & gb) / max(1, len(ga | gb))


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    CLEAN.mkdir(parents=True, exist_ok=True)
    docs: dict[str, dict] = {}   # doc_id -> {md, meta}
    pdf_inventory, failures, warnings = {}, [], []

    # ---- pass 1: extract everything into memory -------------------------
    for entry in manifest:
        name = entry.get("slug") or str(entry["id"])
        for lang in ("en", "ar"):
            raw = RAW_HTML / f"{name}_{lang}.html"
            if not raw.exists():
                failures.append((name, lang, "raw file missing"))
                continue
            url = entry[f"url_{lang}"]
            try:
                md, info = extract_page(raw.read_text(encoding="utf-8"), url)
            except ValueError as exc:
                failures.append((name, lang, str(exc)))
                continue
            if len(md) < 200:
                # Thin pages are usually legitimate (PDF-pointer pages,
                # one-paragraph descriptions) — warn, don't fail.
                warnings.append((name, lang, f"only {len(md)} chars extracted"))

            doc_id = f"{name}_{lang}"
            docs[doc_id] = {
                "md": md,
                "meta": {
                    "doc_id": doc_id, "id": entry["id"], "title": entry["title"],
                    "language": lang, "source_url": url,
                    "service_groups": entry["service_groups"],
                    "last_update": info["last_update"], "kind": entry["kind"],
                    "pair_id": f"{name}_{'ar' if lang == 'en' else 'en'}",
                    "chars": len(md),
                },
            }
            if info["pdf_links"]:
                pdf_inventory[doc_id] = info["pdf_links"]

    # ---- pass 2: content language + pair-redundancy flags ---------------
    # If an EN-URL doc actually contains Arabic AND is near-identical to
    # its _ar pair (Arabic-only laws), mark it redundant so the chunker
    # skips it — otherwise the same Arabic text would be indexed twice and
    # distort retrieval metrics. Below-threshold mixed docs are kept.
    for doc_id, d in docs.items():
        meta = d["meta"]
        meta["content_language"] = detect_language(d["md"])
        meta["redundant_with_pair"] = False
        if (meta["content_language"] != meta["language"]
                and meta["pair_id"] in docs):
            sim = similarity(d["md"], docs[meta["pair_id"]]["md"])
            meta["pair_similarity"] = round(sim, 3)
            if sim > 0.9:
                meta["redundant_with_pair"] = True

    # ---- pass 3: write files + index -------------------------------------
    index = []
    for doc_id, d in docs.items():
        meta = d["meta"]
        header = "\n".join([
            "---",
            f"doc_id: {meta['doc_id']}",
            f"title: {meta['title']}",
            f"language: {meta['language']}",
            f"content_language: {meta['content_language']}",
            f"redundant_with_pair: {str(meta['redundant_with_pair']).lower()}",
            f"source_url: {meta['source_url']}",
            f"service_groups: {json.dumps(meta['service_groups'])}",
            f"last_update: {meta['last_update']}",
            f"pair_id: {meta['pair_id']}",
            f"kind: {meta['kind']}",
            "---",
        ])
        (CLEAN / f"{doc_id}.md").write_text(
            header + "\n\n" + d["md"] + "\n", encoding="utf-8")
        index.append(meta)

    (CLEAN / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    (CLEAN / "pdf_links.json").write_text(
        json.dumps(pdf_inventory, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"extracted {len(index)} documents -> {CLEAN}")
    print(f"documents with PDF links: {len(pdf_inventory)}")
    if warnings:
        print("\nwarnings (thin pages, usually legitimate):")
        for name, lang, why in warnings:
            print(f"  {name}_{lang}: {why}")
    if failures:
        print("\nFAILURES:")
        for name, lang, why in failures:
            print(f"  {name}_{lang}: {why}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Phase 0 — LMRA corpus crawler (lmra.gov.bh only).

Design
------
Two-stage, manifest-driven crawl (NOT a recursive spider):

  1. DISCOVER: fetch the 7 E-Services hub pages, extract only the
     "Details" leaf-page links into a manifest. Hub pages are navigation
     menus — we use them to find links, never as content.
  2. FETCH: download exactly the manifest's leaf pages, English and
     Arabic (same page ID, /en/ <-> /ar/), plus the FAQ and Legislations
     bonus sources. Save raw HTML untouched; cleaning happens in Phase 1.

Why manifest-driven? A recursive crawler on a government site is how you
accidentally ingest login portals, nav pages, or external sites. The
hub->leaf structure here is clean and known, so we enumerate it explicitly.
The manifest (data/raw/manifest.json) doubles as a provenance record:
re-run discovery later and diff it to detect new or removed services.

Politeness: realistic browser User-Agent, 1.5 s delay between requests,
sequential fetching (no concurrency against a small gov site), retries
with exponential backoff.

Usage
-----
    pip install requests beautifulsoup4
    python crawl.py discover   # build/refresh the leaf-page manifest
    python crawl.py fetch      # download everything in the manifest
    python crawl.py all        # both

Output layout (relative to repo root)
-------------------------------------
    data/raw/manifest.json         one entry per document (see schema below)
    data/raw/html/{id}_{lang}.html raw leaf pages
    data/raw/pdf/{name}.pdf        informational PDFs (guides, laws)
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BASE = "https://lmra.gov.bh"

# The 7 E-Services hub pages. Keys are the page IDs, values are labels used
# as `service_group` metadata on every leaf discovered from that hub.
HUBS: dict[int, str] = {
    213: "Establishment Employer",
    224: "Domestic Employer",
    210: "Expatriate Employee",
    424: "Registered Worker",
    621: "Golden Residency",
    592: "Licensed CRs",
    593: "General Services",
}

# Bonus sources (not hubs, not discovered leaves — fetched directly).
FAQ_PATH = "/{lang}/faq"
LEGISLATIONS_ID = 221  # authoritative legal layer; some docs Arabic-only

# Never ingest anything whose URL contains one of these substrings:
# logins, appointment/booking systems, external portals, other sites.
BLACKLIST = (
    "EMS_Web",
    "ams.lmra.gov.bh",
    "services.bahrain.bh/wps/portal",
    "wafid.com",
)

# Leaf pages look like /en/page/show/<id> (language swaps to /ar/).
LEAF_RE = re.compile(r"/(en|ar)/page/show/(\d+)")

# Site-chrome pages that appear in the header/footer of EVERY page
# (Contact, About, Terms, Careers, Sitemap, ...). They match LEAF_RE but
# are not service content, so discovery must skip them. NOTE: 631 (Wage
# Protection System) also appears in the global nav but is additionally a
# real "Details" tile on the Establishment Employer hub — it stays IN.
CHROME_IDS = {
    4,    # Terms and Conditions
    95,   # Contact Us
    171,  # Accessibility
    226,  # Careers
    230,  # Site Statistic
    446,  # Sitemap
    453,  # E-Services breadcrumb page
    500,  # Remote Services portal entry ("Start" target, not content)
    506,  # Protection Centre (global nav)
    600,  # About the Authority (global nav)
    616,  # Inspection E-Services portal entry ("Start" target)
    647,  # Virtual Centre (global nav)
}

# Realistic desktop UA — we are not disguising anything nefarious, just
# avoiding naive bot filters that block default python-requests UAs.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept-Language": "en,ar;q=0.9",
}

DELAY_SECONDS = 1.5
RETRIES = 3

ROOT = Path(__file__).resolve().parent.parent  # repo root (lmra-rag/)
RAW = ROOT / "data" / "raw"
MANIFEST_PATH = RAW / "manifest.json"


# --------------------------------------------------------------------------
# HTTP helper
# --------------------------------------------------------------------------

def get(url: str, session: requests.Session) -> requests.Response:
    """GET with polite delay and simple exponential-backoff retries."""
    for attempt in range(RETRIES):
        time.sleep(DELAY_SECONDS)  # delay BEFORE every request, always
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            wait = 2 ** attempt * 5
            print(f"  ! {exc} — retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"gave up on {url}")


def blacklisted(url: str) -> bool:
    return any(bad in url for bad in BLACKLIST)


# --------------------------------------------------------------------------
# Stage 1 — discovery
# --------------------------------------------------------------------------

def discover(session: requests.Session) -> list[dict]:
    """
    Fetch each hub (English version only — page IDs are language-independent,
    so discovering in one language is enough) and collect leaf-page links.

    We take any on-site /en/page/show/<id> link, from both the "Sub Pages"
    sidebar and the service-tile "Details" buttons, then drop hub IDs and
    blacklisted URLs. The two sources overlap heavily; the union catches
    tiles that are missing from the sidebar (e.g. Domestic page 678).
    """
    entries: dict[int, dict] = {}

    def content_region(html: str):
        """Scope link extraction to #page_content — global nav and footer
        link to pages like WPS (631) from EVERY page, which would wrongly
        add every hub's group to them."""
        soup = BeautifulSoup(html, "html.parser")
        return soup.find(id="page_content") or soup

    for hub_id, group in HUBS.items():
        url = f"{BASE}/en/page/show/{hub_id}"
        print(f"discovering: {group} ({url})")
        soup = content_region(get(url, session).text)

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if blacklisted(href):
                continue
            m = LEAF_RE.search(href)
            if not m:
                continue
            leaf_id = int(m.group(2))
            if leaf_id in HUBS:          # hub linking to another hub — skip
                continue
            if leaf_id in CHROME_IDS:    # header/footer chrome — skip
                continue
            title = (a.get("title") or a.get_text(" ", strip=True) or "").strip()
            # First hub to claim a leaf keeps it; record duplicates' groups too.
            if leaf_id not in entries:
                entries[leaf_id] = {
                    "id": leaf_id,
                    "title": title,
                    "service_groups": [group],
                    "url_en": f"{BASE}/en/page/show/{leaf_id}",
                    "url_ar": f"{BASE}/ar/page/show/{leaf_id}",
                    "kind": "leaf",
                }
            elif group not in entries[leaf_id]["service_groups"]:
                entries[leaf_id]["service_groups"].append(group)

    # Bonus sources, added explicitly (not discovered).
    entries[LEGISLATIONS_ID] = {
        "id": LEGISLATIONS_ID,
        "title": "Legislations",
        "service_groups": ["Legal"],
        "url_en": f"{BASE}/en/page/show/{LEGISLATIONS_ID}",
        "url_ar": f"{BASE}/ar/page/show/{LEGISLATIONS_ID}",
        "kind": "legislations",
    }
    # FAQ is two-level: /en/faq is itself a hub listing ~10 category pages,
    # and the category pages hold the actual Q&A inline (with per-question
    # "Last Update" dates and links to the service pages they reference —
    # ideal both as corpus and as evaluation ground truth). Discover the
    # category links live rather than hardcoding them.
    print(f"discovering: FAQ categories ({BASE}/en/faq)")
    faq_soup = content_region(get(f"{BASE}/en/faq", session).text)
    faq_cat_re = re.compile(r"/en/faq/category/(\d+)")
    for a in faq_soup.find_all("a", href=True):
        m = faq_cat_re.search(a["href"])
        if not m:
            continue
        cat = int(m.group(1))
        key = -cat  # negative IDs keep FAQ distinct from page/show IDs
        if key in entries:
            continue
        title = (a.get("title") or a.get_text(" ", strip=True)).strip()
        entries[key] = {
            "id": key,
            "slug": f"faq_cat{cat}",  # used as the on-disk filename
            "title": f"FAQ: {title}",
            "service_groups": ["FAQ"],
            "url_en": f"{BASE}/en/faq/category/{cat}",
            "url_ar": f"{BASE}/ar/faq/category/{cat}",
            "kind": "faq",
        }

    # Preserve fetch metadata from a previous run so that re-discovery
    # (e.g. checking for new services months later) never loses provenance.
    if MANIFEST_PATH.exists():
        old = {e["id"]: e for e in json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))}
        for leaf_id, entry in entries.items():
            if leaf_id in old and "fetched" in old[leaf_id]:
                entry["fetched"] = old[leaf_id]["fetched"]

    # Legislations (/221) is also a hub, not a leaf: the actual laws live
    # at /en/legal/category/{n} (grouped lists of laws/resolutions, PDFs
    # included) and two dedicated pages — 5 (LMRA Law) and 199 (Labour
    # Law). Discover both kinds from the live page.
    print(f"discovering: Legislations ({BASE}/en/page/show/{LEGISLATIONS_ID})")
    leg_soup = content_region(
        get(f"{BASE}/en/page/show/{LEGISLATIONS_ID}", session).text)
    legal_cat_re = re.compile(r"/en/legal/category/(\d+)")
    for a in leg_soup.find_all("a", href=True):
        href = a["href"]
        title = (a.get("title") or a.get_text(" ", strip=True)).strip()
        m = legal_cat_re.search(href)
        if m:
            cat = int(m.group(1))
            key = -(100 + cat)  # offset avoids colliding with FAQ negatives
            if key not in entries:
                entries[key] = {
                    "id": key, "slug": f"legal_cat{cat}",
                    "title": f"Legislations: category {cat}",
                    "service_groups": ["Legal"],
                    "url_en": f"{BASE}/en/legal/category/{cat}",
                    "url_ar": f"{BASE}/ar/legal/category/{cat}",
                    "kind": "legal",
                }
            continue
        m = LEAF_RE.search(href)
        if m:
            leaf_id = int(m.group(2))
            if leaf_id in HUBS or leaf_id in CHROME_IDS or leaf_id == LEGISLATIONS_ID:
                continue
            if leaf_id not in entries:
                entries[leaf_id] = {
                    "id": leaf_id, "title": title or f"page {leaf_id}",
                    "service_groups": ["Legal"],
                    "url_en": f"{BASE}/en/page/show/{leaf_id}",
                    "url_ar": f"{BASE}/ar/page/show/{leaf_id}",
                    "kind": "legal",
                }
            elif "Legal" not in entries[leaf_id]["service_groups"]:
                entries[leaf_id]["service_groups"].append("Legal")

    # Third level: each legal category lists resolutions/laws that link to
    # /en/legal/show/{id} — the FULL law text as clean HTML (no PDF needed).
    # This is the authoritative legal layer. Some laws are Arabic-only;
    # we fetch both language URLs regardless and let extraction flag thin
    # English shells.
    legal_show_re = re.compile(r"/en/legal/show/(\d+)")
    legal_cats = [e for e in entries.values() if e["kind"] == "legal" and "slug" in e]
    for cat_entry in legal_cats:
        print(f"discovering laws in: {cat_entry['title']}")
        cat_soup = content_region(get(cat_entry["url_en"], session).text)
        for a in cat_soup.find_all("a", href=True):
            m = legal_show_re.search(a["href"])
            if not m:
                continue
            law_id = int(m.group(1))
            key = -(1000 + law_id)  # own ID range, clear of FAQ/legal-cat
            if key in entries:
                continue
            title = (a.get("title") or a.get_text(" ", strip=True)).strip()
            entries[key] = {
                "id": key, "slug": f"law_{law_id}",
                "title": title,
                "service_groups": ["Legal"],
                "url_en": f"{BASE}/en/legal/show/{law_id}",
                "url_ar": f"{BASE}/ar/legal/show/{law_id}",
                "kind": "law",
            }

    manifest = sorted(entries.values(), key=lambda e: e["id"])
    print(f"discovered {len(manifest)} documents")
    return manifest


# --------------------------------------------------------------------------
# Stage 2 — fetch
# --------------------------------------------------------------------------

def fetch(manifest: list[dict], session: requests.Session) -> None:
    """Download every manifest entry in both languages; save raw HTML."""
    html_dir = RAW / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()

    for entry in manifest:
        name = entry.get("slug") or str(entry["id"])
        for lang in ("en", "ar"):
            out = html_dir / f"{name}_{lang}.html"
            if out.exists():
                continue  # idempotent: re-runs only fetch what's missing
            url = entry[f"url_{lang}"]
            print(f"fetching [{lang}] {entry['title']} — {url}")
            resp = get(url, session)
            resp.encoding = "utf-8"  # Arabic pages: trust the UTF-8 meta tag
            out.write_text(resp.text, encoding="utf-8")
            entry.setdefault("fetched", {})[lang] = {
                "file": str(out.relative_to(ROOT)),
                "fetch_date": today,
                "final_url": resp.url,  # records any redirect
            }
        save_manifest(manifest)  # checkpoint after each doc — crash-safe


def save_manifest(manifest: list[dict]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# --------------------------------------------------------------------------
# PDFs — deliberately narrow in v1
# --------------------------------------------------------------------------
# Policy: skip blank application-form PDFs (not prose; useless for RAG),
# but DO ingest informational guide PDFs and the legislation texts.
# Rather than guessing from URLs, Phase 1 will list PDF links found inside
# the fetched leaf/legislation pages and we whitelist the informational
# ones there, where we can see their context. Keeping Phase 0 HTML-only
# (plus that explicit whitelist later) avoids silently downloading dozens
# of scanned forms.


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    session = requests.Session()

    if cmd in ("discover", "all"):
        manifest = discover(session)
        save_manifest(manifest)
    if cmd in ("fetch", "all"):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        fetch(manifest, session)
        print("done.")


if __name__ == "__main__":
    main()

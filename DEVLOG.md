# Development log — bilingual LMRA RAG assistant

A complete record of how this project was built: the decisions, the
alternatives rejected, the bugs found, and what the numbers looked like at
each stage. The README is the outward-facing summary; this is the working
history.

## Project goal

A retrieval-augmented assistant over Bahrain's Labour Market Regulatory
Authority (lmra.gov.bh) answering work-permit and labour-service questions
in English or Modern Standard Arabic. Two non-negotiable design centers:
**cross-lingual retrieval** — a query in either language must retrieve from
a corpus containing both, because some authoritative content exists only in
Arabic — and **abstain-and-cite behaviour**, because a confidently wrong
answer could send someone to a ministry with the wrong documents.

---

## Phase 0 — Corpus crawl

**Design.** A manifest-driven, two-stage crawler (`crawler/crawl.py`)
instead of a recursive spider: stage one fetches the seven E-Services hub
pages (Establishment Employer /213, Domestic Employer /224, Expatriate
Employee /210, Registered Worker /424, Golden Residency /621, Licensed CRs
/592, General Services /593) and extracts only the "Details"/sub-page links
into `data/raw/manifest.json`; stage two fetches exactly those pages.
Recursive crawling on a government site is how you accidentally ingest
login portals and external sites; the hub→leaf structure made explicit
enumeration both safer and auditable. The manifest doubles as a provenance
record — re-run discovery and diff it to detect new or removed services.

A URL-substring blacklist (EMS_Web, ams.lmra.gov.bh,
services.bahrain.bh/wps/portal, wafid.com) filters links before they enter
the manifest. A chrome-page denylist (Terms, Careers, Sitemap, Contact…)
keeps header/footer navigation out. Language pairing is free: every page
exists at `/en/…` and `/ar/…` with the same numeric ID, giving perfectly
aligned document pairs. Politeness: realistic browser User-Agent, 1.5 s
delay before every request, sequential fetching, exponential-backoff
retries, idempotent re-runs (only missing files are fetched).

**Discoveries that reshaped the crawl.** The site turned out to have more
depth than the two-page-type model suggested. The FAQ page (`/en/faq`) is
itself a hub linking ~10 category pages that hold the actual Q&A inline —
with per-question "Last Update" dates and links to the exact service pages
they reference (which later became free evaluation ground truth). The
Legislations page (/221) is a hub over `legal/category/{1..5}` pages, which
are themselves listings linking `legal/show/{id}` pages containing **full
law texts as clean HTML** — meaning the "authoritative legal layer" needed
no PDF ingestion at all. Discovery was extended two levels accordingly, and
scoped to the `#page_content` region after finding that whole-page link
extraction let global-nav links (e.g. WPS /631, present on every page)
pollute every hub's `service_groups` metadata.

**Environment quirk.** The development sandbox's network allowlist blocked
lmra.gov.bh, so the crawler ran on the project owner's machine (a better
outcome anyway: the deliverable crawler got exercised end-to-end), with
verification done against the fetched files.

**Result.** 143 documents / 286 pages: 78 service leaves, 10 FAQ
categories, 47 full law texts, plus legal hub pages — every page in both
languages, tagged with URL, language, fetch date, and redirect record. All
Arabic pages passed a codepoint/mojibake integrity check.

## Phase 1 — Extraction & cleaning

**Design.** Deterministic scoping instead of heuristic content extraction
(trafilatura/readability): every LMRA page wraps real content in
`id="page_content"`, so extraction parses inside that anchor and removes
known chrome by name — and fails loudly if the anchor is missing, rather
than silently degrading. A custom ~150-line HTML→markdown walker (no
library) preserves the structure that carries meaning: h2/h3/h4 become
`#`/`##`/`###` headings (the site's section boundaries: Service Conditions,
Required Documents, Service Fees, Processing Time…), fee tables become
markdown tables with their captions kept as bold lines above, alert boxes
become blockquotes, links keep their URLs.

**Bugs found and fixed, in order.** (1) The junk-class remover initially
decomposed `grid-container-dependents`, which turned out to wrap both the
tab navigation *and* `main-content-dependents` — the entire page body.
Result: 134 near-empty documents on the first run. Fixed by removing only
the tab row itself. (2) `inline_text()` skipped text nodes whose parent was
a link, expecting an enclosing branch to emit them — which never fired when
the element passed in *was* the link, so the legal category listings
rendered as bare dates with no law titles. (3) The generic block handler
recursed into any element with children, so a `<a><span>badge</span>text</a>`
structure lost its anchor text; fixed by only recursing into elements with
real block-level children.

**Language reality check.** Content-language detection (Arabic codepoint
ratio) revealed that LMRA serves the **Arabic text on the /en/ URL** for
26 Arabic-only laws. Char-trigram Jaccard similarity against each doc's
pair identified 17 docs as near-duplicates (>0.9) of their Arabic twin —
flagged `redundant_with_pair` so the chunker indexes them once — while 20
genuinely mixed-language docs were kept.

**Result.** 286 clean markdown documents (~1M chars), each with
frontmatter: doc_id, title, URL language, detected content language,
source URL, service groups, last-update date, EN↔AR pair link, redundancy
flag. 269 indexable after redundancy exclusion. A PDF-link inventory
(`pdf_links.json`, 95 docs) was captured for a future informational-guide
pass; PDF ingestion was consciously deferred until after the eval harness
existed to measure its value.

## Phase 2 — Chunking & embedding (the technical heart)

**Chunking.** Section-aware, not fixed-window: split at `##` boundaries,
merge tiny neighbours (MIN 300 chars), split oversized sections at
paragraph boundaries (MAX 2000 chars ≈ 400–600 tokens, character-based so
the chunker stays embedding-model-agnostic). Tables are atomic — never
split mid-row; a table exceeding the cap splits *between* rows with the
caption and header row repeated in every fragment. Every chunk gets a
breadcrumb header (`<doc title> › <section path>`), which for Arabic docs
is naturally bilingual (English manifest title + Arabic section headings) —
an accidental but useful cross-lingual anchor. Language is detected
per-chunk, because mixed-language law docs make doc-level tags unreliable.

**Embedding model.** Three candidates compared: multilingual-e5-large
(battle-tested, but 512-token window truncates the 16 longest chunks and it
needs query/passage prefixes), **BGE-M3** (chosen: strongest Arabic
retrieval among open models, 8192-token window covers every chunk, dense
mode is drop-in), and hosted APIs (rejected: key/cost/external dependency
in a civic-info demo). Dense vectors only — M3's sparse/ColBERT modes were
deliberately skipped for transparency.

Why cross-lingual embedding works at all (the conceptual core): the XLM-R
backbone is pretrained on ~100 languages through a *shared subword
vocabulary and shared weights* — one network compressing all languages
through the same parameters reuses semantic features across them — and the
contrastive fine-tuning stage explicitly pulls translation pairs and
cross-lingual QA pairs toward the same points. What is said dominates the
geometry; which language said it becomes a weak residual direction.

**Vector store.** A deliberate anti-choice: no Chroma, no FAISS. At ~1,100
chunks, exact cosine search over L2-normalized vectors is one visible
matmul (`store.py`, ~60 lines). An ANN index would add opacity and nothing
else at this scale.

**Operational bumps.** The 2.27 GB model download stalled repeatedly from
the HF CDN (resolved by resume + patience; a phantom duplicate-format
`model.safetensors` download kept appearing and was neutralized with
`HF_HUB_OFFLINE=1` after caching). First encode run hit MPS out-of-memory
at batch 0 — sentence-transformers sorts by length, so the first batch was
16 of the longest chunks padded to equal length. Fixed with batch size 4
and a 2048-token sequence cap (truncates nothing; stops padding-buffer
allocation toward 8192).

**The make-or-break gate** (`test_crosslingual.py`). Forced-direction
batteries plus a language-residual measurement. Results: EN query → AR
corpus 5/5 (four at rank 1); AR query → EN corpus 4/5 (the miss was
arguably a labeling judgment); English → Arabic-only laws **1/3** — the
compound-hard case (cross-lingual + casual-vs-legal register + long
documents) failed. Geometry numbers: parallel EN/AR chunk pairs mean
cosine **0.781**, random cross-language **0.548**, random same-language
only ~0.03 above random cross-language — meaning dominates, language
barely registers. Verdict: core premise proven, legal layer needed help.

## Phase 3 — Retrieval + generation with guardrails

**The legal-layer fix: query-time bilingual expansion.** Chosen over
index-time English glosses on Arabic law titles specifically because
glosses would put LLM-generated text inside the index of a civic-info
system. Instead, a small LLM call (claude-haiku) translates each query into
the other language; both versions are embedded; every chunk is scored by
the max of the two cosines. After the fix, the two failed law cases
retrieved at ranks 1–2, correctly answering English questions from
Arabic-only resolutions.

**Abstention in three layers.** (1) A hard relevance gate: top merged score
below threshold (0.62, read off the battery score distributions) means the
LLM is never called — nothing to hallucinate from — and a fixed bilingual
template points the user to LMRA. (2) A grounding-only system prompt
(answer only from numbered sources, cite [n], answer in the query's
language, preserve exact figures, no procedural advice absent from
sources), with refusal via an explicit `[NOT_IN_SOURCES]` marker. (3) A
citation post-check mapping [n] markers back to retrieved chunks; an
answer citing nothing is not presented as fact.

**Findings during verification.** The model repeatedly named other
authorities in refusals ("contact the General Directorate of Traffic…")
even after an explicit rule against it — unverifiable outside knowledge.
Consequence: refusals display the **fixed template**, never model prose
(the model's refusal text goes to debug only). Instructions alone don't
beat the helpfulness reflex; the guarantee has to be structural. Also
found: the marker sometimes appears mid-answer (after a summary of what IS
covered), so detection was widened from the first 120 chars to the whole
text — a partial answer that self-declares the core question unanswered
must not pass as grounded.

Generation uses claude-sonnet; answers carry numbered citations resolved
to source URLs with language tags.

## Phase 4 — Evaluation harness (first-class deliverable)

**Design.** Dataset-agnostic by format: `run_eval.py` consumes any JSONL of
cases `{question, type: answerable|out_of_corpus, expected_urls,
cross_lingual}`. Deterministic metrics (retrieval hit@6, citation
correctness, abstention both ways — URL matching, no LLM) are separated
from the judged metric (faithfulness: an LLM judge sees *only* the
retrieved chunks and the answer, never the expected answer, and returns
SUPPORTED/PARTIAL/UNSUPPORTED with the unsupported claims listed).
Abstention is measured in both directions — out-of-corpus cases must
abstain AND answerable cases must not — because optimizing only the first
produces a system that refuses everything. Results append incrementally
(crash-safe resume); `--rescore` recomputes deterministic metrics against
edited labels with zero API calls; case IDs are content hashes so the
cache survives testset regeneration.

**Seeding.** 24 cases auto-derived from LMRA's own FAQ (questions whose
answers link the exact service page — ground-truth citations written by
the authority itself), 9 curated cross-lingual cases (including the
Arabic-only-law killers), 6 out-of-corpus probes. 39 total.

**Run 1: 29/33 retrieval, 23/29 citations, 4 false abstentions.**
Diagnosis separated testset artifacts from real defects. Artifacts: seeds
like "What are the fees?" that lost their FAQ-category context, a
non-question heading that slipped through, and labels that listed only one
language's URL when the system legitimately cited the same page's twin.
Real defects: two questions ("local transfer of a domestic employee",
EN+AR) where retrieval genuinely missed, and one faithfulness PARTIAL
where the answer attributed the human-trafficking hotline to the Expat
Protection Centre — a civic-info error no string metric would catch.

**The bug the eval caught.** Chasing the retrieval miss revealed that *no
chunk contained the question text at all*: the chunker's small-section
merge kept only the first piece's breadcrumb, so the headings of absorbed
sections — including entire FAQ questions — silently vanished from the
index. The same bug was deleting small service-page headings ("Processing
Time"). Fix: a merged-in piece from a different section re-embeds its own
heading inside the chunk content. Chunk count went 1105 → 1133; re-embed;
full re-run.

**Final numbers** (after fixes: chunker, marker detection, seed hygiene,
language-agnostic labels, an exact-attribution prompt rule):

| metric | overall | EN | AR | cross-lingual |
|---|---|---|---|---|
| retrieval hit@6 | 33/33 | 18/18 | 15/15 | 9/9 |
| citation correct | 33/33 | 18/18 | 15/15 | 9/9 |
| faithfulness SUPPORTED | 32/33 | 17/18 | 15/15 | 9/9 |
| abstained on out-of-corpus | 6/6 | 3/3 | 3/3 | — |
| false abstention (answerable) | 0/33 | 0/18 | 0/15 | 0/9 |

The residual PARTIAL (a call-centre number attributed to an adjacent
service's sources) is documented rather than tuned away. Bonus finding:
the judge surfaced that LMRA's own EN and AR pages *contradict each other*
in places (tripartite contract: "Original" in English, «نسخة»/copy in
Arabic) — a data-quality insight about bilingual government content.

**Held-out evaluation.** The final table is measured on cases the fixes
were iterated against — a legitimate development loop but a weak
generalization claim. So a held-out set of 15 fresh cases (services and
laws never touched in development: deportation deposit, authorized
persons, optional insurance, WPS, the Arabic-only inspection resolution…)
was written and run exactly once, no fixes permitted, results reported
as-is. Outcome: retrieval 12/12, citations 10/10, faithfulness 10/10,
out-of-corpus abstention 3/3, cross-lingual 3/3 — and **2/12 false
abstentions**. Diagnosis: both refusals had the correct page at rank 1;
they were two-part questions partially covered by thin service pages, and
the safety-first policy (any `[NOT_IN_SOURCES]` marker → full abstention,
adopted after case_8b7bd6d1) converts partial answers into refusals. The
system's residual bias is conservative: when it errs, it errs toward
"verify with LMRA," never toward invention. That trade-off is the honest
headline of the whole guardrail design.

## Phase 5 — Chat UI

A thin FastAPI server (`app/api.py`) wraps the exact `RagPipeline.answer()`
path the eval scored — the UI demos the evaluated system, no parallel
logic. The frontend is a deliberately single-file React app (CDN, no build
step; `app/index.html`): bilingual input with `dir="auto"` so Arabic
renders RTL natively, markdown answers (sanitized) so fee tables display
properly, sources as links with EN/AR language badges — making
cross-lingual retrieval *visible* in the demo — amber-styled abstention
bubbles presenting refusal as a feature, retrieval-score footnote per
answer, and an "unofficial demo — verify with LMRA" disclaimer.

## Phase 6 — Write-up & release

README in three buckets: well-justified choices (cross-lingual retrieval
with the measured geometry backing it, query-time expansion over index
glosses, source whitelisting, three-layer abstention, eval-first
development), defensible simplifications (one authority, MSA only,
leaf-page HTML only, exact search over ANN, single-turn), and genuine
limitations & future work (multi-hop FAQ pointers, Gulf-dialect queries →
a planned embedding fine-tune, NPRA/v2 "live and work in Bahrain"
expansion, content freshness via manifest-diff re-crawls, the known
residuals). Release hygiene: `.gitignore` excluding environments, secrets,
the regenerable index and raw eval results; MIT license; a secret scan;
pushed to GitHub (`Adel-10/lmra-bilingual-rag`) with the raw corpus
committed (~16 MB) for reproducibility.

---

## The numbers, end to end

| | |
|---|---|
| documents crawled | 143 (× 2 languages = 286 pages) |
| clean corpus | ~1.0 M chars, 269 indexable docs |
| chunks | 1,133 (606 EN / 527 AR), median 683 chars |
| embedding | BGE-M3 dense, 1024-d, L2-normalized |
| parallel-pair cosine | 0.781 vs 0.548 random cross-language |
| eval | 39 cases; retrieval 33/33, citations 33/33, faithfulness 32/33, abstention 6/6, false abstentions 0 |
| cross-lingual slice | perfect across all metrics (9/9) |

## What I'd tell someone reading this repo

The three most transferable lessons: **evaluation catches what inspection
can't** — the two most consequential bugs (vanishing headings, a
misattributed hotline) were found by metrics, not by reading output;
**safety guarantees must be structural, not instructional** — the model
ignored "don't name other authorities" until refusal text was replaced by
a deterministic template; and **cross-lingual retrieval genuinely works**,
but its hard edge (casual queries vs formal legal prose in another
language) needs an explicit mechanism — here, query-time translation —
not just faith in the embedding space.

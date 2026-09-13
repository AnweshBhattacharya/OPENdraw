# clippy

Milestone 3 deliverable: a **frozen, non-learned comparator baseline**.

It is not a step towards LineScout's retrieval models (that's `ml/`,
Milestone 4). It exists so the release gates in the Phase-I report have a
concrete floor to beat: the confirmatory evaluation (report §4.8) pools
OPENdraw's results against this comparator, and the Validated-v1 retrieval
gate (report Table 4.2) requires nDCG@8 to improve ≥20% relative to it.

"Clippy-style" (report §2.3) means: compare image and edge embeddings, no
training, no external service, no parity claim against any specific product.
Here that's a downsampled grayscale grid (`image_descriptor`) plus a small
unlearned HOG-style gradient-orientation histogram (`edge_descriptor`), each
L2-normalized, scored by a fixed-weight cosine blend. See
[`clippy/embeddings.py`](clippy/embeddings.py) for the exact method.

It is headless by design: no canvas, no FastAPI app, no warmup/readiness
state. It only touches a manifest and PNG files on disk.

## Quick start

```bash
cd eval
uv sync --extra dev --python 3.11

# Freeze an index over the committed synthetic fixture
.venv/bin/clippy build-index ../ml/fixtures/synthetic/manifest.json

# Rank the frozen index against a query image
.venv/bin/clippy query --index ../ml/fixtures/synthetic/clippy_index.json \
    ../ml/fixtures/synthetic/line_art/ls_synthetic_ac1f55b7390698a7.png

# Checks
.venv/bin/ruff check . && .venv/bin/mypy clippy && .venv/bin/pytest -q
```

## Freezing a real index

`build_index` only reads a manifest's `servable_records` — the same
eligibility rule the live API's gallery loader uses — so a Clippy index
never scores an asset OPENdraw itself wouldn't be allowed to show. Once a
manifest is locked for evaluation (report M3: "frozen splits/gallery/
protocol"), freeze its index and commit the `clippy_index.json` alongside
it; `ClippyIndex.is_current_for()` checks both the Clippy method version and
the manifest's content hash, so a later re-run can tell a stale index from a
current one instead of silently reusing outdated scores.

## What's deliberately out of scope here

- No stroke/vector query support yet — `embed_image` takes a flattened PNG
  (a canvas snapshot or a line-art file), matching what `/search` already
  rasterizes today. Wiring this into the confirmatory-evaluation harness
  (report §4.8: 250 queries at five completion percentages) is a Milestone 7
  concern, not this one.
- No `/health`-style readiness — Clippy isn't part of the running API and
  never blocks its startup.

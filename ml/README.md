# linescout-ml

Dataset manifest schema, taxonomy, and validation for LineScout, plus the
Milestone 2 ingestion pipeline that fills a gallery from raw source artwork.
Training, index construction, and evaluation arrive in later milestones.

Two halves live here:

* **`linescout_ml/`** — the manifest contract (`manifest.py`, `taxonomy.py`), the
  deterministic fixture generator, the `linescout-manifest` CLI, and
  **`linescout_ml/colab/`**: the ingestion pipeline (line-art extraction,
  measurement, pHash de-duplication, zero-shot labelling, feature embedding,
  manifest assembly, export).
* **`colab/`** — the Google Colab notebook that drives that pipeline on a free
  GPU runtime. See [`colab/README.md`](colab/README.md).

```bash
cd ml
uv sync --frozen --extra dev --python 3.11

# Validate a manifest (and check that enabled assets' files exist)
.venv/bin/linescout-manifest validate ../data/gallery/manifest.json --require-files

# Emit the JSON schema
.venv/bin/linescout-manifest schema --out ../packages/contracts/manifest.schema.json

# Regenerate the committed synthetic fixture (deterministic; safe to commit)
.venv/bin/linescout-manifest synth --out fixtures/synthetic --count 24 --seed 7

# Checks (what CI runs)
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy linescout_ml && .venv/bin/pytest -q

# Run the pipeline headless instead of in Colab: same lockfile, plus the GPU stack
uv sync --frozen --extra gpu --python 3.11

# Audit the Colab setup without a GPU: pins, lock coverage, the notebook's imports
.venv/bin/linescout-repro selfcheck
```

The `dev` extra is enough for everything CI does: the pipeline's CPU stages need
only `numpy` and `pillow`, and the GPU stages keep their imports lazy so a fresh
clone never has to install torch to validate a manifest. Nothing here installs a
GPU package implicitly: `linescout-repro selfcheck` asserts that, and it also checks
that `colab/requirements-colab.txt` agrees with `uv.lock`, that the checkpoint lock
covers every group the default pipeline wants, and that the notebook installs the plan
rather than a list of bare package names.

`fixtures/synthetic/` is the only dataset committed to Git. Real datasets live
under `data/` (ignored) and are never redistributed — the Colab notebook writes
its output to Drive, and section 13 of it explains how to land a gallery locally.

## Source datasets

Two HuggingFace datasets cover the two primary gallery styles. Both are
acquired locally (CPU-only) and then processed on a cloud GPU.

| Dataset | HF repo | Style coverage | Licence | Pilot budget |
|---|---|---|---|---|
| **PopManga** | `ragavsachdeva/popmanga_test` | `manga_anime` | Research-only — no redistribution | 300 pages (`seen` split) |
| **DCM-13k** | `emanuelevivoli/comix-v0_1-pages` | `western_ink` | US public domain pages; CoMix annotations research-only | 400 pages (streamed, capped) |

Neither dataset is committed to Git. Both live under `data/sources/` (gitignored)
and are never redistributed. The `source_registry.json` each script writes records
SHA-256 checksums, licence terms, and annotation counts — the manifest pipeline
reads these to populate provenance fields and `AllowedUses`.

### Install the download extra

```bash
# From the ml/ directory
cd ml
uv pip install -e ".[download]"
cd ..

# Activate the venv so 'python' uses the venv, not the system Python.
# Windows PowerShell:
ml\.venv\Scripts\Activate.ps1
# Linux/macOS:
source ml/.venv/bin/activate
# Prompt becomes:  (.venv) PS D:\CS\Personal Project\OPENdraw>
```

### PopManga (`manga_anime`)

```bash
# First: confirm your token works and inspect what chapters are available
# (venv must be active — see above)
python ml/scripts/download_popmanga.py --annotations-only

# Pilot run: ~50 pages across a few chapters
python scripts/download_popmanga.py --limit 50 --verbose

# Full seen split (all available chapters, no page limit)
python scripts/download_popmanga.py --split seen
```

Output lands in `data/sources/popmanga/`. Pages are saved per-chapter under
`pages/chapter_<id>/`. The script records chapter IDs and SHA-256 checksums
in `source_registry.json` for the manifest provenance pipeline.

> **How it works:** PopManga images are not on HuggingFace. The script downloads
> the 3.9 MB `annotations.zip` from the HF repo (needs your token), reads the
> chapter IDs from the annotation pickle, then fetches pages directly from
> MangaPlus via `mloader`.

### DCM-13k (`western_ink`)

DCM-13k has ~952 k pages across ~13 000 comic books. **Never download it
wholesale** — use the `--budget` cap to stay within the 20 % free-space
reserve the project requires (~80–160 MB for 400 pages).

```bash
# Pilot run: 200 pages spread across different books (fast, ~40–80 MB)
python scripts/download_dcm.py --budget 200 --verbose

# Standard acquisition: 400 pages, max 3 per book, skip covers (< 2 panels)
python scripts/download_dcm.py --budget 400 --max-per-book 3 --min-panels 2

# Resume a partial download (already-saved pages are skipped automatically)
python scripts/download_dcm.py --budget 400
```

Output lands in `data/sources/dcm/pages/`. Pages are saved as JPEG (native
format; ~100–400 KB each). The `--max-per-book` cap ensures variety across
comic titles rather than 400 pages from the same book.

### After downloading — curation and GPU pipeline

Both scripts print the same next-steps summary on exit:

1. Open `http://127.0.0.1:5173/curate` and review pages (assign scopes,
   SFW approval, quality 2–3, crop panel bounding boxes if needed).
2. Run the Colab GPU pipeline to extract line-art and embed:
   `ml/colab/linescout_gpu_pipeline.ipynb`
3. Validate the manifest: `linescout-manifest validate data/gallery/<v>/manifest.json --require-files`
4. Point the API at it: `LINESCOUT_GALLERY_MANIFEST=data/gallery/<v>/manifest.json`

---

## Cloud GPU training strategy

All model inference (line-art extraction, MobileCLIP2 + DINOv2 embedding,
scope-head training, ablations) runs on a cloud GPU. The ingestion pipeline
is already designed for this — it is resumable by source checksum and writes
its output to Drive before you bring it home.

### Platform comparison

| Platform | GPU | VRAM | Free quota | Best for |
|---|---|---|---|---|
| **Google Colab** (recommended) | T4 / A100 | 16 / 40 GB | ~12 h/session, shared pool | Full pipeline, notebook already wired |
| **Kaggle Notebooks** | T4 | 16 GB | 30 h/week, reliable | Ablation runs when Colab quota is low |
| **HuggingFace Spaces** | T4 | 16 GB | Limited free tier | Not suited — no custom ingestion |

Google Colab is the primary target because the notebook (`ml/colab/
linescout_gpu_pipeline.ipynb`) already handles pinned commits, SHA-256
checkpoint verification, and Drive export. Use Kaggle as a fallback when
Colab's free-tier GPU queue is long.

### What runs where

| Task | Where | Notes |
|---|---|---|
| `download_popmanga.py` | **Local (CPU)** | ~3 GB HF cache, no GPU needed |
| `download_dcm.py` | **Local (CPU)** | Streaming, ~80–160 MB output |
| Curation UI (`/curate`) | **Local (CPU)** | Runs against the API fixture gallery |
| Line-art extraction (Anime2Sketch) | **Colab / Kaggle** | 2–3 GB VRAM; batched |
| MobileCLIP2-S2 embedding | **Colab / Kaggle** | ~1.1 GB FP16; batch ≤ 32 on T4 |
| DINOv2-small embedding | **Colab / Kaggle** | ~0.9 GB FP16; batch ≤ 32 on T4 |
| Scope-head training (M4) | **Colab / Kaggle** | Fits easily on T4 |
| FAISS index build | **Local or Colab** | CPU-only; fast on 10 k vectors |
| API serving + search | **Local (GTX 1660 Ti)** | FP16, batch ≤ 8 per query |

### Colab workflow (quick reference)

```
1. Upload data/sources/ to Google Drive  (or mount Drive and run scripts there)
2. Open ml/colab/linescout_gpu_pipeline.ipynb in Colab
3. Set Runtime → Change runtime type → GPU (T4)
4. Run all cells — the notebook clones the repo at a pinned commit,
   verifies checkpoints, and writes gallery/ + indexes/ to Drive
5. Download the output zip and unpack into data/
6. Run:  linescout-manifest validate data/gallery/<version>/manifest.json --require-files
```

---

## GPU memory budget — GTX 1660 Ti (6 GB VRAM)

The 1660 Ti covers local API serving and quick single-batch checks.
Heavy extraction and training always run on Colab/Kaggle.

| Task | Memory (FP16) | Safe batch size |
|---|---|---|
| MobileCLIP2-S2 embedding | ~1.1 GB | ≤ 8 |
| DINOv2-small embedding | ~0.9 GB | ≤ 8 |
| Both encoders simultaneously | ~2.2 GB | ≤ 4 |
| Anime2Sketch line-art extraction | ~2–3 GB | Run separately |
| RTMPose (pose branch, optional) | ~1.5 GB | After encoders unloaded |

**Always use FP16 on the 1660 Ti.** BF16 is not supported on Turing-generation
GPUs — `model.half()` gives you FP16 instead. The batch sizes above keep peak
allocation under 4 GB, leaving 2 GB headroom for PyTorch's caching allocator.

If a local run hits OOM the API's recovery path (`linescout_api/state.py`)
retries once with pose disabled, then falls back to CPU. For offline index
building, halve the batch size and re-run — the pipeline is resumable.

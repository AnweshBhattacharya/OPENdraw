# OPENdraw · Milestone 4 Training Walkthrough (DCM)

Step-by-step guide for running `m4_training_pipeline.ipynb` on Google Colab.
This covers uploading your 200 DCM pages, running the full pipeline, downloading
the resulting gallery, and wiring it into the local API.

---

## Current state — what you already have

| Item | Status |
|---|---|
| DCM images downloaded | ✅ `data/sources/dcm/pages/` — 200 `.jpg` files, 69 MB |
| DCM `source_registry.json` | ✅ `data/sources/dcm/source_registry.json` |
| `m4_training_pipeline.ipynb` | ✅ `ml/colab/m4_training_pipeline.ipynb` |
| PopManga | ❌ All chapters locked on MangaPlus — skipped for now |

You do **not** need to run any download scripts. Everything is ready locally.

---

## Part 1 — Upload DCM images to Google Drive

### Step 1.1 — Open Google Drive

Go to [drive.google.com](https://drive.google.com) and sign in with your Google account.

### Step 1.2 — Create the folder structure

Click **New → Folder** and create this exact layout:

```
MyDrive/
  LineScout/
    sources/
      dcm/
        pages/        ← you will upload the .jpg files here
```

You can create nested folders by opening each one and clicking **New → Folder** inside it.

### Step 1.3 — Upload the DCM pages

1. Open `MyDrive/LineScout/sources/dcm/pages/` in Drive
2. Open a second window showing `d:\CS\Personal Project\OPENdraw\data\sources\dcm\pages\` in Explorer
3. Select all 200 `.jpg` files and drag them into the Drive browser tab
4. Wait for the upload progress bar to finish — 200 files at ~350 KB each takes ~2–5 min

**Do not upload `source_registry.json` or anything else — only the `.jpg` files.**

### Step 1.4 — Verify the upload

After uploading, Drive should show **200 items** inside the `pages/` folder.
If it shows fewer, some files may have failed — drag and drop again; Drive skips duplicates.

---

## Part 2 — Open the notebook in Colab

### Option A — Upload from your machine (easiest)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Navigate to `d:\CS\Personal Project\OPENdraw\ml\colab\m4_training_pipeline.ipynb`
4. The notebook opens in your browser

### Option B — Open from Kiro (Colab extension)

1. In Kiro's Explorer, open `ml/colab/m4_training_pipeline.ipynb`
2. In the top-right kernel picker, click **Select Kernel → Colab Kernels**
3. Sign in to Google when prompted
4. The notebook connects to a Colab T4 runtime

---

## Part 3 — Set the runtime to T4 GPU

**This must be done before running any cell.**

1. In the Colab menu bar: **Runtime → Change runtime type**
2. Under **Hardware accelerator**, select **T4 GPU**
3. Click **Save**

The RAM/disk meter in the top-right should now show values like `12.7 GB / 78.2 GB`.
If it shows nothing, the runtime hasn't started yet — run Cell 1 to trigger it.

---

## Part 4 — Run the cells

Run **one cell at a time** on your first pass. Click the ▶ button on the left of each cell,
or press `Shift+Enter`.

**Total estimated time: ~35–55 min** for 200 pages on a free T4.

---

### Cell 1 · Mount Drive + configure paths

**What happens:** Mounts your Google Drive at `/content/drive/` and verifies the DCM images
are visible. Prompts you to allow Drive access in a browser pop-up — approve it.

**Expected output:**
```
Mounted at /content/drive
DCM pages found: 200
Gallery output  : /content/drive/MyDrive/LineScout/gallery/2026.10.03-dcmpilot
Sample filenames: ['c00004_p000.jpg', 'c00004_p001.jpg', …]
```

**If you see `DCM pages found: 0`:** The upload path doesn't match. Check that
the files are at `MyDrive/LineScout/sources/dcm/pages/` (not a subfolder inside it).

> **Optional:** Change `DATASET_VERSION` to today's date before running if you want
> a fresh output directory: `DATASET_VERSION = '2026.10.03-dcmpilot'`

---

### Cell 2 · GPU check

**What happens:** Confirms a real CUDA T4 is connected. Raises an error if it isn't.

**Expected output:**
```
GPU : Tesla T4  |  15.8 GB VRAM
CUDA: 12.4   PyTorch: 2.14.0+cu124
```

**If you see `No GPU detected`:** Go to `Runtime → Change runtime type → T4 GPU`,
reconnect, and start from Cell 1.

---

### Cell 3 · Clone repo + install dependencies

**What happens:**
- Clones `junosapollo/drawable` at the pinned commit (the code that matches the manifest schema)
- Installs `open-clip-torch`, `timm`, `controlnet-aux` at pinned versions
- Freezes Colab's CUDA torch so pip can't replace it with a CPU build

**Takes:** 3–5 min on first run. Subsequent runs reuse the clone.

**Expected output ends with:**
```
HEAD : 81a8683a53ea  (matches pin)
  open_clip ✓
  timm ✓
  controlnet_aux ✓
Ready.
```

**If you see `open_clip ✗`:** The pip install hit a version conflict. Look at the
output above for the specific error. Usually safe to ignore if the other two are ✓ —
re-run the cell once before escalating.

**If you see `torch lost its CUDA build`:** Colab's image updated. Go to
`Runtime → Disconnect and delete runtime`, reconnect, and re-run from Cell 2.

---

### Cell 4 · Build pipeline config

**What happens:** Creates the `PipelineConfig` and `SourceSpec` objects. No GPU work.
Uses `informative_drawings` extractor (better than `anime2sketch` for Western comic
cross-hatching and brushwork).

**Expected output:**
```
PipelineConfig OK
  dataset_version : 2026.10.03-dcmpilot
  source          : dcm  (200 images)
  extractor       : informative_drawings
  embedders       : ['mobileclip2_s2', 'dinov2_vits14']
  batch_size      : 8
  output          : /content/drive/MyDrive/LineScout/gallery/2026.10.03-dcmpilot
```

**If you get a `ValidationError`:** One of the `SourceSpec` fields is wrong.
The error message will name the field. Most common cause: `permission_basis` doesn't
match `allowed_display/training`. The DCM config should always pass as-is.

---

### Cell 5 · Stage 1 — Discover images

**What happens:** Scans the DCM pages folder, assigns a deterministic `asset_id` and
learning split (70/15/15 train/val/test) to every image. CPU-only, ~10 seconds.

**Expected output:**
```
Stage 1/6: Discovering …
Discovered 200 candidates
Splits  →  none: 0  test: 30  train: 140  validation: 30
```

---

### Cell 6 · Stage 2 — Extract line art  ⚡ GPU

**What happens:** Runs `informative_drawings` on every DCM page. Downloads model
weights (~60 MB) on first run. Saves three files per page to Drive:
- `originals/<id>.jpg` — original JPEG
- `line_art/<id>.png` — extracted line art
- `thumbnails/<id>.png` — 256×256 thumbnail

**Resumable:** If the session disconnects, re-run Cells 1–4 then jump straight back
to this cell. Already-extracted images are skipped.

**Estimated time:** 10–20 min for 200 pages.

**Expected output ends with:**
```
  extract: 200/200
Extracted  done=200  skipped=0  failed=0
```

**If you get OOM:** Set `BATCH_SIZE = 4` in Cell 4, re-run Cell 4, then re-run this cell.

**If you see `failed=N`:** Some images were unreadable or too small. A few failures
are normal (corrupted DCM scans). If more than 10 fail, check the notes printed below.

---

### Cell 7 · Stage 3 — Measure + deduplicate

**What happens:** CPU-only. Measures ink coverage, text coverage, quality score, and
a perceptual hash (pHash) for each line-art file. Then groups near-duplicates
(pages with pHash Hamming distance ≤ 6) and marks all but one as duplicates.

**Expected output:**
```
Measured 200/200
Deduplication: removed=0  unique=200
```

A few duplicates are normal if your DCM sample happened to include the same comic page
from two different scans. Duplicates are kept in the manifest but never served.

---

### Cell 8 · Stage 4 — Zero-shot labelling  ⚡ GPU

**What happens:** Uses MobileCLIP2-S2 to assign provisional `primary_scope` and
`primary_style` labels via text-image matching. Also runs the SFW classifier.

**These labels are provisional** — they are a starting point, not ground truth.
The `/curate` UI lets you correct any wrong label before the asset is served.

**Estimated time:** 4–8 min for 200 images.

**Expected output:**
```
Labelled 200/200
Top scopes: full_body=72  upper_body_clothing=48  face_head=36  hand=20 …
SFW: safe=200  quarantined=0
```

DCM is a public-domain archive; `quarantined=0` is expected. If any assets are
quarantined they need SFW adjudication in the `/curate` UI before they can be served.

---

### Cell 9 · Stage 5 — Embed  ⚡ GPU

**What happens:** Runs both encoders on every line-art image:

| Encoder | Dim | Purpose |
|---|---|---|
| MobileCLIP2-S2 | 512 | Semantic (text-image aligned, for scope matching) |
| DINOv2-small | 384 | Structural (self-supervised, for contour/line matching) |

Writes resumable `.npz` shard files to `indexes/<version>/` alongside the gallery.

**Estimated time:** 8–15 min for 200 images.

**Expected output:**
```
  mobileclip2_s2: 200/200  (512d)
  dinov2_vits14 : 200/200  (384d)
Embeddings written.
```

---

### Cell 10 · Build manifest + validate

**What happens:** Assembles the schema v3 `manifest.json` from all stage outputs,
then validates it with the same validator the API uses.

`0 servable` is **correct and expected** — all assets start as `unreviewed`.
They become servable only after human curation (Part 5).

**Expected output:**
```
Manifest written: 200 records  0 servable  (unreviewed — needs curation)
Validation: OK  content_hash=3a7f9c2b1d4e
```

**If validation fails:** The error message names the specific field/record.
This should not happen with the pre-built config; if it does, paste the error here.

---

### Cell 11 · Build FAISS index

**What happens:** Loads all embedding shards and builds two `IndexFlatIP` indexes
(exact cosine search — at 200 vectors, approximation would only add noise).

Writes `faiss.index` and `id_map.json` into each encoder's shard directory.

**Expected output:**
```
Building FAISS indexes …
  mobileclip2_s2 : 200 vectors  dim=512  → …/mobileclip2_s2/faiss.index
  dinov2_vits14  : 200 vectors  dim=384  → …/dinov2_vits14/faiss.index
```

---

### Cell 12 · Export + download

**What happens:**
1. Creates `gallery.zip` — manifest + line_art + thumbnails (~30–50 MB, no originals)
2. Creates `indexes.zip` — FAISS indexes + id maps (~5–10 MB)
3. Copies both ZIPs to `MyDrive/LineScout/exports/<version>/` for safe-keeping
4. Triggers browser downloads for both files

**Do not close the Colab tab until both downloads complete.**
If a download dialog doesn't appear, check your browser's pop-up blocker.

**Expected output:**
```
gallery.zip : 602 files  (38.4 MB)
indexes.zip : 8 files    (6.1 MB)
Copied to Drive: /content/drive/MyDrive/LineScout/exports/2026.10.03-dcmpilot
Starting downloads to your machine …
```

---

### Cell 13 · Summary

Shows record counts and prints the exact commands to unpack and wire up the gallery
locally. Read this before closing the tab.

---

## Part 5 — Bring the gallery home (run locally after downloading)

Open PowerShell in the repo root and run these commands.
Replace `2026.10.03-dcmpilot` with your actual `DATASET_VERSION` if you changed it.

### Step 5.1 — Unpack the ZIPs

```powershell
# Unpack gallery
New-Item -ItemType Directory -Force "data\gallery\2026.10.03-dcmpilot"
Expand-Archive "$env:USERPROFILE\Downloads\gallery.zip" -DestinationPath "data\gallery\2026.10.03-dcmpilot\"

# Unpack indexes
New-Item -ItemType Directory -Force "data\indexes"
Expand-Archive "$env:USERPROFILE\Downloads\indexes.zip" -DestinationPath "data\indexes\"
```

### Step 5.2 — Validate the manifest

```powershell
ml\.venv\Scripts\python -m linescout_ml.cli validate `
  data\gallery\2026.10.03-dcmpilot\manifest.json --require-files
```

**Expected output:**
```
OK data\gallery\2026.10.03-dcmpilot\manifest.json: 200 records, 0 servable
artifact_contract=…
```

### Step 5.3 — Configure the API

Edit (or create) `services/api/.env`:

```ini
LINESCOUT_GALLERY_MANIFEST=data/gallery/2026.10.03-dcmpilot/manifest.json
LINESCOUT_CURATION_MODE=1
LINESCOUT_FIXTURE_MODE=false
LINESCOUT_DB_PATH=data/linescout.sqlite3
```

### Step 5.4 — Start the app

```powershell
npm run dev:all
```

This starts the API on port 8000 and the web app on port 5173.

---

## Part 6 — Curate the gallery

Go to **http://127.0.0.1:5173/curate**

The curation queue shows all 200 DCM pages as `unreviewed`. For each page:

1. **Check the scope** — the zero-shot label is a starting point, not ground truth.
   Common corrections for DCM comic pages:
   - Full-page splash panel → `full_body`
   - Head/face close-up → `face_head`
   - Fight scene with characters → `full_body` or `multi_character`
   - Clothing/costume detail → `upper_body_clothing`

2. **Approve SFW** — DCM is public domain; all pages should pass

3. **Set quality** — `2` for a usable reference, `3` for an excellent one

4. **Click Accept**

Once an asset is accepted + SFW-approved + quality≥2, it becomes **servable** and
will appear in search results on the Draw page.

> **Tip:** After curating ~20 pages, go to http://127.0.0.1:5173/draw, draw a
> few strokes, and check that references appear in the right panel. The badge
> should change from **Fixture** to **API fixture** or **CPU fallback**.

---

## Troubleshooting quick reference

| Symptom | Fix |
|---|---|
| `DCM pages found: 0` in Cell 1 | Upload `.jpg` files to `MyDrive/LineScout/sources/dcm/pages/` |
| `No GPU detected` in Cell 2 | `Runtime → Change runtime type → T4 GPU` |
| `torch lost its CUDA build` | `Runtime → Disconnect and delete runtime`, reconnect, re-run Cell 2 |
| OOM in Cell 6 or 9 | Set `BATCH_SIZE = 4` in Cell 4, re-run Cell 4 then the failing cell |
| Session disconnected mid-run | Re-run Cells 1–4, then jump back to the interrupted cell |
| Download dialog doesn't appear | Check browser pop-up blocker — allow colab.research.google.com |
| `0 servable` after unpack | Expected — run `/curate` to accept assets (Part 6) |
| Badge shows **Fixture** on Draw page | `LINESCOUT_FIXTURE_MODE=false` not set in `services/api/.env` |
| Curation queue is empty | Manifest path in `.env` is wrong or `LINESCOUT_CURATION_MODE=1` is missing |

---

## Kaggle fallback (if Colab GPU queue is full)

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. **Settings** (right sidebar) → **Accelerator → GPU T4 x1**
3. **File → Import Notebook** → upload `m4_training_pipeline.ipynb`
4. For Cell 1, replace the Drive mount with a Kaggle dataset:
   - Upload your `dcm/pages/` as a Kaggle dataset first:
     `kaggle datasets create -p data/sources/dcm/pages/`
   - Then add it to the notebook via **Add Data → Your Datasets**
   - Change `DCM_ROOT = Path('/kaggle/input/<your-dataset-slug>')`
5. After Cell 12, download outputs from the **Output** tab (right sidebar)

Kaggle gives 30 h/week of free T4.

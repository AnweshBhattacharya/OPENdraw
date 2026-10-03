"""Download PopManga pages for the LineScout gallery pipeline.

How it works
------------
The HuggingFace repo (ragavsachdeva/popmanga_test) contains:
  - annotations.zip  — per-page JSON annotations (character/panel bboxes)
  - mangaplus_links.txt inside the ZIP — 50 MangaPlus chapter viewer URLs
  - seen.txt / unseen.txt — which pages belong to each evaluation split

Images live on MangaPlus (mangaplus.shueisha.co.jp), NOT on HuggingFace.
This script:
  1. Downloads annotations.zip from HF (requires token, repo is gated)
  2. Reads mangaplus_links.txt → extracts chapter IDs
  3. Uses mloader to download pages chapter-by-chapter from MangaPlus
  4. Matches downloaded pages to seen.txt / unseen.txt annotation entries
  5. Copies per-page annotation JSON alongside each image
  6. Writes source_registry.json for the manifest pipeline

Usage::

    # Activate the venv first (from repo root):
    #   ml\\.venv\\Scripts\\Activate.ps1

    # Step 1 — download annotations only (fast, confirms token works):
    python ml/scripts/download_popmanga.py --annotations-only

    # Step 2 — download first 2 chapters (smoke test, ~100 pages):
    python ml/scripts/download_popmanga.py --chapters 2

    # Step 3 — download all 50 chapters (full seen+unseen, ~1900 pages):
    python ml/scripts/download_popmanga.py

    # Resumable: re-run the same command to skip already-downloaded chapters.

Requires the download extra (from ml/ directory):
    uv pip install -e ".[download]"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def _check_deps() -> None:
    missing = []
    for pkg, label in [("huggingface_hub", "huggingface-hub"), ("mloader", "mloader")]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(label)
    if missing:
        print(
            f"ERROR: Missing packages: {', '.join(missing)}\n"
            "Fix: cd ml && uv pip install -e \".[download]\"",
            file=sys.stderr,
        )
        sys.exit(1)


def _ensure_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        token_file = Path.home() / ".cache" / "huggingface" / "token"
        if token_file.exists():
            token = token_file.read_text(encoding="utf-8").strip() or None
    if not token:
        print(
            "\nERROR: HuggingFace token required (repo is gated).\n"
            "  1. Accept terms: https://huggingface.co/datasets/ragavsachdeva/popmanga_test\n"
            "  2. Get a READ token: https://huggingface.co/settings/tokens\n"
            "  3. Set it:  $env:HF_TOKEN = 'hf_...'   (PowerShell)\n"
            "             huggingface-cli login         (saves permanently)\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return token  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Step 1 — download annotations.zip
# ---------------------------------------------------------------------------

DATASET_REPO = "ragavsachdeva/popmanga_test"
ZIP_NAME = "annotations.zip"

SOURCE_TERMS = {
    "license_id": "popmanga-research",
    "basis": "license_terms",
    "allowed_display": True,
    "allowed_training": True,
    "allowed_trace": False,
    "attribution_required": False,
    "notes": (
        "Research dataset. Images from MangaPlus (Shueisha). "
        "Do not redistribute images or use commercially."
    ),
}


def _download_annotations_zip(out_dir: Path, token: str) -> Path:
    from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]
    zip_path = out_dir / ZIP_NAME
    if zip_path.exists():
        print(f"  annotations.zip already at {zip_path} — skipping download")
        return zip_path
    print(f"Downloading {ZIP_NAME} from {DATASET_REPO} …")
    path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=ZIP_NAME,
        repo_type="dataset",
        token=token,
        local_dir=str(out_dir),
    )
    print(f"  → {path}  ({Path(path).stat().st_size / 1024:.0f} KB)")
    return Path(path)


# ---------------------------------------------------------------------------
# Step 2 — parse annotations.zip
# ---------------------------------------------------------------------------

def _parse_annotations_zip(zip_path: Path) -> tuple[list[int], dict[str, dict]]:
    """Return (chapter_ids, page_annotations).

    chapter_ids  — list of ints from mangaplus_links.txt, e.g. [1000001, ...]
    page_annotations — dict mapping bare page filename (no path prefix) to the
                       annotation dict parsed from the per-page JSON, e.g.
                       {'Naruto - c001 - p000.jpg': {...bboxes...}}
    """
    chapter_ids: list[int] = []
    page_annotations: dict[str, dict] = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

        # --- chapter IDs from mangaplus_links.txt ---
        link_file = next((n for n in names if n.endswith("mangaplus_links.txt")), None)
        if link_file:
            text = zf.read(link_file).decode("utf-8")
            for line in text.splitlines():
                m = re.search(r"/viewer/(\d+)", line.strip())
                if m:
                    chapter_ids.append(int(m.group(1)))
        else:
            print("WARNING: mangaplus_links.txt not found in annotations.zip", file=sys.stderr)

        # --- per-page annotation JSONs ---
        json_files = [n for n in names if n.endswith(".jpg.json")]
        for json_path in json_files:
            # Key: the bare filename without directory, without .json suffix
            # e.g.  "annotations/One Piece/One Piece - c001 .../page.jpg.json"
            #   →   "page.jpg"
            bare = Path(json_path).name.removesuffix(".json")
            try:
                data = json.loads(zf.read(json_path).decode("utf-8"))
                page_annotations[bare] = data
            except Exception:  # noqa: BLE001
                pass

    print(f"  Parsed {len(chapter_ids)} chapter URLs, {len(page_annotations)} page annotations")
    return chapter_ids, page_annotations


# ---------------------------------------------------------------------------
# Step 3 — download pages via mloader
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_chapter(chapter_id: int, chapters_dir: Path) -> list[Path]:
    """Run mloader for one chapter. Returns list of downloaded image paths."""
    chapter_dir = chapters_dir / f"{chapter_id}"
    chapter_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already downloaded
    existing = sorted(chapter_dir.glob("*.jpg")) + sorted(chapter_dir.glob("*.png"))
    if existing:
        return existing

    cmd = [
        sys.executable, "-m", "mloader",
        "-c", str(chapter_id),
        "-o", str(chapter_dir),
        "--raw",          # save individual images, not CBZ
        "-q", "super_high",
        "--chapter-subdir",  # puts pages in a sub-folder per chapter
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # mloader sometimes exits non-zero but still saves files
        pass

    # Collect everything mloader saved (may be in a sub-directory)
    found = sorted(chapter_dir.rglob("*.jpg")) + sorted(chapter_dir.rglob("*.png"))
    return found


# ---------------------------------------------------------------------------
# Step 4 — match downloaded pages to annotation entries
# ---------------------------------------------------------------------------

def _best_match(img_path: Path, page_annotations: dict[str, dict]) -> dict | None:
    """Try to find annotation data for an image file by fuzzy-matching its name."""
    name = img_path.name  # e.g. "p000.jpg"
    # Direct hit (unlikely given path differences, but try)
    if name in page_annotations:
        return page_annotations[name]
    # Try stem match against any annotation key
    stem = img_path.stem  # e.g. "p000"
    for key, data in page_annotations.items():
        if stem in key:
            return data
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download PopManga pages from MangaPlus into the LineScout data tree.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--annotations-only", action="store_true",
        help="Only download annotations.zip — skip image download.",
    )
    parser.add_argument(
        "--chapters", type=int, default=None, metavar="N",
        help="Download only the first N chapters (default: all 50).",
    )
    parser.add_argument(
        "--out", default="data/sources/popmanga", metavar="DIR",
        help="Output directory (default: data/sources/popmanga).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print every chapter and page.",
    )
    args = parser.parse_args(argv)

    _check_deps()
    token = _ensure_hf_token()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: annotations ZIP ──────────────────────────────────────────
    zip_path = _download_annotations_zip(out_dir, token)

    if args.annotations_only:
        print(f"\n--annotations-only: done. ZIP at {zip_path}")
        return 0

    # ── Step 2: parse chapter IDs + page annotations ─────────────────────
    print("\nParsing annotations.zip …")
    chapter_ids, page_annotations = _parse_annotations_zip(zip_path)

    if not chapter_ids:
        print("ERROR: No chapter IDs found in annotations.zip.", file=sys.stderr)
        return 1

    cap = args.chapters if args.chapters is not None else len(chapter_ids)
    chapters_to_fetch = chapter_ids[:cap]
    print(f"\nWill download {len(chapters_to_fetch)} of {len(chapter_ids)} chapters …")

    # ── Step 3 + 4: download + record ────────────────────────────────────
    chapters_dir = out_dir / "pages"
    chapters_dir.mkdir(exist_ok=True)

    registry_entries: list[dict] = []
    total_pages = 0
    failed_chapters: list[int] = []

    for idx, chapter_id in enumerate(chapters_to_fetch):
        print(f"  [{idx+1}/{len(chapters_to_fetch)}] chapter {chapter_id}", end=" … ", flush=True)
        pages = _download_chapter(chapter_id, chapters_dir)

        if not pages:
            print("FAILED (no images saved)")
            failed_chapters.append(chapter_id)
            continue

        print(f"{len(pages)} pages")
        for img in pages:
            annotation = _best_match(img, page_annotations)
            char_bboxes = []
            panel_bboxes = []
            if annotation:
                bboxes = annotation.get("bboxes_as_x1y1x2y2", [])
                labels = annotation.get("labels", [])
                char_bboxes  = [b for b, l in zip(bboxes, labels) if l == 0]
                panel_bboxes = [b for b, l in zip(bboxes, labels) if l == 2]

            registry_entries.append({
                "chapter_id": chapter_id,
                "filename": img.name,
                "relative_path": str(img.relative_to(out_dir)).replace("\\", "/"),
                "sha256": _sha256(img),
                "character_bbox_count": len(char_bboxes),
                "panel_bbox_count": len(panel_bboxes),
                "has_annotations": annotation is not None,
            })
            if verbose := args.verbose:
                print(f"      {img.name}  chars={len(char_bboxes)}  panels={len(panel_bboxes)}")
        total_pages += len(pages)

    # ── Write registry ────────────────────────────────────────────────────
    registry = {
        "source_dataset": "popmanga",
        "hf_repo": DATASET_REPO,
        "downloaded_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_chapters": len(chapters_to_fetch) - len(failed_chapters),
        "total_pages": total_pages,
        "failed_chapters": failed_chapters,
        "terms": SOURCE_TERMS,
        "assets": registry_entries,
    }
    reg_path = out_dir / "source_registry.json"
    reg_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'─'*60}")
    print(f"Downloaded : {total_pages} pages across "
          f"{len(chapters_to_fetch) - len(failed_chapters)} chapters")
    if failed_chapters:
        print(f"Failed     : {len(failed_chapters)} chapters — {failed_chapters}")
        print("  Re-run the script to retry failed chapters (resumable).")
    print(f"Registry   : {reg_path}")
    print(f"{'─'*60}")
    print("\nNext step: upload data/sources/ to Google Drive, then open")
    print("ml/colab/m4_training_pipeline.ipynb in Colab (T4 GPU).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

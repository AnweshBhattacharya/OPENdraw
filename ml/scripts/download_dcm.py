"""Download a budget-capped sample from DCM-13k (comix-v0_1-pages) for the LineScout gallery.

DCM-13k contains ~952 k pages across ~13,000 Western comic books sourced from the
Digital Comic Museum (digitalcomicmuseum.com). The full dataset is hundreds of GB;
this script streams it and stops at a configurable page budget so local storage is
never exhausted.

What this script does
---------------------
1. Streams ``emanuelevivoli/comix-v0_1-pages`` from HuggingFace without downloading
   the full dataset to disk first.
2. Samples pages spread across distinct books (``book_id`` deduplicated) so the pilot
   gallery gets style and composition variety, not 500 pages from the same title.
3. Saves each page as a JPEG (the dataset's native format) into
   ``data/sources/dcm/pages/``.
4. Writes ``data/sources/dcm/source_registry.json`` with per-page SHA-256 checksums,
   panel/character detection counts from the MAGI annotations, and the licence terms
   that flow into the manifest pipeline's ``AllowedUses`` fields.
5. Prints a next-steps summary matching the PopManga workflow.

Usage::

    # From the repo root, with the ml venv active:
    python ml/scripts/download_dcm.py                        # default: 400 pages
    python ml/scripts/download_dcm.py --budget 200           # pilot subset
    python ml/scripts/download_dcm.py --budget 400 --max-per-book 2 --verbose

    # Resume a partial download (already-saved files are skipped):
    python ml/scripts/download_dcm.py --budget 400

Requires the ``download`` extra:
    uv pip install -e "ml[download]"

DCM / CoMix licence
-------------------
The underlying comic pages are from the Digital Comic Museum and are in the
**US public domain** (pre-1923 or explicitly released). The CoMix annotation
layer (panel/character bounding boxes) is released for **research use**.

Allowed uses recorded in source_registry.json:
    display=True, training=True, trace=False

``trace=False`` is conservative: individual publisher/artist rights within the
DCM collection are heterogeneous. A curation pass can upgrade specific assets to
``trace=True`` if the work is verifiably public domain and the line-art is native.
The manifest pipeline will not serve any asset as traceable until a human curation
decision explicitly sets it.

Storage budget
--------------
Each saved JPEG is ~100–400 KB. A 400-page pilot sample uses roughly 80–160 MB.
The full ~952 k-page dataset would need 100–400 GB — never download it wholesale.
Use ``--budget`` to stay within the 20 % free-space reserve the project requires.

Why streaming works here
------------------------
HuggingFace Datasets streaming mode fetches shards on demand and never writes the
full dataset to the HF cache. Each page is decoded, optionally saved, and then
released. Peak memory stays under 500 MB regardless of the total dataset size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Lazy imports (only in the 'download' extra)
# ---------------------------------------------------------------------------

def _require_imports() -> None:
    try:
        import datasets  # noqa: F401  # type: ignore[import-untyped]
    except ImportError:
        print(
            "ERROR: 'datasets' is not installed.\n"
            "Install the download extra:  uv pip install -e \"ml[download]\"\n"
            "or:                          pip install datasets pillow",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        from PIL import Image  # noqa: F401  # type: ignore[import-untyped]
    except ImportError:
        print(
            "ERROR: 'pillow' is not installed.\n"
            "Install the download extra:  uv pip install -e \"ml[download]\"",
            file=sys.stderr,
        )
        sys.exit(1)


def _ensure_hf_token() -> None:
    """Authenticate with HuggingFace Hub, required for gated datasets.

    Resolution order:
      1. HF_TOKEN environment variable
      2. ~/.cache/huggingface/token  (written by ``huggingface-cli login``)

    If neither is present, prints a clear error with the exact steps and exits.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if not token:
        token_file = Path.home() / ".cache" / "huggingface" / "token"
        if token_file.exists():
            token = token_file.read_text(encoding="utf-8").strip() or None

    if not token:
        print(
            "\nERROR: comix-v0_1-pages is a gated dataset — authentication is required.\n"
            "\nSteps to fix:\n"
            "  1. Create a free HuggingFace account at https://huggingface.co/join\n"
            "  2. Accept the dataset terms at:\n"
            "     https://huggingface.co/datasets/emanuelevivoli/comix-v0_1-pages\n"
            "     (click 'Access repository' and agree to the terms)\n"
            "  3. Create a read token at https://huggingface.co/settings/tokens\n"
            "  4. Either:\n"
            "     a) Set the environment variable:\n"
            "        $env:HF_TOKEN = 'hf_your_token_here'   # PowerShell\n"
            "        export HF_TOKEN=hf_your_token_here       # bash/zsh\n"
            "     b) Or log in via the CLI (saves token to ~/.cache/huggingface/token):\n"
            "        huggingface-cli login\n"
            "  5. Re-run this script.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from huggingface_hub import login  # type: ignore[import-untyped]
        login(token=token, add_to_git_credential=False)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: HuggingFace login failed: {exc}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_HF_REPO = "emanuelevivoli/comix-v0_1-pages"
SOURCE_DATASET_NAME = "dcm"

# Per-asset licence terms recorded in the source registry.
# Public-domain status applies to the *pages*; the CoMix annotation layer is
# research-only, but annotations are never redistributed by this project.
SOURCE_TERMS = {
    "license_id": "dcm-public-domain",
    "basis": "public_domain",
    "allowed_display": True,
    "allowed_training": True,
    # Conservative default: individual works need a curation decision before
    # tracing is enabled. A human reviewer can upgrade specific assets.
    "allowed_trace": False,
    "attribution_required": False,
    "notes": (
        "Pages sourced from the Digital Comic Museum (US public domain). "
        "CoMix annotations are research-only and are NOT redistributed. "
        "Trace use disabled by default: upgrade per-asset via curation if "
        "the work is verifiably public domain and origin is native line-art."
    ),
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_metadata(example: dict) -> tuple[str, str, int, int, int, int]:
    """Parse page_id, book_id, and annotation counts from one dataset example.

    Returns (page_id, book_id, panel_count, character_count, width, height).
    Falls back to safe defaults for any missing field so a malformed example
    never aborts the whole download run.
    """
    meta: dict = example.get("json", {}) or {}
    page_id: str = str(meta.get("page_id", f"unknown_{id(example)}"))

    # book_id is not always a top-level key; derive it from page_id when absent.
    # PopManga uses a similar composite-id pattern.
    book_id: str = str(meta.get("book_id", page_id.rsplit("_", 1)[0] if "_" in page_id else page_id))

    detections: dict = meta.get("detections", {}) or {}
    fasterrcnn: dict = detections.get("fasterrcnn", {}) or {}
    panel_count: int = len(fasterrcnn.get("panels", []) or [])
    char_count: int = len(fasterrcnn.get("characters", []) or [])

    # Image dimensions from metadata when available; 0 means "not recorded".
    width: int = int(meta.get("width", 0))
    height: int = int(meta.get("height", 0))

    return page_id, book_id, panel_count, char_count, width, height


def download(
    out_dir: Path,
    budget: int,
    max_per_book: int,
    min_panels: int,
    verbose: bool,
) -> dict:
    """Main streaming download loop. Returns the source-registry dict."""
    _require_imports()
    _ensure_hf_token()
    import datasets  # type: ignore[import-untyped]
    from PIL import Image  # type: ignore[import-untyped]

    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    # Load in streaming mode — no full-dataset download, no HF cache bloat.
    # trust_remote_code was removed in datasets>=3.0; standard Parquet datasets
    # no longer need it. Gated datasets require authentication via HF_TOKEN instead.
    print(f"Opening {DATASET_HF_REPO} in streaming mode …")
    stream = datasets.load_dataset(DATASET_HF_REPO, split="train", streaming=True)

    # Capture dataset info (revision/version) without iterating.
    hf_info = getattr(stream, "info", None)
    hf_version: str | None = str(hf_info.version) if hf_info and hf_info.version else None

    registry_entries: list[dict] = []
    saved = 0
    skipped_exists = 0
    skipped_panels = 0
    skipped_budget = 0
    book_counts: dict[str, int] = defaultdict(int)

    # Load existing registry to resume without re-hashing every file.
    existing_ids: set[str] = set()
    registry_path = out_dir / "source_registry.json"
    if registry_path.exists():
        try:
            old = json.loads(registry_path.read_text(encoding="utf-8"))
            existing_ids = {entry["page_id"] for entry in old.get("assets", [])}
            # Restore book counts from the prior run so per-book caps are honoured.
            for entry in old.get("assets", []):
                book_counts[entry["book_id"]] += 1
            print(f"Resuming: {len(existing_ids)} pages already saved.")
        except Exception:  # noqa: BLE001
            pass

    print(f"Budget: {budget} pages | max {max_per_book} per book | min {min_panels} panels")

    for example in stream:
        if saved >= budget:
            break

        page_id, book_id, panel_count, char_count, width, height = _extract_metadata(example)

        # Skip if already on disk from a previous run.
        if page_id in existing_ids:
            skipped_exists += 1
            continue

        # Skip near-blank pages (covers, title pages) that have no panel detections.
        if panel_count < min_panels:
            skipped_panels += 1
            if verbose:
                print(f"  skip (panels={panel_count}<{min_panels}) {page_id}")
            continue

        # Enforce per-book cap to ensure variety across titles.
        if book_counts[book_id] >= max_per_book:
            skipped_budget += 1
            continue

        # The dataset yields PIL Images under the "jpg" key.
        pil_image: Image.Image | None = example.get("jpg")
        if pil_image is None:
            if verbose:
                print(f"  skip (no image) {page_id}")
            continue

        # Derive actual dimensions from the image when metadata doesn't carry them.
        if width == 0 or height == 0:
            width, height = pil_image.size

        dest_path = pages_dir / f"{page_id}.jpg"

        try:
            # Save as JPEG (native format; keeps file sizes small for storage budget).
            pil_image.save(dest_path, format="JPEG", quality=90, optimize=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR saving {page_id}: {exc}", file=sys.stderr)
            continue

        checksum = _sha256_path(dest_path)
        file_size = dest_path.stat().st_size

        registry_entries.append({
            "source_dataset": SOURCE_DATASET_NAME,
            "page_id": page_id,
            "book_id": book_id,
            "filename": dest_path.name,
            "relative_path": str(dest_path.relative_to(out_dir)),
            "width": width,
            "height": height,
            "file_size_bytes": file_size,
            "sha256": checksum,
            "panel_count": panel_count,
            "character_count": char_count,
            # Curation hints derived from annotations — the /curate UI will
            # pre-populate bounding boxes from these when available.
            "has_panel_annotations": panel_count > 0,
            "has_character_annotations": char_count > 0,
        })
        book_counts[book_id] += 1
        saved += 1

        if verbose or saved % 50 == 0:
            mb = file_size / 1_048_576
            print(f"  [{saved}/{budget}] {page_id}  {width}×{height}  {mb:.1f} MB  panels={panel_count}")

    # Merge with any prior run's entries if resuming.
    prior_entries: list[dict] = []
    if registry_path.exists():
        try:
            old = json.loads(registry_path.read_text(encoding="utf-8"))
            prior_entries = [e for e in old.get("assets", []) if e["page_id"] not in {r["page_id"] for r in registry_entries}]
        except Exception:  # noqa: BLE001
            pass

    total_entries = prior_entries + registry_entries
    total_bytes = sum(e.get("file_size_bytes", 0) for e in total_entries)

    print(
        f"\nRun summary:"
        f"\n  Saved this run : {saved}"
        f"\n  Skipped (exists): {skipped_exists}"
        f"\n  Skipped (panels): {skipped_panels}"
        f"\n  Skipped (per-book cap): {skipped_budget}"
        f"\n  Total on disk  : {len(total_entries)} pages  ({total_bytes / 1_048_576:.1f} MB)"
        f"\n  Unique books   : {len(book_counts)}"
    )

    return {
        "source_dataset": SOURCE_DATASET_NAME,
        "hf_repo": DATASET_HF_REPO,
        "hf_version": hf_version,
        "downloaded_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "budget": budget,
        "max_per_book": max_per_book,
        "min_panels_filter": min_panels,
        "total_pages": len(total_entries),
        "total_bytes": total_bytes,
        "unique_books": len(book_counts),
        "terms": SOURCE_TERMS,
        "assets": total_entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stream a budget-capped sample of DCM-13k into the LineScout data tree.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=400,
        metavar="N",
        help="Maximum total pages to save (default: 400). Use 200 for a quick pilot.",
    )
    parser.add_argument(
        "--max-per-book",
        type=int,
        default=3,
        metavar="N",
        help=(
            "Maximum pages to save from any single comic book title (default: 3). "
            "Keeps variety across books; raise it if you want denser per-title coverage."
        ),
    )
    parser.add_argument(
        "--min-panels",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Skip pages with fewer than N detected panels (default: 2). "
            "Filters out covers and title pages that have no character references."
        ),
    )
    parser.add_argument(
        "--out",
        default="data/sources/dcm",
        metavar="DIR",
        help="Output directory (default: data/sources/dcm).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print every saved page.",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = download(
        out_dir=out_dir,
        budget=args.budget,
        max_per_book=args.max_per_book,
        min_panels=args.min_panels,
        verbose=args.verbose,
    )

    registry_path = out_dir / "source_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote source registry → {registry_path}")

    n = registry["total_pages"]
    print(
        f"\n{'─'*62}\n"
        f"Next steps for these {n} DCM pages:\n"
        f"  1. Open http://127.0.0.1:5173/curate and review each page:\n"
        f"       – panel bounding boxes are pre-filled from MAGI detections\n"
        f"       – assign primary scope (face_head, upper_body_clothing, …)\n"
        f"       – verify SFW and set quality 2–3\n"
        f"       – upgrade trace=True for verifiably public-domain native art\n"
        f"  2. Run the Colab GPU pipeline for line-art extraction + embedding:\n"
        f"       ml/colab/linescout_gpu_pipeline.ipynb\n"
        f"  3. Validate the resulting manifest:\n"
        f"       linescout-manifest validate data/gallery/<version>/manifest.json\n"
        f"  4. Point the API at the combined gallery:\n"
        f"       LINESCOUT_GALLERY_MANIFEST=data/gallery/<version>/manifest.json\n"
        f"{'─'*62}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

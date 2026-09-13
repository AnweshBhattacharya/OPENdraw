"""Build and load Clippy's frozen index over a validated manifest.

An index is a snapshot: it is only ever built fresh or reused verbatim,
never patched in place, so a comparator score from one evaluation run can't
quietly drift because an asset was reprocessed after the index was built.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from linescout_ml.manifest import Manifest, ManifestRecord

from clippy import CLIPPY_VERSION
from clippy.embeddings import ClippyEmbedding, embed_image

log = logging.getLogger(__name__)

INDEX_FILENAME = "clippy_index.json"


@dataclass(frozen=True)
class ClippyIndexEntry:
    asset_id: str
    line_art_path: str  # relative to the manifest's data root, for reporting
    embedding: ClippyEmbedding


@dataclass(frozen=True)
class ClippyIndex:
    """A frozen comparator index. Reload with :func:`load_index`, never edit."""

    clippy_version: str
    dataset_version: str
    manifest_content_hash: str
    entries: list[ClippyIndexEntry]

    def is_current_for(self, manifest: Manifest) -> bool:
        return (
            self.clippy_version == CLIPPY_VERSION
            and self.manifest_content_hash == manifest.content_hash()
        )


def _load_manifest(manifest_path: Path) -> Manifest:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return Manifest.model_validate(payload)


def _servable_line_art(record: ManifestRecord, data_root: Path) -> Path | None:
    """The line-art file for a record, or ``None`` if it isn't on disk.

    A missing file degrades that one asset out of the index (skip + log)
    rather than failing the whole build — mirrors the live gallery's
    disable-and-report rule for missing derivatives.
    """
    path = data_root / record.line_art_path
    return path if path.is_file() else None


def build_index(manifest_path: Path, data_root: Path | None = None) -> ClippyIndex:
    """Compute embeddings for every servable asset and freeze them.

    ``data_root`` defaults to the manifest's own directory, matching the
    synthetic fixture layout (``manifest.json`` next to ``line_art/``).
    """
    data_root = data_root or manifest_path.parent
    manifest = _load_manifest(manifest_path)

    entries: list[ClippyIndexEntry] = []
    for record in manifest.servable_records:
        line_art = _servable_line_art(record, data_root)
        if line_art is None:
            log.warning("clippy: skipping %s, line art missing on disk", record.asset_id)
            continue
        entries.append(
            ClippyIndexEntry(
                asset_id=record.asset_id,
                line_art_path=record.line_art_path,
                embedding=embed_image(line_art),
            )
        )

    return ClippyIndex(
        clippy_version=CLIPPY_VERSION,
        dataset_version=manifest.dataset_version,
        manifest_content_hash=manifest.content_hash(),
        entries=entries,
    )


def save_index(index: ClippyIndex, out_path: Path) -> None:
    payload = {
        "clippy_version": index.clippy_version,
        "dataset_version": index.dataset_version,
        "manifest_content_hash": index.manifest_content_hash,
        "entries": [
            {
                "asset_id": entry.asset_id,
                "line_art_path": entry.line_art_path,
                "embedding": entry.embedding.as_json(),
            }
            for entry in index.entries
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload), encoding="utf-8")


def load_index(path: Path) -> ClippyIndex:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ClippyIndex(
        clippy_version=payload["clippy_version"],
        dataset_version=payload["dataset_version"],
        manifest_content_hash=payload["manifest_content_hash"],
        entries=[
            ClippyIndexEntry(
                asset_id=raw["asset_id"],
                line_art_path=raw["line_art_path"],
                embedding=ClippyEmbedding.from_json(raw["embedding"]),
            )
            for raw in payload["entries"]
        ],
    )

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from clippy import CLIPPY_VERSION
from clippy.embeddings import edge_descriptor, embed_image, image_descriptor
from clippy.gallery import build_index, load_index, save_index
from clippy.scorer import rank, rank_query_image

FIXTURE_MANIFEST = (
    Path(__file__).resolve().parents[2] / "ml" / "fixtures" / "synthetic" / "manifest.json"
)


@pytest.fixture(scope="module")
def index():
    if not FIXTURE_MANIFEST.is_file():
        pytest.skip("synthetic fixture not present in this checkout")
    return build_index(FIXTURE_MANIFEST)


def test_descriptors_are_unit_normalized(index):
    entry = index.entries[0]
    assert np.linalg.norm(entry.embedding.image) == pytest.approx(1.0, abs=1e-4)
    assert np.linalg.norm(entry.embedding.edge) == pytest.approx(1.0, abs=1e-4)


def test_descriptors_are_deterministic(index):
    path = FIXTURE_MANIFEST.parent / index.entries[0].line_art_path
    first = embed_image(path)
    second = embed_image(path)
    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.edge, second.edge)


def test_index_covers_every_servable_asset(index):
    from clippy.gallery import _load_manifest

    manifest = _load_manifest(FIXTURE_MANIFEST)
    # Servable, not "every record": the fixture deliberately includes a
    # rejected/blocked row (review_rejected, quality_below_floor,
    # blocker_anatomy) that neither Clippy nor the live gallery should serve.
    assert len(index.entries) == len(manifest.servable_records)
    assert len(index.entries) < len(manifest.records)


def test_query_retrieves_itself_top1(index):
    query_entry = index.entries[5]
    query_path = FIXTURE_MANIFEST.parent / query_entry.line_art_path
    results = rank_query_image(index, query_path, top_k=3)
    assert results[0].asset_id == query_entry.asset_id
    assert results[0].score == pytest.approx(1.0, abs=1e-4)


def test_rank_orders_by_descending_score(index):
    query_path = FIXTURE_MANIFEST.parent / index.entries[0].line_art_path
    results = rank_query_image(index, query_path, top_k=len(index.entries))
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rank_respects_top_k(index):
    query = embed_image(FIXTURE_MANIFEST.parent / index.entries[0].line_art_path)
    results = rank(index, query, top_k=3)
    assert len(results) == 3


def test_save_and_load_round_trip(index, tmp_path):
    out = tmp_path / "clippy_index.json"
    save_index(index, out)
    reloaded = load_index(out)
    assert reloaded.clippy_version == CLIPPY_VERSION
    assert reloaded.manifest_content_hash == index.manifest_content_hash
    assert len(reloaded.entries) == len(index.entries)
    assert np.array_equal(reloaded.entries[0].embedding.image, index.entries[0].embedding.image)


def test_zero_ink_image_yields_zero_vector(tmp_path):
    from PIL import Image

    blank = tmp_path / "blank.png"
    Image.new("RGB", (256, 256), (255, 255, 255)).save(blank)
    descriptor = image_descriptor(blank)
    assert np.linalg.norm(descriptor) == pytest.approx(0.0, abs=1e-6)
    edges = edge_descriptor(blank)
    assert np.linalg.norm(edges) == pytest.approx(0.0, abs=1e-6)


def test_index_is_current_for_its_own_manifest(index):
    from clippy.gallery import _load_manifest

    manifest = _load_manifest(FIXTURE_MANIFEST)
    assert index.is_current_for(manifest)


def test_index_is_stale_after_manifest_content_change(index):
    """A manifest_content_hash mismatch must be detected, not silently reused."""
    from clippy.gallery import _load_manifest

    manifest = _load_manifest(FIXTURE_MANIFEST)
    stale = replace(index, manifest_content_hash="0" * 64)
    assert not stale.is_current_for(manifest)


def test_index_is_stale_after_clippy_version_bump(index):
    """A clippy_version mismatch must be detected, not silently reused."""
    from clippy.gallery import _load_manifest

    manifest = _load_manifest(FIXTURE_MANIFEST)
    stale = replace(index, clippy_version="clippy-0")
    assert not stale.is_current_for(manifest)

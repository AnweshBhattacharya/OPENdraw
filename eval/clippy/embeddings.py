"""Non-learned image and edge descriptors.

Two descriptors, concatenated at scoring time rather than here, so callers
can re-weight or inspect them separately:

``image_descriptor``
    A downsampled grayscale grid. Cheap global-appearance signal — mostly
    ink coverage and rough placement, the same kind of thing a thumbnail
    comparison would catch.

``edge_descriptor``
    A spatial grid of gradient-orientation histograms (a small, unlearned
    HOG). This is the "edge embedding" the Phase-I report calls for: it
    responds to line direction and contour layout rather than shading or
    tone, which suits line-art better than the raw grid alone.

Both are deterministic given the same input image and both L2-normalize to
unit vectors, so a zero-ink or otherwise degenerate image analyzed by
``PIL.Image`` will produce a saved zero vector rather than error out. Nothing
here is a stand-in for the MobileCLIP2 / DINOv2 branches in the Milestone 4
retrieval models — this stays intentionally simple.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

IMAGE_GRID = 32  # image_descriptor: IMAGE_GRID x IMAGE_GRID grayscale grid
EDGE_CELLS = 4  # edge_descriptor: EDGE_CELLS x EDGE_CELLS spatial cells
EDGE_BINS = 8  # ...each an EDGE_BINS-bin unsigned orientation histogram
EDGE_DESCRIPTOR_DIM = EDGE_CELLS * EDGE_CELLS * EDGE_BINS


@dataclass(frozen=True)
class ClippyEmbedding:
    """One asset's (or query's) frozen descriptor pair."""

    image: NDArray[np.float32]
    edge: NDArray[np.float32]

    def as_json(self) -> dict[str, list[float]]:
        return {"image": self.image.tolist(), "edge": self.edge.tolist()}

    @staticmethod
    def from_json(payload: dict[str, list[float]]) -> ClippyEmbedding:
        return ClippyEmbedding(
            image=np.asarray(payload["image"], dtype=np.float32),
            edge=np.asarray(payload["edge"], dtype=np.float32),
        )


def _l2_normalize(vector: NDArray[np.float32]) -> NDArray[np.float32]:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return vector
    return vector / norm


def _load_grayscale(path: Path) -> Image.Image:
    with Image.open(path) as handle:
        # Flatten transparency onto white first: a transparent import should
        # read as "blank paper," matching the live search preprocessing.
        rgba = handle.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        flattened = Image.alpha_composite(background, rgba).convert("RGB")
        return ImageOps.grayscale(flattened)


def image_descriptor(path: Path) -> NDArray[np.float32]:
    """Downsampled grayscale grid, ink-dark-high, unit-normalized."""
    grayscale = _load_grayscale(path)
    small = grayscale.resize((IMAGE_GRID, IMAGE_GRID), Image.Resampling.BILINEAR)
    values = np.asarray(small, dtype=np.float32) / 255.0
    # Invert so ink (dark strokes on white paper) contributes positive
    # weight — two heavily-inked sketches should look similar even if their
    # paper isn't pixel-identical.
    inked = 1.0 - values
    return _l2_normalize(inked.flatten())


def edge_descriptor(path: Path) -> NDArray[np.float32]:
    """Spatial grid of unsigned gradient-orientation histograms."""
    grayscale = _load_grayscale(path)
    # A fixed working resolution keeps cell boundaries and bin counts
    # comparable across assets of different native sizes.
    side = EDGE_CELLS * 16
    resized = grayscale.resize((side, side), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32)

    gy, gx = np.gradient(pixels)
    magnitude = np.hypot(gx, gy)
    # Unsigned orientation: a line's direction, not which side is darker.
    orientation = np.mod(np.arctan2(gy, gx), np.pi)

    cell_size = side // EDGE_CELLS
    histogram = np.zeros((EDGE_CELLS, EDGE_CELLS, EDGE_BINS), dtype=np.float32)
    bin_edges = np.linspace(0.0, np.pi, EDGE_BINS + 1)
    for row in range(EDGE_CELLS):
        for col in range(EDGE_CELLS):
            r0, r1 = row * cell_size, (row + 1) * cell_size
            c0, c1 = col * cell_size, (col + 1) * cell_size
            cell_orientation = orientation[r0:r1, c0:c1].ravel()
            cell_magnitude = magnitude[r0:r1, c0:c1].ravel()
            weighted, _ = np.histogram(cell_orientation, bins=bin_edges, weights=cell_magnitude)
            histogram[row, col] = weighted

    return _l2_normalize(histogram.flatten())


def embed_image(path: Path) -> ClippyEmbedding:
    return ClippyEmbedding(image=image_descriptor(path), edge=edge_descriptor(path))

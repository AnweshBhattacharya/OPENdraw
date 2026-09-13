"""Clippy: the frozen, non-learned comparator baseline.

Milestone 3 deliverable (see the Phase-I report, sections 2.3 and 4.8, and
``docs/contracts`` for how OPENdraw's own gallery loads). Clippy is *not* a
step towards LineScout's retrieval models — it exists so the Milestone 4
retrieval-improvement gate ("nDCG@8 improves >=20% relative to the frozen
shared-gallery comparator") has something concrete to beat.

It is deliberately simple:

* No learned weights, no torch, no GPU. Only image resampling and a Sobel
  gradient histogram — classic, deterministic computer vision.
* It never changes after being frozen for a given manifest. ``build_index``
  stamps the output with the manifest's ``content_hash`` and its own
  ``CLIPPY_VERSION``; a mismatch on either means "rebuild", not "reuse".
* It is headless: no canvas, no FastAPI app, no warmup/readiness state. A
  CLI (``clippy build-index`` / ``clippy query``) is the only interface.

This package intentionally does not import from ``services.api`` or
``linescout_ml.colab`` — it only depends on ``linescout_ml`` for the manifest
contract, so it stays buildable without the GPU pipeline installed.
"""

from __future__ import annotations

CLIPPY_VERSION = "clippy-1"

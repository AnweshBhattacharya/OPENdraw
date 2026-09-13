"""``clippy build-index`` and ``clippy query`` — headless, no server required."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from clippy.gallery import INDEX_FILENAME, build_index, load_index, save_index
from clippy.scorer import rank_query_image


def _cmd_build_index(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    data_root = Path(args.data_root) if args.data_root else None
    index = build_index(manifest_path, data_root=data_root)

    out_path = Path(args.out) if args.out else manifest_path.parent / INDEX_FILENAME
    save_index(index, out_path)

    print(  # noqa: T201 — CLI output, not app logging
        f"clippy: froze {len(index.entries)} assets from {manifest_path} "
        f"(dataset {index.dataset_version}) -> {out_path}"
    )
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    index = load_index(Path(args.index))
    results = rank_query_image(index, Path(args.image), top_k=args.top_k)
    payload = [
        {
            "asset_id": r.asset_id,
            "line_art_path": r.line_art_path,
            "score": round(r.score, 6),
            "image_similarity": round(r.image_similarity, 6),
            "edge_similarity": round(r.edge_similarity, 6),
        }
        for r in results
    ]
    print(json.dumps(payload, indent=2))  # noqa: T201
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clippy", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build-index", help="freeze embeddings for every servable asset in a manifest"
    )
    build.add_argument("manifest", help="path to manifest.json")
    build.add_argument(
        "--data-root",
        help="directory line_art_path is relative to (default: manifest's own directory)",
    )
    build.add_argument("--out", help=f"output path (default: alongside manifest, {INDEX_FILENAME})")
    build.set_defaults(func=_cmd_build_index)

    query = subparsers.add_parser("query", help="rank a frozen index against one query image")
    query.add_argument("--index", required=True, help="path to a frozen clippy_index.json")
    query.add_argument("image", help="path to a query line-art / snapshot PNG")
    query.add_argument("--top-k", type=int, default=8)
    query.set_defaults(func=_cmd_query)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

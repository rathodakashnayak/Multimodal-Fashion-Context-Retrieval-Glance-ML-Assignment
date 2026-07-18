"""
index_images.py
---------------
CLI script that encodes all images in a directory and stores them in ChromaDB.

Usage
-----
  python indexer/index_images.py --image_dir data/images/
  python indexer/index_images.py --image_dir data/images/ --reset

Arguments
---------
  --image_dir   Path to the folder containing images (jpg/png/webp).
  --batch_size  Number of images to encode per forward pass (default: 32).
  --reset       Clear the existing index before indexing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

# ── make imports work regardless of cwd ──
sys.path.insert(0, str(Path(__file__).parent.parent))

from indexer.feature_extractor import CLIPFeatureExtractor
from indexer.vector_store import VectorStore

# ─────────────────────────── helpers ─────────────────────────────

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def collect_image_paths(image_dir: str) -> List[Path]:
    """Recursively collect all image files under image_dir."""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    paths = [
        p
        for p in image_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file()
    ]
    paths.sort()
    return paths


def make_image_id(path: Path, image_dir: Path) -> str:
    """Create a stable unique ID from the relative path."""
    return str(path.relative_to(image_dir)).replace(os.sep, "/")


# ─────────────────────────── main ────────────────────────────────

def run_indexing(image_dir: str, batch_size: int = 32, reset: bool = False) -> None:
    """
    Full indexing pipeline:
    1. Collect all image paths.
    2. Encode with CLIP in batches.
    3. Store embeddings in ChromaDB.
    """
    # Step 1: collect images
    print(f"\n{'='*60}")
    print(f"  Indexing images from: {image_dir}")
    print(f"{'='*60}")
    paths = collect_image_paths(image_dir)
    print(f"[Indexer] Found {len(paths)} images.")

    if len(paths) == 0:
        print("[Indexer] No images found. Run data/download_dataset.py first.")
        sys.exit(1)

    # Step 2: filter already-indexed images
    store = VectorStore()
    if reset:
        store.reset()

    existing_ids_result = store.collection.get(include=[])
    existing_ids = set(existing_ids_result["ids"])

    image_dir_path = Path(image_dir)
    new_paths = [p for p in paths if make_image_id(p, image_dir_path) not in existing_ids]
    print(f"[Indexer] {len(existing_ids)} already indexed. {len(new_paths)} to index.")

    if len(new_paths) == 0:
        print("[Indexer] Nothing new to index.")
        return

    # Step 3: encode
    extractor = CLIPFeatureExtractor()
    embeddings = extractor.encode_images_batch(new_paths, batch_size=batch_size)

    # Step 4: prepare metadata
    ids = [make_image_id(p, image_dir_path) for p in new_paths]
    metadatas = [
        {
            "path": str(p.resolve()),
            "filename": p.name,
            "source": p.parent.name,  # sub-folder name as "source" tag
        }
        for p in new_paths
    ]

    # Step 5: store
    # ChromaDB has a max batch size; chunk if large
    CHROMA_BATCH_LIMIT = 5000
    for start in range(0, len(ids), CHROMA_BATCH_LIMIT):
        end = start + CHROMA_BATCH_LIMIT
        store.add(ids[start:end], embeddings[start:end], metadatas[start:end])

    print(f"\n[Indexer] Done! {store.count()} images indexed.")


# ─────────────────────────── CLI ─────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index fashion images into the ChromaDB vector store."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/images",
        help="Path to the folder containing images.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of images per encoding batch.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the existing index before indexing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_indexing(
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        reset=args.reset,
    )

"""
vector_store.py
---------------
ChromaDB wrapper for storing and querying CLIP image embeddings.

Design notes:
- Uses ChromaDB's persistent client so the index survives between runs.
- Stores each image as: embedding + metadata (path, filename, source).
- Query returns image paths + cosine-similarity scores.
- cosine similarity is the default distance metric in ChromaDB (cosine).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
import numpy as np

# ─────────────────────────── constants ───────────────────────────

COLLECTION_NAME = "fashion_images"
DEFAULT_DB_PATH = str(Path(__file__).parent.parent / "data" / "chroma_db")


# ─────────────────────────── store ───────────────────────────────

class VectorStore:
    """
    Thin wrapper around ChromaDB for fashion image embeddings.

    Usage
    -----
    store = VectorStore()
    store.add(ids, embeddings, metadatas)
    results = store.query(query_embedding, top_k=5)
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH, collection_name: str = COLLECTION_NAME):
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )
        print(f"[VectorStore] Initialized collection '{collection_name}' at '{db_path}'")
        print(f"[VectorStore] Current count: {self.collection.count()} images")

    # ── write ───────────────────────────────────────────────────────

    def add(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict],
    ) -> None:
        """
        Add image embeddings to the collection.

        Parameters
        ----------
        ids        : unique string IDs per image (e.g. filename)
        embeddings : (N, 512) numpy array of L2-normalized CLIP embeddings
        metadatas  : list of dicts with keys like 'path', 'filename', 'source'
        """
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )
        print(f"[VectorStore] Added {len(ids)} images. Total: {self.collection.count()}")

    def reset(self) -> None:
        """Delete all documents in the collection (useful for re-indexing)."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print("[VectorStore] Collection reset.")

    # ── read ────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Query the collection with a text/image embedding.

        Returns a list of dicts with keys: id, path, filename, source, distance, score.
        score = 1 - distance  (higher is better; cosine similarity in [0, 1])
        """
        kwargs: Dict = dict(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.collection.count()),
            include=["metadatas", "distances", "embeddings"],
        )
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        output = []
        for doc_id, metadata, distance, embedding in zip(
            results["ids"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["embeddings"][0],
        ):
            output.append(
                {
                    "id": doc_id,
                    "path": metadata.get("path", ""),
                    "filename": metadata.get("filename", ""),
                    "source": metadata.get("source", ""),
                    "distance": distance,
                    "score": 1.0 - distance,  # cosine similarity
                    "embedding": np.array(embedding),
                }
            )
        return output

    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray, List[Dict]]:
        """
        Retrieve all stored embeddings.
        Returns (ids, embeddings array, metadatas).
        Used by the attribute scorer for re-ranking.
        """
        result = self.collection.get(include=["embeddings", "metadatas"])
        ids = result["ids"]
        embeddings = np.array(result["embeddings"])
        metadatas = result["metadatas"]
        return ids, embeddings, metadatas

    def count(self) -> int:
        return self.collection.count()

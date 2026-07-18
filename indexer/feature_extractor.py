"""
feature_extractor.py
--------------------
CLIP-based feature extractor for fashion images.

Uses openai/clip-vit-base-patch32 via HuggingFace transformers.
Produces L2-normalized 512-dimensional embeddings for images and text.

Design notes:
- Image embeddings are computed once at index time (Part A).
- Text embeddings are computed at query time (Part B).
- Both are L2-normalized so cosine similarity == dot product.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ─────────────────────────── constants ───────────────────────────

MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512


# ─────────────────────────── extractor ───────────────────────────

class CLIPFeatureExtractor:
    """
    Wraps HuggingFace CLIP for both image and text encoding.

    Usage
    -----
    extractor = CLIPFeatureExtractor()
    img_emb   = extractor.encode_image("path/to/image.jpg")   # (512,)
    txt_emb   = extractor.encode_text("a yellow raincoat")     # (512,)
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Force offline mode: use cached model files without contacting HuggingFace
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

        print(f"[FeatureExtractor] Loading CLIP model '{model_name}' on {device} ...")
        try:
            self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
            self.model = CLIPModel.from_pretrained(model_name, local_files_only=True).to(device)
        except OSError:
            # Cache not found — try online (first run)
            print("[FeatureExtractor] Cache miss — downloading from HuggingFace ...")
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()
        print("[FeatureExtractor] Model loaded - OK")

    # ── image ──────────────────────────────────────────────────────

    def encode_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Encode a single image file → L2-normalized numpy array of shape (512,).
        """
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        features = self._extract_tensor(features)
        return self._normalize(features).cpu().numpy().squeeze()

    def encode_images_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode a list of image files in batches.
        Returns array of shape (N, 512).
        """
        from tqdm import tqdm

        all_embeddings = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
            batch_paths = image_paths[i : i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(
                self.device
            )
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
            features = self._extract_tensor(features)
            embeddings = self._normalize(features).cpu().numpy()
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    # ── text ───────────────────────────────────────────────────────

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single text string → L2-normalized numpy array of shape (512,).
        """
        inputs = self.processor(
            text=[text], return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        features = self._extract_tensor(features)
        return self._normalize(features).cpu().numpy().squeeze()

    def encode_texts_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of text strings.
        Returns array of shape (N, 512).
        """
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        features = self._extract_tensor(features)
        return self._normalize(features).cpu().numpy()

    def encode_text_ensemble(self, templates: List[str]) -> np.ndarray:
        """
        Prompt ensembling: encode multiple text templates and return their
        L2-normalized average embedding.

        Why this works
        --------------
        CLIP is sensitive to how a concept is phrased. A single prompt
        like "a blue shirt" may emphasize different CLIP directions than
        "photo of someone wearing blue clothing". Averaging across diverse
        phrasings produces a more robust, centred embedding.

        This is the same technique used in CLIP's original paper for its
        80-template zero-shot ImageNet classifier.

        Parameters
        ----------
        templates : list of text strings, all describing the same concept

        Returns
        -------
        (512,) L2-normalized numpy array (ensemble average)
        """
        embeddings = self.encode_texts_batch(templates)      # (N, 512)
        avg = embeddings.mean(axis=0)                        # (512,)
        avg = avg / (np.linalg.norm(avg) + 1e-8)            # re-normalize
        return avg

    # ── internal ───────────────────────────────────────────────────

    @staticmethod
    def _extract_tensor(features) -> torch.Tensor:
        """
        Unwrap model output objects into a plain tensor.

        Transformers 5.x can return BaseModelOutputWithPooling or similar
        objects from get_text_features / get_image_features in some configs.
        This method handles both plain tensors and wrapped outputs.
        """
        if isinstance(features, torch.Tensor):
            return features
        # BaseModelOutputWithPooling and similar dataclasses
        if hasattr(features, "pooler_output") and features.pooler_output is not None:
            return features.pooler_output
        if hasattr(features, "last_hidden_state"):
            return features.last_hidden_state[:, 0, :]  # CLS token
        # Last resort: first element (tuple/list output)
        if hasattr(features, "__getitem__"):
            return features[0]
        raise TypeError(f"Cannot extract tensor from {type(features)}")

    @staticmethod
    def _normalize(tensor: torch.Tensor) -> torch.Tensor:
        """L2-normalize along last dimension."""
        return tensor / tensor.norm(dim=-1, keepdim=True)

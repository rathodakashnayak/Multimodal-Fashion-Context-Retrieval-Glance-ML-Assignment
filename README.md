# Multimodal Fashion & Context Retrieval

> **Glance ML Internship Assignment** — Intelligent image search for fashion using CLIP + Attribute-Aware Re-ranking

---

## Overview

This system retrieves fashion images from a dataset based on **natural language descriptions**. It understands:
- **What** someone is wearing (garment type)
- **Where** they are (environment/setting)
- **Color** of their clothing
- **Vibe/Style** of their outfit

### Why Not Just CLIP?

Vanilla CLIP struggles with **compositional queries** — e.g., distinguishing *"red tie and white shirt"* from *"white tie and red shirt"*. Our approach decomposes queries into specialist sub-queries and fuses scores:

```
Query → Parser → Sub-queries (color, clothing, location, style)
                      ↓
               CLIP encodes each sub-query
                      ↓
         Global CLIP Score + Attribute Scores
                      ↓
              Weighted Fusion → Re-ranked Top-K
```

---

## Project Structure

```
Glance ML Interview/
├── indexer/
│   ├── feature_extractor.py   # CLIP image/text encoding
│   ├── vector_store.py        # ChromaDB wrapper
│   └── index_images.py        # CLI: index all images
├── retriever/
│   ├── query_parser.py        # Decompose query → attributes
│   ├── scorer.py              # Dual-scoring + weighted fusion
│   └── retrieve.py            # CLI: query the index
├── data/
│   ├── download_dataset.py    # Download fashion images
│   └── images/                # Image store (created on download)
├── app/
│   └── demo.py                # Gradio web demo
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download dataset (~500 images)

```bash
python data/download_dataset.py
# Uses HuggingFace Fashionpedia first, Unsplash as fallback
# To force Unsplash: python data/download_dataset.py --source unsplash
```

### 3. Index images (Part A)

```bash
python indexer/index_images.py --image_dir data/images/
# To re-index from scratch: add --reset
```

### 4. Run a query (Part B)

```bash
python retriever/retrieve.py --query "A person in a bright yellow raincoat" --top_k 5
```

### 5. Run all evaluation queries

```bash
python retriever/retrieve.py --run_eval_queries --top_k 5
```

### 6. Launch web demo

```bash
python app/demo.py
# Opens at http://localhost:7860
```

---

## Architecture Deep Dive

### Part A: The Indexer

| Component | Description |
|-----------|-------------|
| `CLIPFeatureExtractor` | Wraps `openai/clip-vit-base-patch32`. Encodes images to 512-d L2-normalized embeddings in batches. |
| `VectorStore` | ChromaDB persistent collection with cosine similarity (HNSW index). Stores embeddings + metadata (path, filename, source folder). |
| `index_images.py` | CLI script. Incrementally indexes new images only (skips already-indexed). |

### Part B: The Retriever

| Component | Description |
|-----------|-------------|
| `QueryParser` | Rule-based decomposition of queries into `color`, `clothing`, `location`, `style` attributes using keyword vocabularies. Fast, offline, deterministic. |
| `AttributeScorer` | The core innovation. Encodes each sub-query with CLIP and computes cosine similarity against all candidate embeddings separately, then fuses with learned weights. |
| `retrieve.py` | CLI with output saving and batch evaluation. |

### Fusion Weights

| Attribute | Weight | Rationale |
|-----------|--------|-----------|
| `global` | 0.40 | Semantic anchor for overall meaning |
| `color + clothing` | 0.35 | Most discriminative for fashion |
| `clothing` | 0.25 | Strong category signal |
| `location` | 0.15 | Context signal |
| `style` | 0.10 | Soft signal |

*Weights can be tuned in `retriever/scorer.py → DEFAULT_WEIGHTS`.*

---

## Evaluation Queries

| # | Query | Key Attributes |
|---|-------|---------------|
| 1 | "A person in a bright yellow raincoat." | color=yellow, clothing=raincoat |
| 2 | "Professional business attire inside a modern office." | clothing=business, location=office |
| 3 | "Someone wearing a blue shirt sitting on a park bench." | color=blue, clothing=shirt, location=park |
| 4 | "Casual weekend outfit for a city walk." | style=casual, location=city |
| 5 | "A red tie and a white shirt in a formal setting." | color=red+white, clothing=tie+shirt, location=formal |

---

## Scalability

| Scale | Strategy |
|-------|----------|
| Current (~1K images) | Direct ChromaDB query |
| 100K images | ChromaDB HNSW index keeps query time to ~10ms |
| 1M images | ChromaDB HNSW + candidate pool filtering (default: top-150 pre-filter) |
| 10M+ images | Shard ChromaDB across machines, or migrate to Pinecone/Weaviate |

The scorer's **candidate pool** design means attribute re-ranking is always O(150) regardless of collection size. Only the initial HNSW lookup scales with N (logarithmically).

---

## Future Work

### 1. Locations & Weather Extension

- Add a **location classifier** (GeoClip or a fine-tuned ResNet) to tag images with city/location metadata at index time.
- Store `city`, `weather`, `season` as ChromaDB metadata fields.
- Extend `QueryParser` to extract location/weather terms.
- Filter ChromaDB results by metadata before re-ranking.

### 2. Improving Precision

| Improvement | Impact |
|-------------|--------|
| Fine-tune CLIP on FashionPedia with attribute labels | +High compositionality |
| Add a fashion-specific reranker (cross-encoder) | +High precision |
| Segment garments separately (SAM + CLIP per region) | +High color/garment isolation |
| Use BLIP-2 or LLaVA for image captioning at index time | +Better semantic metadata |
| Human feedback loop (RLHF-style) | +Personalization |

---

## Dependencies

- `transformers` — CLIP model from HuggingFace
- `chromadb` — vector database with HNSW index
- `torch` — model inference
- `Pillow` — image processing
- `gradio` — web demo
- `datasets` — HuggingFace dataset download

---

## Author

Built for the **Glance ML Internship Assignment**: Multimodal Fashion & Context Retrieval.

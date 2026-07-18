# Multimodal Fashion & Context Retrieval Glance ML Assignment

This repository contains a multimodal fashion image retrieval system that combines CLIP embeddings with attribute-aware re-ranking for natural language fashion queries.

## What this project does

The system can:
- retrieve fashion images from a local dataset using natural language queries
- understand clothing, color, location, and style cues
- re-rank results using attribute-aware fusion for better precision
- run through a simple Gradio web demo

## Project structure

- `indexer/` — image indexing and embedding generation
- `retriever/` — query parsing, scoring, and retrieval logic
- `data/` — dataset download and image storage
- `app/` — web demo entry point

## Setup

1. Create a Python environment
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download or generate the image dataset:
   ```bash
   python data/download_dataset.py
   ```
4. Index the images:
   ```bash
   python indexer/index_images.py --image_dir data/images/
   ```
5. Run a sample query:
   ```bash
   python retriever/retrieve.py --query "A person in a bright yellow raincoat" --top_k 5
   ```
6. Launch the demo:
   ```bash
   python app/demo.py
   ```

## Demo

The Gradio demo runs locally at:
- http://127.0.0.1:7861

## Notes

This project was built as part of the Glance ML internship assignment and focuses on a practical retrieval pipeline rather than a large-scale production system.


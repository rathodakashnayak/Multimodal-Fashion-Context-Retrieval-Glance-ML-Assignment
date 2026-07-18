"""
demo.py
-------
Gradio web demo for the Multimodal Fashion & Context Retrieval system.

Features
--------
- Type any natural language query
- View top-k retrieved images in a gallery
- See relevance scores and attribute breakdown per result
- Run all 5 evaluation queries with one click

Usage
-----
  python app/demo.py
  # Opens at http://localhost:7860
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr

from retriever.retrieve import EVAL_QUERIES
from retriever.scorer import AttributeScorer

# ─────────────────────────── global state ────────────────────────

# Load scorer once (CLIP model is expensive to reload)
_scorer: AttributeScorer | None = None


def get_scorer() -> AttributeScorer:
    global _scorer
    if _scorer is None:
        _scorer = AttributeScorer()
    return _scorer


# ─────────────────────────── logic ───────────────────────────────

def search_fashion(query: str, top_k: int) -> tuple:
    """
    Main search function called by Gradio.
    Returns (gallery_images, results_json_str).
    """
    if not query.strip():
        return [], "Please enter a query."

    scorer = get_scorer()
    results = scorer.search(query.strip(), top_k=int(top_k))

    if not results:
        return [], json.dumps(
            {"error": "No results. Ensure images are indexed first."}, indent=2
        )

    # Build gallery: list of (image_path, caption) tuples
    gallery = []
    for rank, r in enumerate(results, 1):
        path = r["path"]
        if Path(path).exists():
            caption = (
                f"#{rank} · Score: {r['final_score']:.3f}\n"
                f"File: {r['filename']}\n"
                f"Global: {r['breakdown'].get('global', 0):.3f}"
            )
            gallery.append((path, caption))

    # Build JSON breakdown for the info panel
    breakdown_data = []
    for rank, r in enumerate(results, 1):
        breakdown_data.append(
            {
                "rank": rank,
                "filename": r["filename"],
                "final_score": round(r["final_score"], 4),
                "breakdown": {k: round(v, 4) for k, v in r["breakdown"].items()},
            }
        )
    info_json = json.dumps(breakdown_data, indent=2)

    return gallery, info_json


def run_eval_query(query_idx: int, top_k: int) -> tuple:
    """Run a pre-defined evaluation query by index."""
    query = EVAL_QUERIES[int(query_idx)]
    gallery, info = search_fashion(query, top_k)
    return query, gallery, info


# ─────────────────────────── UI ──────────────────────────────────

CSS = """
.gradio-container {
    font-family: 'Inter', sans-serif;
    background: #0f0f13;
    color: #e0e0e0;
}
#title {
    text-align: center;
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5em;
    font-weight: 800;
    margin-bottom: 0.2em;
}
#subtitle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 1.5em;
    font-size: 1em;
}
.search-btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1.1em !important;
}
"""

EVAL_QUERY_LABELS = [f"Q{i+1}: {q[:55]}" for i, q in enumerate(EVAL_QUERIES)]


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Fashion Retrieval Demo") as demo:
        gr.HTML('<h1 id="title">🔍 Fashion Retrieval</h1>')
        gr.HTML(
            '<p id="subtitle">Multimodal search engine · CLIP + Attribute-Aware Re-ranking</p>'
        )

        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Natural Language Query",
                    placeholder='e.g., "A person in a bright yellow raincoat"',
                    lines=2,
                    elem_id="query-box",
                )
                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=6,
                        step=1,
                        label="Top K results",
                    )
                    search_btn = gr.Button("🔍 Search", elem_classes=["search-btn"])

            with gr.Column(scale=1):
                gr.Markdown("### 📋 Evaluation Queries")
                eval_selector = gr.Radio(
                    choices=EVAL_QUERY_LABELS,
                    label="Assignment evaluation queries",
                    value=None,
                )
                run_eval_btn = gr.Button("▶ Run Selected Query")

        gr.Markdown("---")
        gr.Markdown("### 🖼️ Results")

        gallery = gr.Gallery(
            label="Retrieved Images",
            columns=3,
            height=500,
            show_label=False,
            object_fit="cover",
        )

        with gr.Accordion("📊 Score Breakdown (JSON)", open=False):
            info_output = gr.Code(language="json", label="Per-result scores")

        # ── event handlers ─────────────────────────────────────────

        search_btn.click(
            fn=search_fashion,
            inputs=[query_input, top_k_slider],
            outputs=[gallery, info_output],
        )
        query_input.submit(
            fn=search_fashion,
            inputs=[query_input, top_k_slider],
            outputs=[gallery, info_output],
        )

        def on_eval_run(eval_label: str, top_k: int):
            if eval_label is None:
                return gr.update(), [], ""
            idx = EVAL_QUERY_LABELS.index(eval_label)
            query = EVAL_QUERIES[idx]
            imgs, info = search_fashion(query, top_k)
            return query, imgs, info

        run_eval_btn.click(
            fn=on_eval_run,
            inputs=[eval_selector, top_k_slider],
            outputs=[query_input, gallery, info_output],
        )

        gr.Markdown(
            "---\n"
            "**Architecture**: CLIP ViT-B/32 + Attribute Decomposition + Weighted Fusion · "
            "Vector DB: ChromaDB (cosine similarity) · "
            "[GitHub](https://github.com/)"
        )

    return demo


# ─────────────────────────── entrypoint ──────────────────────────

if __name__ == "__main__":
    print("[Demo] Starting Fashion Retrieval Demo …")
    print("[Demo] Make sure images are indexed: python indexer/index_images.py")
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
        css=CSS,
    )

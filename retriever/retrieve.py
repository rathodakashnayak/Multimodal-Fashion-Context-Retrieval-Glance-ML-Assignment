"""
retrieve.py
-----------
CLI script for querying the fashion image index.

Usage
-----
  python retriever/retrieve.py --query "A person in a bright yellow raincoat"
  python retriever/retrieve.py --query "Casual weekend outfit" --top_k 10
  python retriever/retrieve.py --run_eval_queries

Arguments
---------
  --query         Natural language search query.
  --top_k         Number of results to return (default: 5).
  --run_eval_queries  Run all 5 evaluation queries from the assignment.
  --output_dir    If set, copies top-k result images to this directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retriever.scorer import AttributeScorer

# ─────────────────────────── evaluation queries ───────────────────

EVAL_QUERIES = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
]

# ─────────────────────────── display ─────────────────────────────

def print_results(query: str, results: list, top_k: int) -> None:
    print(f"\n{'='*70}")
    print(f"  Query: \"{query}\"")
    print(f"{'='*70}")
    if not results:
        print("  No results found. Make sure images are indexed.")
        return
    print(f"  Top {min(top_k, len(results))} results:\n")
    for rank, r in enumerate(results, 1):
        score_str = f"{r['final_score']:.4f}"
        attr_str = ", ".join(
            f"{k}={v:.3f}" for k, v in r["breakdown"].items() if k != "global"
        )
        print(f"  [{rank}] {r['filename']}")
        print(f"       Score: {score_str} (global={r['breakdown'].get('global', 0):.3f} | {attr_str})")
        print(f"       Path: {r['path']}")


def save_results(query: str, results: list, output_dir: str) -> None:
    """Copy top-k result images to output_dir and write a JSON summary."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = []
    for rank, r in enumerate(results, 1):
        src = Path(r["path"])
        if src.exists():
            dst = out / f"{rank:02d}_{src.name}"
            shutil.copy2(src, dst)
        summary.append(
            {
                "rank": rank,
                "filename": r["filename"],
                "final_score": r["final_score"],
                "breakdown": r["breakdown"],
                "path": r["path"],
            }
        )

    summary_path = out / "results.json"
    with open(summary_path, "w") as f:
        json.dump({"query": query, "results": summary}, f, indent=2)
    print(f"\n  Results saved to: {out}")


# ─────────────────────────── main ────────────────────────────────

def run_query(query: str, top_k: int = 5, output_dir: str | None = None) -> list:
    scorer = AttributeScorer()
    results = scorer.search(query, top_k=top_k)
    print_results(query, results, top_k)
    if output_dir:
        save_results(query, results, output_dir)
    return results


def run_eval_queries(top_k: int = 5) -> None:
    """Run all 5 assignment evaluation queries."""
    scorer = AttributeScorer()
    print("\n" + "="*70)
    print("  EVALUATION: Running all 5 assignment queries")
    print("="*70)
    for i, query in enumerate(EVAL_QUERIES, 1):
        results = scorer.search(query, top_k=top_k)
        print_results(query, results, top_k)


# ─────────────────────────── CLI ─────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve fashion images using natural language queries."
    )
    parser.add_argument("--query", type=str, help="Natural language search query.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return.")
    parser.add_argument(
        "--run_eval_queries",
        action="store_true",
        help="Run all 5 assignment evaluation queries.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="If set, copies top-k result images here and writes results.json.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.run_eval_queries:
        run_eval_queries(top_k=args.top_k)
    elif args.query:
        run_query(query=args.query, top_k=args.top_k, output_dir=args.output_dir)
    else:
        print("Error: provide --query or --run_eval_queries")
        sys.exit(1)

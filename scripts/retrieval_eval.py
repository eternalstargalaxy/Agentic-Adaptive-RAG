from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph.retrieval_eval_utils import evaluate_sample, load_jsonl, summarize_by_field
from graph.retrieval_metrics import summarize_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation.")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to a JSONL retrieval evaluation dataset.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Cutoff used for Recall/NDCG/MRR.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/retrieval_eval_summary.json"),
        help="Where to store the JSON summary.",
    )
    args = parser.parse_args()

    from ingestion import get_hybrid_retriever

    dataset = load_jsonl(args.dataset)
    retriever = get_hybrid_retriever()
    metric_rows = []
    numeric_metric_rows: List[Dict[str, float]] = []
    for sample in dataset:
        metrics = evaluate_sample(retriever, sample, args.k)
        metric_rows.append(
            {
                "query": sample["query"],
                "query_type": sample.get("query_type", "unknown"),
                **metrics,
            }
        )
        numeric_metric_rows.append(
            {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
        )
    summary = summarize_metrics(numeric_metric_rows)
    by_query_type = summarize_by_field(metric_rows, "query_type")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "k": args.k,
                "num_samples": len(dataset),
                "summary": summary,
                "by_query_type": by_query_type,
                "per_sample": metric_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

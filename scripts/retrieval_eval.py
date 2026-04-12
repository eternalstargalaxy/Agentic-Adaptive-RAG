from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from graph.retrieval_metrics import mrr_at_k, ndcg_at_k, recall_at_k, summarize_metrics
from ingestion import get_hybrid_retriever


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _document_identifier(document: Any) -> str:
    metadata = getattr(document, "metadata", {}) or {}
    return (
        metadata.get("url")
        or metadata.get("source")
        or getattr(document, "page_content", "")[:120]
    )


def build_relevance_flags(documents: List[Any], sample: Dict[str, Any]) -> List[int]:
    relevant_ids = set(sample.get("relevant_ids", []))
    relevant_sources = set(sample.get("relevant_sources", []))
    relevant_keywords = [keyword.lower() for keyword in sample.get("relevant_keywords", [])]

    flags = []
    for document in documents:
        identifier = _document_identifier(document)
        source = (getattr(document, "metadata", {}) or {}).get("source", "")
        content = getattr(document, "page_content", "").lower()

        is_relevant = (
            identifier in relevant_ids
            or source in relevant_sources
            or any(keyword in content for keyword in relevant_keywords)
        )
        flags.append(1 if is_relevant else 0)
    return flags


def evaluate_sample(retriever: Any, sample: Dict[str, Any], k: int) -> Dict[str, float]:
    documents = retriever.invoke(sample["query"])
    flags = build_relevance_flags(documents, sample)

    return {
        "recall@k": recall_at_k(flags, k),
        "ndcg@k": ndcg_at_k(flags, k),
        "mrr@k": mrr_at_k(flags, k),
    }


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

    dataset = load_jsonl(args.dataset)
    retriever = get_hybrid_retriever()
    metric_rows = [evaluate_sample(retriever, sample, args.k) for sample in dataset]
    summary = summarize_metrics(metric_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "k": args.k,
                "num_samples": len(dataset),
                "summary": summary,
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

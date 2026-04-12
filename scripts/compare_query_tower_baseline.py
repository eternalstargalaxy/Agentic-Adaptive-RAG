from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph.embeddings import BGEM3BiEncoderEmbeddings
from graph.retrieval_eval_utils import (
    build_relevance_flags,
    document_identifier,
    evaluate_documents,
    load_jsonl,
    summarize_by_field,
)
from graph.retrieval_metrics import summarize_metrics


def _evaluate_retriever(
    retriever: Any,
    sample: Dict[str, Any],
    k: int,
) -> Dict[str, Any]:
    documents = retriever.invoke(sample["query"])
    metrics = evaluate_documents(documents, sample, k)
    flags = build_relevance_flags(documents, sample)

    return {
        **metrics,
        "retrieved_ids_topk": [document_identifier(doc) for doc in documents[:k]],
        "retrieved_sources_topk": [
            (getattr(doc, "metadata", {}) or {}).get("source", "")
            for doc in documents[:k]
        ],
        "flags_topk": flags[:k],
    }


def _prefix_metrics(prefix: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in payload.items():
        result[f"{prefix}_{key}"] = value
    return result


def _diff_metric(after: Any, before: Any) -> float | None:
    if after is None or before is None:
        return None
    return float(after) - float(before)


def _delta_summary(
    before_summary: Dict[str, float],
    after_summary: Dict[str, float],
) -> Dict[str, float]:
    keys = sorted(set(before_summary) | set(after_summary))
    return {
        key: after_summary.get(key, 0.0) - before_summary.get(key, 0.0)
        for key in keys
    }


def _group_comparison(rows: Sequence[Dict[str, Any]], field: str) -> Dict[str, Dict[str, Any]]:
    before_rows: List[Dict[str, Any]] = []
    after_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []

    for row in rows:
        group = row.get(field, "unknown")
        before_rows.append(
            {
                field: group,
                "recall@k": row["baseline_recall@k"],
                "ndcg@k": row["baseline_ndcg@k"],
                "mrr@k": row["baseline_mrr@k"],
                "hit@k": row["baseline_hit@k"],
                "top1_hit": row["baseline_top1_hit"],
            }
        )
        after_rows.append(
            {
                field: group,
                "recall@k": row["adapted_recall@k"],
                "ndcg@k": row["adapted_ndcg@k"],
                "mrr@k": row["adapted_mrr@k"],
                "hit@k": row["adapted_hit@k"],
                "top1_hit": row["adapted_top1_hit"],
            }
        )
        delta_rows.append(
            {
                field: group,
                "delta_recall@k": row["delta_recall@k"],
                "delta_ndcg@k": row["delta_ndcg@k"],
                "delta_mrr@k": row["delta_mrr@k"],
                "recovered_from_miss": row["recovered_from_miss"],
                "improved_top1": row["improved_top1"],
                "regressed_hit@k": row["regressed_hit@k"],
            }
        )

    before_grouped = summarize_by_field(before_rows, field)
    after_grouped = summarize_by_field(after_rows, field)
    delta_grouped = summarize_by_field(delta_rows, field)
    counts: Dict[str, int] = {}
    for row in rows:
        group = row.get(field, "unknown")
        counts[group] = counts.get(group, 0) + 1

    grouped: Dict[str, Dict[str, Any]] = {}
    for group in counts:
        grouped[group] = {
            "count": counts[group],
            "baseline": before_grouped.get(group, {}),
            "adapted": after_grouped.get(group, {}),
            "delta": delta_grouped.get(group, {}),
        }
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare retrieval baseline before and after query tower LoRA."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/query_tower_baseline_comparison.json"),
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--dense-k", type=int, default=10)
    parser.add_argument("--sparse-k", type=int, default=10)
    parser.add_argument("--model-name", default="BAAI/bge-m3")
    parser.add_argument(
        "--query-instruction",
        default="Represent this medical query for retrieving supporting evidence: ",
    )
    args = parser.parse_args()

    from ingestion import HybridRetriever, get_local_documents, open_vectorstore

    dataset = load_jsonl(args.dataset)
    base_embeddings = BGEM3BiEncoderEmbeddings(
        model_name=args.model_name,
        query_instruction=args.query_instruction,
    )
    adapted_embeddings = BGEM3BiEncoderEmbeddings(
        model_name=args.model_name,
        query_adapter_path=str(args.adapter_path),
        query_instruction=args.query_instruction,
    )

    documents = get_local_documents(embedding_function=base_embeddings)
    baseline_retriever = HybridRetriever(
        vectorstore=open_vectorstore(embedding_function=base_embeddings),
        documents=documents,
        dense_k=args.dense_k,
        sparse_k=args.sparse_k,
    )
    adapted_retriever = HybridRetriever(
        vectorstore=open_vectorstore(embedding_function=adapted_embeddings),
        documents=documents,
        dense_k=args.dense_k,
        sparse_k=args.sparse_k,
    )

    rows: List[Dict[str, Any]] = []
    baseline_metric_rows: List[Dict[str, float]] = []
    adapted_metric_rows: List[Dict[str, float]] = []

    for sample in dataset:
        baseline = _evaluate_retriever(baseline_retriever, sample, args.k)
        adapted = _evaluate_retriever(adapted_retriever, sample, args.k)

        baseline_metrics = {
            "recall@k": float(baseline["recall@k"]),
            "ndcg@k": float(baseline["ndcg@k"]),
            "mrr@k": float(baseline["mrr@k"]),
            "hit@k": float(baseline["hit@k"]),
            "top1_hit": float(baseline["top1_hit"]),
        }
        adapted_metrics = {
            "recall@k": float(adapted["recall@k"]),
            "ndcg@k": float(adapted["ndcg@k"]),
            "mrr@k": float(adapted["mrr@k"]),
            "hit@k": float(adapted["hit@k"]),
            "top1_hit": float(adapted["top1_hit"]),
        }
        baseline_metric_rows.append(baseline_metrics)
        adapted_metric_rows.append(adapted_metrics)

        delta_recall = _diff_metric(adapted["recall@k"], baseline["recall@k"])
        delta_ndcg = _diff_metric(adapted["ndcg@k"], baseline["ndcg@k"])
        delta_mrr = _diff_metric(adapted["mrr@k"], baseline["mrr@k"])

        rows.append(
            {
                "query": sample["query"],
                "query_type": sample.get("query_type", "unknown"),
                "medical_entities": sample.get("medical_entities", []),
                **_prefix_metrics("baseline", baseline),
                **_prefix_metrics("adapted", adapted),
                "delta_recall@k": delta_recall,
                "delta_ndcg@k": delta_ndcg,
                "delta_mrr@k": delta_mrr,
                "recovered_from_miss": int(
                    baseline["hit@k"] == 0 and adapted["hit@k"] == 1
                ),
                "improved_top1": int(
                    baseline["top1_hit"] == 0 and adapted["top1_hit"] == 1
                ),
                "regressed_hit@k": int(
                    baseline["hit@k"] == 1 and adapted["hit@k"] == 0
                ),
                "relevant_hit_gain": int(adapted["relevant_hits@k"])
                - int(baseline["relevant_hits@k"]),
            }
        )

    baseline_summary = summarize_metrics(baseline_metric_rows)
    adapted_summary = summarize_metrics(adapted_metric_rows)
    comparison = {
        "dataset": str(args.dataset),
        "adapter_path": str(args.adapter_path),
        "k": args.k,
        "dense_k": args.dense_k,
        "sparse_k": args.sparse_k,
        "num_samples": len(dataset),
        "baseline_summary": baseline_summary,
        "adapted_summary": adapted_summary,
        "delta_summary": _delta_summary(baseline_summary, adapted_summary),
        "gain_breakdown": {
            "improved_samples": sum(
                1
                for row in rows
                if (row["delta_ndcg@k"] or 0) > 0 or (row["delta_mrr@k"] or 0) > 0
            ),
            "recovered_from_miss": sum(row["recovered_from_miss"] for row in rows),
            "improved_top1": sum(row["improved_top1"] for row in rows),
            "regressed_samples": sum(row["regressed_hit@k"] for row in rows),
        },
        "by_query_type": _group_comparison(rows, "query_type"),
        "per_sample": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(comparison["delta_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

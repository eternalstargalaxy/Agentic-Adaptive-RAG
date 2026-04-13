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
from graph.prompt_defaults import DEFAULT_QUERY_TOWER_INSTRUCTION
from graph.retrieval_eval_utils import (
    build_relevance_flags,
    document_identifier,
    evaluate_documents_with_cutoffs,
    load_jsonl,
    summarize_by_field,
)
from graph.retrieval_metrics import summarize_metrics


def _evaluate_retriever(
    retriever: Any,
    sample: Dict[str, Any],
    recall_k: int,
    ndcg_k: int,
    mrr_k: int,
) -> Dict[str, Any]:
    documents = retriever.invoke(sample["query"])
    max_k = max(recall_k, ndcg_k, mrr_k)
    metrics = evaluate_documents_with_cutoffs(
        documents,
        sample,
        recall_k=recall_k,
        ndcg_k=ndcg_k,
        mrr_k=mrr_k,
    )
    flags = build_relevance_flags(documents, sample)

    return {
        **metrics,
        "retrieved_ids_topk": [document_identifier(doc) for doc in documents[:max_k]],
        "retrieved_sources_topk": [
            (getattr(doc, "metadata", {}) or {}).get("source", "")
            for doc in documents[:max_k]
        ],
        "flags_topk": flags[:max_k],
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


def _group_comparison(
    rows: Sequence[Dict[str, Any]],
    field: str,
    recall_key: str,
    ndcg_key: str,
    mrr_key: str,
    hit_key: str,
) -> Dict[str, Dict[str, Any]]:
    before_rows: List[Dict[str, Any]] = []
    after_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []

    for row in rows:
        group = row.get(field, "unknown")
        before_rows.append(
            {
                field: group,
                recall_key: row[f"baseline_{recall_key}"],
                ndcg_key: row[f"baseline_{ndcg_key}"],
                mrr_key: row[f"baseline_{mrr_key}"],
                hit_key: row[f"baseline_{hit_key}"],
                "top1_hit": row["baseline_top1_hit"],
            }
        )
        after_rows.append(
            {
                field: group,
                recall_key: row[f"adapted_{recall_key}"],
                ndcg_key: row[f"adapted_{ndcg_key}"],
                mrr_key: row[f"adapted_{mrr_key}"],
                hit_key: row[f"adapted_{hit_key}"],
                "top1_hit": row["adapted_top1_hit"],
            }
        )
        delta_rows.append(
            {
                field: group,
                f"delta_{recall_key}": row[f"delta_{recall_key}"],
                f"delta_{ndcg_key}": row[f"delta_{ndcg_key}"],
                f"delta_{mrr_key}": row[f"delta_{mrr_key}"],
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
    parser.add_argument("--recall-k", type=int, default=5)
    parser.add_argument("--ndcg-k", type=int, default=10)
    parser.add_argument("--mrr-k", type=int, default=10)
    parser.add_argument("--dense-k", type=int, default=10)
    parser.add_argument("--sparse-k", type=int, default=10)
    parser.add_argument("--filter-field", default="")
    parser.add_argument("--filter-value", default="")
    parser.add_argument("--model-name", default="BAAI/bge-m3")
    parser.add_argument(
        "--query-instruction",
        default=DEFAULT_QUERY_TOWER_INSTRUCTION,
    )
    args = parser.parse_args()

    from ingestion import HybridRetriever, get_local_documents, open_vectorstore

    dataset = load_jsonl(args.dataset)
    if args.filter_field:
        dataset = [
            sample
            for sample in dataset
            if str(sample.get(args.filter_field, "")) == args.filter_value
        ]
    if not dataset:
        raise RuntimeError("No retrieval evaluation samples remain after filtering.")
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
    recall_key = f"recall@{args.recall_k}"
    ndcg_key = f"ndcg@{args.ndcg_k}"
    mrr_key = f"mrr@{args.mrr_k}"
    hit_key = f"hit@{max(args.recall_k, args.ndcg_k, args.mrr_k)}"
    relevant_hits_key = f"relevant_hits@{max(args.recall_k, args.ndcg_k, args.mrr_k)}"

    for sample in dataset:
        baseline = _evaluate_retriever(
            baseline_retriever,
            sample,
            recall_k=args.recall_k,
            ndcg_k=args.ndcg_k,
            mrr_k=args.mrr_k,
        )
        adapted = _evaluate_retriever(
            adapted_retriever,
            sample,
            recall_k=args.recall_k,
            ndcg_k=args.ndcg_k,
            mrr_k=args.mrr_k,
        )

        baseline_metrics = {
            recall_key: float(baseline[recall_key]),
            ndcg_key: float(baseline[ndcg_key]),
            mrr_key: float(baseline[mrr_key]),
            hit_key: float(baseline[hit_key]),
            "top1_hit": float(baseline["top1_hit"]),
        }
        adapted_metrics = {
            recall_key: float(adapted[recall_key]),
            ndcg_key: float(adapted[ndcg_key]),
            mrr_key: float(adapted[mrr_key]),
            hit_key: float(adapted[hit_key]),
            "top1_hit": float(adapted["top1_hit"]),
        }
        baseline_metric_rows.append(baseline_metrics)
        adapted_metric_rows.append(adapted_metrics)

        delta_recall = _diff_metric(adapted[recall_key], baseline[recall_key])
        delta_ndcg = _diff_metric(adapted[ndcg_key], baseline[ndcg_key])
        delta_mrr = _diff_metric(adapted[mrr_key], baseline[mrr_key])

        rows.append(
            {
                "query": sample["query"],
                "query_type": sample.get("query_type", "unknown"),
                "medical_entities": sample.get("medical_entities", []),
                "expected_route_strategy": sample.get("expected_route_strategy", ""),
                **_prefix_metrics("baseline", baseline),
                **_prefix_metrics("adapted", adapted),
                f"delta_{recall_key}": delta_recall,
                f"delta_{ndcg_key}": delta_ndcg,
                f"delta_{mrr_key}": delta_mrr,
                "recovered_from_miss": int(
                    baseline[hit_key] == 0 and adapted[hit_key] == 1
                ),
                "improved_top1": int(
                    baseline["top1_hit"] == 0 and adapted["top1_hit"] == 1
                ),
                "regressed_hit@k": int(
                    baseline[hit_key] == 1 and adapted[hit_key] == 0
                ),
                "relevant_hit_gain": int(adapted[relevant_hits_key])
                - int(baseline[relevant_hits_key]),
            }
        )

    baseline_summary = summarize_metrics(baseline_metric_rows)
    adapted_summary = summarize_metrics(adapted_metric_rows)
    comparison = {
        "dataset": str(args.dataset),
        "adapter_path": str(args.adapter_path),
        "recall_k": args.recall_k,
        "ndcg_k": args.ndcg_k,
        "mrr_k": args.mrr_k,
        "dense_k": args.dense_k,
        "sparse_k": args.sparse_k,
        "filter_field": args.filter_field or None,
        "filter_value": args.filter_value or None,
        "num_samples": len(dataset),
        "baseline_summary": baseline_summary,
        "adapted_summary": adapted_summary,
        "delta_summary": _delta_summary(baseline_summary, adapted_summary),
        "focus_metrics": {
            recall_key: {
                "baseline": baseline_summary.get(recall_key, 0.0),
                "adapted": adapted_summary.get(recall_key, 0.0),
                "delta": adapted_summary.get(recall_key, 0.0)
                - baseline_summary.get(recall_key, 0.0),
            },
            ndcg_key: {
                "baseline": baseline_summary.get(ndcg_key, 0.0),
                "adapted": adapted_summary.get(ndcg_key, 0.0),
                "delta": adapted_summary.get(ndcg_key, 0.0)
                - baseline_summary.get(ndcg_key, 0.0),
            },
            mrr_key: {
                "baseline": baseline_summary.get(mrr_key, 0.0),
                "adapted": adapted_summary.get(mrr_key, 0.0),
                "delta": adapted_summary.get(mrr_key, 0.0)
                - baseline_summary.get(mrr_key, 0.0),
            },
        },
        "gain_breakdown": {
            "improved_samples": sum(
                1
                for row in rows
                if (row[f"delta_{ndcg_key}"] or 0) > 0
                or (row[f"delta_{mrr_key}"] or 0) > 0
            ),
            "recovered_from_miss": sum(row["recovered_from_miss"] for row in rows),
            "improved_top1": sum(row["improved_top1"] for row in rows),
            "regressed_samples": sum(row["regressed_hit@k"] for row in rows),
        },
        "by_query_type": _group_comparison(
            rows,
            "query_type",
            recall_key=recall_key,
            ndcg_key=ndcg_key,
            mrr_key=mrr_key,
            hit_key=hit_key,
        ),
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

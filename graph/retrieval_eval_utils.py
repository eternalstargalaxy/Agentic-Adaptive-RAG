from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from graph.retrieval_metrics import mrr_at_k, ndcg_at_k, recall_at_k


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def document_identifier(document: Any) -> str:
    metadata = getattr(document, "metadata", {}) or {}
    return (
        metadata.get("url")
        or metadata.get("doc_id")
        or metadata.get("source")
        or getattr(document, "page_content", "")[:120]
    )


def build_relevance_flags(documents: List[Any], sample: Dict[str, Any]) -> List[int]:
    relevant_ids = set(sample.get("relevant_ids", []))
    relevant_sources = set(sample.get("relevant_sources", []))
    relevant_keywords = [keyword.lower() for keyword in sample.get("relevant_keywords", [])]

    flags = []
    for document in documents:
        identifier = document_identifier(document)
        source = (getattr(document, "metadata", {}) or {}).get("source", "")
        content = getattr(document, "page_content", "").lower()

        is_relevant = (
            identifier in relevant_ids
            or source in relevant_sources
            or any(keyword in content for keyword in relevant_keywords)
        )
        flags.append(1 if is_relevant else 0)
    return flags


def first_relevant_rank(relevance_flags: Iterable[int], k: int | None = None) -> int | None:
    flags = list(relevance_flags)
    if k is not None:
        flags = flags[:k]

    for rank, flag in enumerate(flags, start=1):
        if flag:
            return rank
    return None


def score_retrieval_flags(
    relevance_flags: Iterable[int],
    recall_k: int = 5,
    ndcg_k: int = 10,
    mrr_k: int = 10,
) -> Dict[str, float | int | None]:
    flags = list(relevance_flags)
    max_k = max(recall_k, ndcg_k, mrr_k)
    rank = first_relevant_rank(flags, max_k)
    return {
        f"recall@{recall_k}": recall_at_k(flags, recall_k),
        f"ndcg@{ndcg_k}": ndcg_at_k(flags, ndcg_k),
        f"mrr@{mrr_k}": mrr_at_k(flags, mrr_k),
        f"hit@{max_k}": 1 if rank is not None else 0,
        "top1_hit": 1 if rank == 1 else 0,
        f"relevant_hits@{max_k}": sum(flags[:max_k]),
        "first_relevant_rank": rank,
    }


def evaluate_documents(documents: List[Any], sample: Dict[str, Any], k: int) -> Dict[str, float | int | None]:
    flags = build_relevance_flags(documents, sample)
    return score_retrieval_flags(flags, recall_k=k, ndcg_k=k, mrr_k=k)


def evaluate_documents_with_cutoffs(
    documents: List[Any],
    sample: Dict[str, Any],
    recall_k: int = 5,
    ndcg_k: int = 10,
    mrr_k: int = 10,
) -> Dict[str, float | int | None]:
    flags = build_relevance_flags(documents, sample)
    return score_retrieval_flags(
        flags,
        recall_k=recall_k,
        ndcg_k=ndcg_k,
        mrr_k=mrr_k,
    )


def evaluate_sample(retriever: Any, sample: Dict[str, Any], k: int) -> Dict[str, float]:
    documents = retriever.invoke(sample["query"])
    return evaluate_documents(documents, sample, k)


def evaluate_sample_with_cutoffs(
    retriever: Any,
    sample: Dict[str, Any],
    recall_k: int = 5,
    ndcg_k: int = 10,
    mrr_k: int = 10,
) -> Dict[str, float | int | None]:
    documents = retriever.invoke(sample["query"])
    return evaluate_documents_with_cutoffs(
        documents,
        sample,
        recall_k=recall_k,
        ndcg_k=ndcg_k,
        mrr_k=mrr_k,
    )


def summarize_by_field(rows: Iterable[Dict[str, Any]], field: str) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for row in rows:
        group_name = row.get(field, "unknown")
        grouped.setdefault(group_name, []).append(
            {
                key: value
                for key, value in row.items()
                if isinstance(value, (int, float))
            }
        )

    summary: Dict[str, Dict[str, float]] = {}
    for group_name, group_rows in grouped.items():
        if not group_rows:
            continue
        keys = group_rows[0].keys()
        summary[group_name] = {
            key: sum(item[key] for item in group_rows) / len(group_rows)
            for key in keys
        }
    return summary

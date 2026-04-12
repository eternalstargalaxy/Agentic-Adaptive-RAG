from __future__ import annotations

import math
from typing import Dict, Iterable, List


def recall_at_k(relevance_flags: Iterable[int], k: int) -> float:
    all_flags = list(relevance_flags)
    flags = all_flags[:k]
    total_relevant = sum(all_flags)
    if total_relevant == 0:
        return 0.0
    return sum(flags) / total_relevant


def mrr_at_k(relevance_flags: Iterable[int], k: int) -> float:
    for rank, flag in enumerate(list(relevance_flags)[:k], start=1):
        if flag:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevance_flags: Iterable[int], k: int) -> float:
    flags = list(relevance_flags)[:k]
    dcg = sum(flag / math.log2(rank + 1) for rank, flag in enumerate(flags, start=1))
    ideal = sorted(flags, reverse=True)
    idcg = sum(flag / math.log2(rank + 1) for rank, flag in enumerate(ideal, start=1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def summarize_metrics(metric_rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_rows:
        return {}

    keys = metric_rows[0].keys()
    summary = {}
    for key in keys:
        summary[key] = sum(row[key] for row in metric_rows) / len(metric_rows)
    return summary

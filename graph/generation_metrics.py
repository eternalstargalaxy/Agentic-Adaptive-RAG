from __future__ import annotations

from typing import Dict, List


def compute_rouge_l(prediction: str, reference: str) -> float | None:
    try:
        from rouge_score import rouge_scorer
    except Exception:
        return None

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    result = scorer.score(reference, prediction)
    return float(result["rougeL"].fmeasure)


def compute_bert_score(predictions: List[str], references: List[str]) -> float | None:
    try:
        from bert_score import score
    except Exception:
        return None

    _, _, f1 = score(predictions, references, lang="en", verbose=False)
    return float(f1.mean().item())


def average_metric(values: List[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def summarize_generation_metrics(rows: List[Dict[str, float | None]]) -> Dict[str, float | None]:
    if not rows:
        return {}

    return {
        "rougeL": average_metric([row.get("rougeL") for row in rows]),
        "bert_score": average_metric([row.get("bert_score") for row in rows]),
    }

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph.retrieval_eval_utils import load_jsonl, summarize_by_field


def _transition_exists(history: List[str], source: str, target: str) -> bool:
    return any(
        previous == source and current == target
        for previous, current in zip(history, history[1:])
    )


def _safe_history(state: Dict[str, Any]) -> List[str]:
    history = list(state.get("route_history", []))
    if history:
        return history

    route_strategy = state.get("route_strategy")
    if route_strategy:
        return [route_strategy]
    return []


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure route upgrade rates from No Retrieval and Single-Step."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/route_upgrade_stats.json"),
    )
    parser.add_argument("--question-field", default="query")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from graph.consts import MULTI_HOP, NO_RETRIEVAL, SINGLE_STEP
    from graph.graph import app

    dataset = load_jsonl(args.dataset)
    if args.limit > 0:
        dataset = dataset[: args.limit]

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for index, sample in enumerate(dataset, start=1):
        question = sample.get(args.question_field) or sample.get("question")
        if not question:
            errors.append(
                {
                    "index": str(index),
                    "error": f"Missing '{args.question_field}' field.",
                }
            )
            continue

        try:
            state = app.invoke({"question": question})
        except Exception as exc:
            errors.append(
                {
                    "index": str(index),
                    "query": question,
                    "error": str(exc),
                }
            )
            continue

        history = _safe_history(state)
        if not history:
            errors.append(
                {
                    "index": str(index),
                    "query": question,
                    "error": "No route history returned by graph execution.",
                }
            )
            continue

        row = {
            "query": question,
            "query_type": sample.get("query_type", "unknown"),
            "route_history": history,
            "route_path": " -> ".join(history),
            "initial_route": history[0],
            "final_route": history[-1],
            "route_steps": len(history),
            "entered_no_retrieval": int(NO_RETRIEVAL in history),
            "entered_single_step": int(SINGLE_STEP in history),
            "entered_multi_hop": int(MULTI_HOP in history),
            "upgrade_no_to_single": int(
                _transition_exists(history, NO_RETRIEVAL, SINGLE_STEP)
            ),
            "upgrade_single_to_multi": int(
                _transition_exists(history, SINGLE_STEP, MULTI_HOP)
            ),
        }
        rows.append(row)

    if not rows:
        first_error = errors[0]["error"] if errors else "Unknown error."
        raise RuntimeError(
            "Route upgrade stats could not be computed. "
            f"First error: {first_error}"
        )

    initial_distribution = dict(Counter(row["initial_route"] for row in rows))
    final_distribution = dict(Counter(row["final_route"] for row in rows))
    path_distribution = dict(Counter(row["route_path"] for row in rows))

    initial_no_retrieval = sum(1 for row in rows if row["initial_route"] == NO_RETRIEVAL)
    initial_single_step = sum(1 for row in rows if row["initial_route"] == SINGLE_STEP)
    single_step_entries = sum(row["entered_single_step"] for row in rows)

    summary = {
        "dataset": str(args.dataset),
        "num_samples": len(rows),
        "num_errors": len(errors),
        "initial_route_distribution": initial_distribution,
        "final_route_distribution": final_distribution,
        "route_path_distribution": path_distribution,
        "upgrade_rates": {
            "no_retrieval_to_single_step": {
                "numerator": sum(row["upgrade_no_to_single"] for row in rows),
                "denominator": initial_no_retrieval,
                "rate": _rate(
                    sum(row["upgrade_no_to_single"] for row in rows),
                    initial_no_retrieval,
                ),
            },
            "single_step_to_multi_hop_from_initial_single_step": {
                "numerator": sum(
                    row["upgrade_single_to_multi"]
                    for row in rows
                    if row["initial_route"] == SINGLE_STEP
                ),
                "denominator": initial_single_step,
                "rate": _rate(
                    sum(
                        row["upgrade_single_to_multi"]
                        for row in rows
                        if row["initial_route"] == SINGLE_STEP
                    ),
                    initial_single_step,
                ),
            },
            "single_step_to_multi_hop_from_all_single_step_entries": {
                "numerator": sum(row["upgrade_single_to_multi"] for row in rows),
                "denominator": single_step_entries,
                "rate": _rate(
                    sum(row["upgrade_single_to_multi"] for row in rows),
                    single_step_entries,
                ),
            },
        },
        "by_query_type": summarize_by_field(
            [
                {
                    "query_type": row["query_type"],
                    "route_steps": row["route_steps"],
                    "entered_no_retrieval": row["entered_no_retrieval"],
                    "entered_single_step": row["entered_single_step"],
                    "entered_multi_hop": row["entered_multi_hop"],
                    "upgrade_no_to_single": row["upgrade_no_to_single"],
                    "upgrade_single_to_multi": row["upgrade_single_to_multi"],
                }
                for row in rows
            ],
            "query_type",
        ),
        "per_sample": rows,
        "errors": errors,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary["upgrade_rates"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

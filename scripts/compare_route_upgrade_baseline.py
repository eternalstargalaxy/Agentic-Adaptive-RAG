from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run_upgrade_stats(
    dataset: Path,
    output: Path,
    question_field: str,
    limit: int,
    adapter_path: Path | None,
) -> Dict[str, Any]:
    env = os.environ.copy()
    if adapter_path:
        env["QUERY_TOWER_ADAPTER_PATH"] = str(adapter_path)
    else:
        env.pop("QUERY_TOWER_ADAPTER_PATH", None)

    command: List[str] = [
        sys.executable,
        str(ROOT / "scripts" / "compute_route_upgrade_stats.py"),
        "--dataset",
        str(dataset),
        "--output",
        str(output),
        "--question-field",
        question_field,
    ]
    if limit > 0:
        command.extend(["--limit", str(limit)])

    subprocess.run(
        command,
        check=True,
        cwd=str(ROOT),
        env=env,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def _extract_rate(payload: Dict[str, Any], path: List[str]) -> float:
    current: Any = payload
    for key in path:
        current = current[key]
    return float(current)


def _delta(after: float, before: float) -> float:
    return after - before


def _align_samples(
    baseline_rows: List[Dict[str, Any]],
    adapted_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    aligned: List[Dict[str, Any]] = []
    max_len = max(len(baseline_rows), len(adapted_rows))
    for index in range(max_len):
        baseline_row = baseline_rows[index] if index < len(baseline_rows) else {}
        adapted_row = adapted_rows[index] if index < len(adapted_rows) else {}
        aligned.append(
            {
                "query": baseline_row.get("query", adapted_row.get("query", "")),
                "query_type": baseline_row.get(
                    "query_type",
                    adapted_row.get("query_type", "unknown"),
                ),
                "baseline_initial_route": baseline_row.get("initial_route"),
                "adapted_initial_route": adapted_row.get("initial_route"),
                "baseline_final_route": baseline_row.get("final_route"),
                "adapted_final_route": adapted_row.get("final_route"),
                "baseline_route_history": baseline_row.get("route_history", []),
                "adapted_route_history": adapted_row.get("route_history", []),
                "baseline_upgrade_no_to_single": baseline_row.get(
                    "upgrade_no_to_single",
                    0,
                ),
                "adapted_upgrade_no_to_single": adapted_row.get(
                    "upgrade_no_to_single",
                    0,
                ),
                "baseline_upgrade_single_to_multi": baseline_row.get(
                    "upgrade_single_to_multi",
                    0,
                ),
                "adapted_upgrade_single_to_multi": adapted_row.get(
                    "upgrade_single_to_multi",
                    0,
                ),
                "reduced_no_to_single_upgrade": int(
                    baseline_row.get("upgrade_no_to_single", 0) == 1
                    and adapted_row.get("upgrade_no_to_single", 0) == 0
                ),
                "reduced_single_to_multi_upgrade": int(
                    baseline_row.get("upgrade_single_to_multi", 0) == 1
                    and adapted_row.get("upgrade_single_to_multi", 0) == 0
                ),
            }
        )
    return aligned


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare route upgrade rates before and after query tower LoRA."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/route_upgrade_baseline_comparison.json"),
    )
    parser.add_argument("--question-field", default="query")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        baseline_stats = _run_upgrade_stats(
            dataset=args.dataset,
            output=temp_dir_path / "baseline_route_upgrade.json",
            question_field=args.question_field,
            limit=args.limit,
            adapter_path=None,
        )
        adapted_stats = _run_upgrade_stats(
            dataset=args.dataset,
            output=temp_dir_path / "adapted_route_upgrade.json",
            question_field=args.question_field,
            limit=args.limit,
            adapter_path=args.adapter_path,
        )

    baseline_no_to_single = _extract_rate(
        baseline_stats,
        ["upgrade_rates", "no_retrieval_to_single_step", "rate"],
    )
    adapted_no_to_single = _extract_rate(
        adapted_stats,
        ["upgrade_rates", "no_retrieval_to_single_step", "rate"],
    )
    baseline_single_to_multi = _extract_rate(
        baseline_stats,
        ["upgrade_rates", "single_step_to_multi_hop_from_initial_single_step", "rate"],
    )
    adapted_single_to_multi = _extract_rate(
        adapted_stats,
        ["upgrade_rates", "single_step_to_multi_hop_from_initial_single_step", "rate"],
    )

    comparison = {
        "dataset": str(args.dataset),
        "adapter_path": str(args.adapter_path),
        "question_field": args.question_field,
        "limit": args.limit if args.limit > 0 else None,
        "baseline_upgrade_rates": baseline_stats["upgrade_rates"],
        "adapted_upgrade_rates": adapted_stats["upgrade_rates"],
        "delta_upgrade_rates": {
            "no_retrieval_to_single_step": _delta(
                adapted_no_to_single,
                baseline_no_to_single,
            ),
            "single_step_to_multi_hop_from_initial_single_step": _delta(
                adapted_single_to_multi,
                baseline_single_to_multi,
            ),
        },
        "focus_metrics": {
            "no_retrieval_to_single_step": {
                "baseline": baseline_no_to_single,
                "adapted": adapted_no_to_single,
                "delta": _delta(adapted_no_to_single, baseline_no_to_single),
            },
            "single_step_to_multi_hop": {
                "baseline": baseline_single_to_multi,
                "adapted": adapted_single_to_multi,
                "delta": _delta(adapted_single_to_multi, baseline_single_to_multi),
            },
        },
        "baseline_initial_route_distribution": baseline_stats[
            "initial_route_distribution"
        ],
        "adapted_initial_route_distribution": adapted_stats[
            "initial_route_distribution"
        ],
        "baseline_final_route_distribution": baseline_stats[
            "final_route_distribution"
        ],
        "adapted_final_route_distribution": adapted_stats[
            "final_route_distribution"
        ],
        "per_sample": _align_samples(
            baseline_stats.get("per_sample", []),
            adapted_stats.get("per_sample", []),
        ),
        "baseline_errors": baseline_stats.get("errors", []),
        "adapted_errors": adapted_stats.get("errors", []),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(comparison["delta_upgrade_rates"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

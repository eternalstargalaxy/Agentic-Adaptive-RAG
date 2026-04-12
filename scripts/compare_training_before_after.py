from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run_script(command: List[str]) -> None:
    subprocess.run(
        command,
        check=True,
        cwd=str(ROOT),
    )


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce a unified before/after baseline report for query tower LoRA."
    )
    parser.add_argument("--retrieval-dataset", type=Path, required=True)
    parser.add_argument("--routing-dataset", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/training_before_after_comparison.json"),
    )
    parser.add_argument("--recall-k", type=int, default=5)
    parser.add_argument("--ndcg-k", type=int, default=10)
    parser.add_argument("--mrr-k", type=int, default=10)
    parser.add_argument("--dense-k", type=int, default=10)
    parser.add_argument("--sparse-k", type=int, default=10)
    parser.add_argument(
        "--single-step-filter-field",
        default="expected_route_strategy",
    )
    parser.add_argument(
        "--single-step-filter-value",
        default="single_step",
    )
    parser.add_argument("--question-field", default="query")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        retrieval_output = temp_dir_path / "retrieval_comparison.json"
        route_output = temp_dir_path / "route_comparison.json"

        retrieval_command = [
            sys.executable,
            str(ROOT / "scripts" / "compare_query_tower_baseline.py"),
            "--dataset",
            str(args.retrieval_dataset),
            "--adapter-path",
            str(args.adapter_path),
            "--output",
            str(retrieval_output),
            "--recall-k",
            str(args.recall_k),
            "--ndcg-k",
            str(args.ndcg_k),
            "--mrr-k",
            str(args.mrr_k),
            "--dense-k",
            str(args.dense_k),
            "--sparse-k",
            str(args.sparse_k),
        ]
        if args.single_step_filter_field:
            retrieval_command.extend(
                [
                    "--filter-field",
                    args.single_step_filter_field,
                    "--filter-value",
                    args.single_step_filter_value,
                ]
            )

        route_command = [
            sys.executable,
            str(ROOT / "scripts" / "compare_route_upgrade_baseline.py"),
            "--dataset",
            str(args.routing_dataset),
            "--adapter-path",
            str(args.adapter_path),
            "--output",
            str(route_output),
            "--question-field",
            args.question_field,
        ]
        if args.limit > 0:
            route_command.extend(["--limit", str(args.limit)])

        _run_script(retrieval_command)
        _run_script(route_command)

        retrieval_report = _load_json(retrieval_output)
        route_report = _load_json(route_output)

    comparison = {
        "retrieval_dataset": str(args.retrieval_dataset),
        "routing_dataset": str(args.routing_dataset),
        "adapter_path": str(args.adapter_path),
        "focus_summary": {
            "single_step_recall": retrieval_report["focus_metrics"].get(
                f"recall@{args.recall_k}",
                {},
            ),
            "single_step_ndcg": retrieval_report["focus_metrics"].get(
                f"ndcg@{args.ndcg_k}",
                {},
            ),
            "single_step_mrr": retrieval_report["focus_metrics"].get(
                f"mrr@{args.mrr_k}",
                {},
            ),
            "no_retrieval_to_single_step_upgrade_rate": route_report[
                "focus_metrics"
            ].get("no_retrieval_to_single_step", {}),
            "single_step_to_multi_hop_upgrade_rate": route_report[
                "focus_metrics"
            ].get("single_step_to_multi_hop", {}),
        },
        "single_step_metrics": {
            "baseline": retrieval_report["baseline_summary"],
            "adapted": retrieval_report["adapted_summary"],
            "delta": retrieval_report["delta_summary"],
            "gain_breakdown": retrieval_report["gain_breakdown"],
            "filter_field": args.single_step_filter_field or None,
            "filter_value": args.single_step_filter_value or None,
        },
        "route_upgrade_rates": {
            "baseline": route_report["baseline_upgrade_rates"],
            "adapted": route_report["adapted_upgrade_rates"],
            "delta": route_report["delta_upgrade_rates"],
        },
        "artifacts": {
            "retrieval_report": retrieval_report,
            "route_report": route_report,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "focus_summary": comparison["focus_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

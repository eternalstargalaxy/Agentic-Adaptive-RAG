from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from graph.generation_metrics import compute_bert_score, compute_rouge_l, summarize_generation_metrics


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate synthetic generation outputs.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/synthetic_generation_eval.json"),
    )
    args = parser.parse_args()

    rows = load_jsonl(args.dataset)
    scored_rows = []
    predictions = [row["prediction"] for row in rows]
    references = [row["reference_answer"] for row in rows]
    bert_score_value = compute_bert_score(predictions, references)

    for row in rows:
        rouge_l = compute_rouge_l(row["prediction"], row["reference_answer"])
        scored_rows.append(
            {
                "question": row["question"],
                "rougeL": rouge_l,
                "bert_score": bert_score_value,
            }
        )

    summary = summarize_generation_metrics(scored_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "rows": scored_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

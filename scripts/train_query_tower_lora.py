from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


@dataclass
class TrainingConfig:
    model_name: str
    output_dir: Path
    batch_size: int
    learning_rate: float
    epochs: int
    max_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float


class QueryTowerLoRATrainer:
    """
    Query-tower-only LoRA training for BGE-M3.

    The document tower stays frozen. The query tower learns to better align
    short, ambiguous, and terminology-shifted medical questions with the
    document embedding space.
    """

    def __init__(self, config: TrainingConfig) -> None:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModel, AutoTokenizer

        self.config = config
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )

        self.doc_model = AutoModel.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        ).to(self.device)
        self.doc_model.eval()
        for param in self.doc_model.parameters():
            param.requires_grad = False

        query_model = AutoModel.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["query", "key", "value"],
        )
        self.query_model = get_peft_model(query_model, lora_config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.query_model.parameters(),
            lr=config.learning_rate,
        )

    def _mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _encode(self, model, texts: List[str], normalize: bool = True):
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        outputs = model(**tokens)
        embeddings = self._mean_pool(outputs.last_hidden_state, tokens["attention_mask"])
        if normalize:
            embeddings = self.torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def _build_batch(self, batch: List[Dict[str, Any]]):
        queries = [row["query"] for row in batch]
        positives = [row["positive_passage"] for row in batch]
        negatives = [row.get("hard_negative_passages", []) for row in batch]
        return queries, positives, negatives

    def train(self, rows: List[Dict[str, Any]]) -> None:
        self.query_model.train()
        batch_size = self.config.batch_size
        loss_fn = self.torch.nn.CrossEntropyLoss()

        for epoch in range(self.config.epochs):
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                queries, positives, negatives = self._build_batch(batch)
                query_embeddings = self._encode(self.query_model, queries)

                flat_candidates = []
                labels = []
                for row_index, positive in enumerate(positives):
                    candidate_group = [positive] + negatives[row_index]
                    labels.append(0)
                    flat_candidates.extend(candidate_group)

                with self.torch.no_grad():
                    candidate_embeddings = self._encode(self.doc_model, flat_candidates)

                max_candidates = max(1 + len(item) for item in negatives)
                hidden_size = candidate_embeddings.size(-1)
                candidate_tensor = self.torch.zeros(
                    len(batch),
                    max_candidates,
                    hidden_size,
                    device=self.device,
                )

                pointer = 0
                for batch_index, row in enumerate(batch):
                    group_size = 1 + len(row.get("hard_negative_passages", []))
                    candidate_tensor[batch_index, :group_size, :] = candidate_embeddings[
                        pointer : pointer + group_size
                    ]
                    pointer += group_size

                scores = self.torch.einsum("bd,bcd->bc", query_embeddings, candidate_tensor)
                target = self.torch.tensor(labels, device=self.device)
                loss = loss_fn(scores, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"epoch={epoch + 1} loss={loss.item():.4f}")

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.query_model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BGE-M3 query tower with LoRA.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    args = parser.parse_args()

    rows = load_jsonl(args.dataset)
    trainer = QueryTowerLoRATrainer(
        TrainingConfig(
            model_name=args.model_name,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            max_length=args.max_length,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    )
    trainer.train(rows)


if __name__ == "__main__":
    main()

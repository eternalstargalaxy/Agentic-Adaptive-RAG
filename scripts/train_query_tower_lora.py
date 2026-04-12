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
    temperature: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float


class QueryTowerLoRATrainer:
    """
    Query-tower-only LoRA training for BGE-M3.

    The document tower stays frozen. The query tower learns to better align
    short, ambiguous, and terminology-shifted medical questions with the
    document embedding space.

    Training objective:
    - input: (query, correct document, confusable negative document)
    - output: three embeddings
    - loss: InfoNCE contrastive learning
    - goal: make the query closer to the correct medical evidence than to
      easily confused negative evidence
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

    def _extract_triplets(self, batch: List[Dict[str, Any]]):
        queries = [row["query"] for row in batch]
        positives = [
            row.get("positive_document") or row.get("positive_passage")
            for row in batch
        ]
        negatives = [
            (
                row.get("confusable_negative_document")
                or row.get("hard_negative_passages", [None])[0]
            )
            for row in batch
        ]
        return queries, positives, negatives

    def _compute_info_nce_loss(self, query_embeddings, positive_embeddings, negative_embeddings):
        positive_scores = (query_embeddings * positive_embeddings).sum(dim=1, keepdim=True)
        negative_scores = (query_embeddings * negative_embeddings).sum(dim=1, keepdim=True)
        logits = self.torch.cat([positive_scores, negative_scores], dim=1)
        logits = logits / self.config.temperature
        labels = self.torch.zeros(query_embeddings.size(0), dtype=self.torch.long, device=self.device)
        return self.torch.nn.functional.cross_entropy(logits, labels), logits

    def train(self, rows: List[Dict[str, Any]]) -> None:
        self.query_model.train()
        batch_size = self.config.batch_size

        for epoch in range(self.config.epochs):
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                queries, positives, negatives = self._extract_triplets(batch)
                query_embeddings = self._encode(self.query_model, queries)

                with self.torch.no_grad():
                    positive_embeddings = self._encode(self.doc_model, positives)
                    negative_embeddings = self._encode(self.doc_model, negatives)

                loss, logits = self._compute_info_nce_loss(
                    query_embeddings,
                    positive_embeddings,
                    negative_embeddings,
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            margin = (logits[:, 0] - logits[:, 1]).mean().item()
            print(
                f"epoch={epoch + 1} loss={loss.item():.4f} "
                f"avg_pos_neg_margin={margin:.4f}"
            )

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
    parser.add_argument("--temperature", type=float, default=0.05)
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
            temperature=args.temperature,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    )
    trainer.train(rows)


if __name__ == "__main__":
    main()

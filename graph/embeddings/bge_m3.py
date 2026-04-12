from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List


@dataclass
class EncoderRuntime:
    tokenizer: object
    model: object
    device: str
    torch: object


class BGEM3BiEncoderEmbeddings:
    """
    Dense embedding wrapper for ChromaDB.

    Document embeddings are produced by the base BGE-M3 encoder, while query
    embeddings can optionally load a LoRA adapter trained only for the query
    tower. This keeps the document side stable and makes query-side adaptation
    easy to evaluate.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        query_adapter_path: str | None = None,
        query_instruction: str | None = None,
        batch_size: int = 16,
        max_length: int = 512,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.query_adapter_path = query_adapter_path
        self.query_instruction = (
            query_instruction
            or "Represent this medical query for retrieving supporting evidence: "
        )
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        self._document_runtime: EncoderRuntime | None = None
        self._query_runtime: EncoderRuntime | None = None

    def _resolve_device(self) -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _load_runtime(self, adapter_path: str | None = None) -> EncoderRuntime:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)

        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)

        device = self._resolve_device()
        model = model.to(device)
        model.eval()
        return EncoderRuntime(tokenizer=tokenizer, model=model, device=device, torch=torch)

    def _get_document_runtime(self) -> EncoderRuntime:
        if self._document_runtime is None:
            self._document_runtime = self._load_runtime()
        return self._document_runtime

    def _get_query_runtime(self) -> EncoderRuntime:
        if self._query_runtime is None:
            self._query_runtime = self._load_runtime(self.query_adapter_path)
        return self._query_runtime

    def _mean_pool(self, last_hidden_state, attention_mask, torch_module):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _prepare_texts(self, texts: List[str], is_query: bool) -> List[str]:
        prepared = []
        for text in texts:
            text = (text or "").strip()
            if is_query:
                prepared.append(f"{self.query_instruction}{text}")
            else:
                prepared.append(text)
        return prepared

    def _encode(self, texts: List[str], is_query: bool) -> List[List[float]]:
        if not texts:
            return []

        runtime = self._get_query_runtime() if is_query else self._get_document_runtime()
        prepared_texts = self._prepare_texts(texts, is_query=is_query)
        outputs: List[List[float]] = []

        for start in range(0, len(prepared_texts), self.batch_size):
            batch_texts = prepared_texts[start : start + self.batch_size]
            tokens = runtime.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(runtime.device) for key, value in tokens.items()}

            with runtime.torch.no_grad():
                model_output = runtime.model(**tokens)
                sentence_embeddings = self._mean_pool(
                    model_output.last_hidden_state,
                    tokens["attention_mask"],
                    runtime.torch,
                )
                if self.normalize_embeddings:
                    sentence_embeddings = runtime.torch.nn.functional.normalize(
                        sentence_embeddings,
                        p=2,
                        dim=1,
                    )
            outputs.extend(sentence_embeddings.cpu().tolist())

        return outputs

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts, is_query=False)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text], is_query=True)[0]

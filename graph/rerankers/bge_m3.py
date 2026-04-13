from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Sequence

from langchain.schema import Document

from model import embed_model


@dataclass(frozen=True)
class RerankResult:
    score: float
    rank: int
    document: Document


class BGEM3Reranker:
    """
    Lightweight reranker based on BGE-M3 bi-encoder similarity.

    This is not a cross-encoder. It reuses the query tower + document tower
    embeddings to reorder the already retrieved candidate set at low latency.
    """

    def __init__(self, embedding_model=None) -> None:
        self.embedding_model = embedding_model or embed_model

    @staticmethod
    def _dot(query_vector: Sequence[float], document_vector: Sequence[float]) -> float:
        return float(sum(left * right for left, right in zip(query_vector, document_vector)))

    def rerank(
        self,
        query: str,
        documents: Iterable[Document],
        top_k: int | None = None,
    ) -> List[Document]:
        documents = list(documents)
        if not documents:
            return []

        query_vector = self.embedding_model.embed_query(query)
        document_vectors = self.embedding_model.embed_documents(
            [document.page_content for document in documents]
        )

        results: List[RerankResult] = []
        for rank, (document, document_vector) in enumerate(
            zip(documents, document_vectors),
            start=1,
        ):
            score = self._dot(query_vector, document_vector)
            metadata = {
                **(document.metadata or {}),
                "rerank_score": score,
                "pre_rerank_rank": rank,
            }
            results.append(
                RerankResult(
                    score=score,
                    rank=rank,
                    document=Document(
                        page_content=document.page_content,
                        metadata=metadata,
                    ),
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        reranked_documents: List[Document] = []
        for final_rank, result in enumerate(results, start=1):
            metadata = {
                **(result.document.metadata or {}),
                "rerank_rank": final_rank,
            }
            reranked_documents.append(
                Document(
                    page_content=result.document.page_content,
                    metadata=metadata,
                )
            )

        if top_k is not None:
            return reranked_documents[:top_k]
        return reranked_documents


@lru_cache(maxsize=1)
def get_bge_m3_reranker() -> BGEM3Reranker:
    return BGEM3Reranker()

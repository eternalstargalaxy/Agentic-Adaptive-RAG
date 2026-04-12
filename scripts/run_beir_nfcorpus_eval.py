from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List

from langchain.schema import Document
from langchain_chroma import Chroma

from graph.retrieval_metrics import mrr_at_k, ndcg_at_k, recall_at_k, summarize_metrics
from ingestion import HybridRetriever
from model import embed_model


def build_documents(corpus: Dict[str, Dict[str, str]]) -> List[Document]:
    documents = []
    for doc_id, payload in corpus.items():
        text = f"{payload.get('title', '')}\n\n{payload.get('text', '')}".strip()
        documents.append(
            Document(
                page_content=text,
                metadata={"source": "nfcorpus", "doc_id": doc_id},
            )
        )
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BEIR nfcorpus retrieval evaluation.")
    parser.add_argument("--output", type=Path, default=Path("reports/nfcorpus_eval.json"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
    except Exception as exc:
        raise RuntimeError("beir is required to run nfcorpus evaluation.") from exc

    dataset_url = (
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
    )
    with TemporaryDirectory() as temp_dir:
        data_path = util.download_and_unzip(dataset_url, temp_dir)
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
        documents = build_documents(corpus)

        vectorstore = Chroma.from_documents(
            documents=documents,
            collection_name="nfcorpus-eval",
            embedding=embed_model,
            persist_directory=str(Path(temp_dir) / "chroma_nfcorpus"),
        )
        retriever = HybridRetriever(
            vectorstore=vectorstore,
            documents=documents,
            dense_k=args.k,
            sparse_k=args.k,
        )

        rows = []
        for index, (query_id, query_text) in enumerate(queries.items()):
            if index >= args.limit:
                break
            retrieved_docs = retriever.invoke(query_text)
            qrel = qrels.get(query_id, {})
            flags = [
                1 if doc.metadata.get("doc_id") in qrel and qrel[doc.metadata.get("doc_id")] > 0 else 0
                for doc in retrieved_docs
            ]
            rows.append(
                {
                    "query_id": query_id,
                    "recall@5": recall_at_k(flags, 5),
                    "ndcg@10": ndcg_at_k(flags, 10),
                    "mrr@10": mrr_at_k(flags, 10),
                }
            )

    summary = summarize_metrics(
        [
            {
                "recall@5": row["recall@5"],
                "ndcg@10": row["ndcg@10"],
                "mrr@10": row["mrr@10"],
            }
            for row in rows
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph.embeddings import BGEM3BiEncoderEmbeddings
from graph.prompt_defaults import DEFAULT_QUERY_TOWER_INSTRUCTION
from graph.retrieval_eval_utils import build_relevance_flags, load_jsonl


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _tokenize(text: str) -> List[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9\-\+\.]+|[\u4e00-\u9fff]{2,}", text or "")
        if token.strip()
    ]


def _dedupe(values: Iterable[str]) -> List[str]:
    unique: List[str] = []
    for value in values:
        value = _normalize_text(value)
        if value and value not in unique:
            unique.append(value)
    return unique


def _sample_keywords(sample: Dict[str, Any], include_negative_hints: bool = True) -> List[str]:
    values: List[str] = []
    values.append(sample.get("query", ""))
    values.extend(sample.get("medical_entities", []))
    values.extend(sample.get("relevant_keywords", []))
    if include_negative_hints:
        values.extend(sample.get("negative_hint_keywords", []))

    tokens: List[str] = []
    for value in values:
        if not value:
            continue
        tokens.append(value)
        tokens.extend(_tokenize(value))
    return _dedupe(tokens)


def _document_source(document: Document) -> str:
    metadata = document.metadata or {}
    return (
        metadata.get("source")
        or metadata.get("url")
        or metadata.get("doc_id")
        or ""
    )


def _document_text(document: Document) -> str:
    title = (document.metadata or {}).get("title", "")
    content = document.page_content or ""
    return _normalize_text(f"{title}\n\n{content}".strip())


def _keyword_overlap_score(text: str, keywords: Sequence[str]) -> float:
    lowered = text.lower()
    score = 0.0
    for keyword in keywords:
        keyword = keyword.lower().strip()
        if not keyword:
            continue
        if keyword in lowered:
            score += 1.0 if len(keyword) < 10 else 1.5
    return score


def _resolve_positive_document(
    sample: Dict[str, Any],
    documents: Sequence[Any],
) -> Tuple[str | None, str]:
    if sample.get("positive_document"):
        return _normalize_text(sample["positive_document"]), sample.get(
            "positive_source",
            (sample.get("relevant_sources") or ["seed_positive"])[0],
        )

    positive_sources = set(sample.get("relevant_sources", []))
    relevant_keywords = _sample_keywords(sample, include_negative_hints=False)
    candidates: List[Tuple[float, Document]] = []

    for document in documents:
        doc_source = _document_source(document)
        if positive_sources and doc_source not in positive_sources:
            continue

        flags = build_relevance_flags([document], sample)
        if not flags or not flags[0]:
            continue

        doc_text = _document_text(document)
        score = _keyword_overlap_score(doc_text, relevant_keywords)
        candidates.append((score, document))

    if not candidates:
        return None, ""

    _, best_document = sorted(
        candidates,
        key=lambda item: item[0],
        reverse=True,
    )[0]
    return _document_text(best_document), _document_source(best_document)


def _mine_negative_document(
    sample: Dict[str, Any],
    documents: Sequence[Any],
    retriever: Any,
    positive_source: str,
    positive_document: str,
    top_k: int,
) -> Tuple[str | None, str, str]:
    confusion_keywords = _sample_keywords(sample, include_negative_hints=True)
    negative_hint_keywords = _dedupe(sample.get("negative_hint_keywords", []))
    retrieved_documents = retriever.invoke(sample["query"])[:top_k]
    retrieved_flags = build_relevance_flags(retrieved_documents, sample)
    candidates: List[Tuple[float, str, Any]] = []

    for rank, (document, flag) in enumerate(zip(retrieved_documents, retrieved_flags), start=1):
        doc_source = _document_source(document)
        doc_text = _document_text(document)
        if flag or not doc_text:
            continue
        if positive_source and doc_source == positive_source:
            continue
        if doc_text == positive_document:
            continue

        score = _keyword_overlap_score(doc_text, confusion_keywords)
        if negative_hint_keywords:
            score += 2.0 * _keyword_overlap_score(doc_text, negative_hint_keywords)
        score += max(0, top_k - rank) * 0.1
        candidates.append((score, "retriever_mined_hard_negative", document))

    if not candidates:
        for document in documents:
            doc_source = _document_source(document)
            doc_text = _document_text(document)
            if not doc_text:
                continue
            if positive_source and doc_source == positive_source:
                continue
            if doc_text == positive_document:
                continue
            if build_relevance_flags([document], sample)[0]:
                continue

            score = _keyword_overlap_score(doc_text, confusion_keywords)
            if negative_hint_keywords:
                score += 2.0 * _keyword_overlap_score(doc_text, negative_hint_keywords)
            candidates.append((score, "corpus_overlap_negative", document))

    if not candidates:
        return None, "", ""

    score, strategy, best_document = sorted(
        candidates,
        key=lambda item: item[0],
        reverse=True,
    )[0]
    if score <= 0:
        return None, "", ""

    return _document_text(best_document), _document_source(best_document), strategy


def build_triplets(
    dataset: Sequence[Dict[str, Any]],
    documents: Sequence[Any],
    retriever: Any,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    skip_reasons = Counter()

    for sample in dataset:
        positive_document, positive_source = _resolve_positive_document(sample, documents)
        if not positive_document:
            skip_reasons["missing_positive_document"] += 1
            continue

        negative_document, negative_source, negative_strategy = _mine_negative_document(
            sample=sample,
            documents=documents,
            retriever=retriever,
            positive_source=positive_source,
            positive_document=positive_document,
            top_k=top_k,
        )
        if not negative_document:
            skip_reasons["missing_confusable_negative"] += 1
            continue

        rows.append(
            {
                "query": sample["query"],
                "positive_document": positive_document,
                "confusable_negative_document": negative_document,
                "query_type": sample.get("query_type", "unknown"),
                "medical_entities": sample.get("medical_entities", []),
                "relevant_sources": sample.get("relevant_sources", []),
                "relevant_keywords": sample.get("relevant_keywords", []),
                "positive_source": positive_source,
                "confusable_negative_source": negative_source,
                "negative_mining_strategy": negative_strategy,
                "source": sample.get("source", "query_tower_seed"),
                "split": sample.get("split", "train"),
            }
        )

    summary = {
        "input_samples": len(dataset),
        "built_triplets": len(rows),
        "skip_reasons": dict(skip_reasons),
        "query_type_distribution": dict(
            Counter(row.get("query_type", "unknown") for row in rows)
        ),
        "negative_strategy_distribution": dict(
            Counter(row.get("negative_mining_strategy", "unknown") for row in rows)
        ),
    }
    return rows, summary


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build BGE-M3 query tower LoRA triplets from medical seed queries."
    )
    parser.add_argument("--seed-dataset", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/train/query_tower_lora_built.jsonl"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/query_tower_training_data_summary.json"),
    )
    parser.add_argument("--model-name", default="BAAI/bge-m3")
    parser.add_argument(
        "--query-instruction",
        default=DEFAULT_QUERY_TOWER_INSTRUCTION,
    )
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    from ingestion import HybridRetriever, get_local_documents, open_vectorstore

    dataset = load_jsonl(args.seed_dataset)
    embedding_function = BGEM3BiEncoderEmbeddings(
        model_name=args.model_name,
        query_instruction=args.query_instruction,
    )
    documents = get_local_documents(embedding_function=embedding_function)
    vectorstore = open_vectorstore(embedding_function=embedding_function)
    retriever = HybridRetriever(
        vectorstore=vectorstore,
        documents=documents,
        dense_k=args.top_k,
        sparse_k=args.top_k,
    )

    rows, summary = build_triplets(
        dataset=dataset,
        documents=documents,
        retriever=retriever,
        top_k=args.top_k,
    )
    write_jsonl(args.output, rows)
    dump_json(
        args.summary_output,
        {
            "seed_dataset": str(args.seed_dataset),
            "output_dataset": str(args.output),
            **summary,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

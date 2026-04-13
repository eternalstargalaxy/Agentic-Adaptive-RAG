from typing import Any, Dict, List

from langchain.schema import Document

from graph.chains.hyde import hypothetical_document_chain
from graph.consts import RERANK_TOP_K
from graph.rerankers import get_bge_m3_reranker
from graph.state import GraphState
from ingestion import get_hybrid_retriever, normalize_queries


def _merge_documents(existing: List[Document], new_documents: List[Document]) -> List[Document]:
    merged = []
    seen = set()

    for document in existing + new_documents:
        content = (document.page_content or "").strip()
        source = document.metadata.get("source", "")
        key = f"{source}::{content[:400]}"
        if key in seen or not content:
            continue
        seen.add(key)
        merged.append(document)
    return merged


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")

    base_queries = normalize_queries(
        state.get("sub_queries") or state.get("retrieval_queries") or state["question"]
    )
    retrieval_queries = list(base_queries)

    if state.get("retrieval_round", 0) == 0 and state.get("use_hyde"):
        hypothetical_passage = hypothetical_document_chain.invoke(
            {"question": state.get("rewritten_question", state["question"])}
        )
        retrieval_queries.append(hypothetical_passage)

    retriever = get_hybrid_retriever()
    new_documents = retriever.invoke(retrieval_queries)
    for retrieval_rank, document in enumerate(new_documents, start=1):
        document.metadata = {
            **(document.metadata or {}),
            "retrieval_rank": retrieval_rank,
            "retrieval_query": state.get("rewritten_question", state["question"]),
        }

    merged_documents = _merge_documents(state.get("documents", []), new_documents)
    reranker = get_bge_m3_reranker()
    documents = reranker.rerank(
        state.get("rewritten_question", state["question"]),
        merged_documents,
        top_k=RERANK_TOP_K,
    )

    return {
        "documents": documents,
        "question": state["question"],
        "rewritten_question": state.get("rewritten_question", state["question"]),
        "retrieval_queries": base_queries,
        "sub_queries": [],
        "retrieval_round": state.get("retrieval_round", 0) + 1,
        "rerank_applied": True,
    }

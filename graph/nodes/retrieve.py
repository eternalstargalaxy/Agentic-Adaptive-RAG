from typing import Any, Dict, List

from langchain.schema import Document

from graph.chains.hyde import hypothetical_document_chain
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
    documents = _merge_documents(state.get("documents", []), new_documents)

    return {
        "documents": documents,
        "question": state["question"],
        "rewritten_question": state.get("rewritten_question", state["question"]),
        "retrieval_queries": base_queries,
        "sub_queries": [],
        "retrieval_round": state.get("retrieval_round", 0) + 1,
    }

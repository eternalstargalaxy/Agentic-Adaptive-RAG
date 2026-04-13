from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState


def _format_context(documents) -> str:
    blocks = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        header_parts = [
            f"Doc {index}",
            f"title={metadata.get('title', '') or 'unknown'}",
            f"source={metadata.get('source', '') or metadata.get('url', '') or 'unknown'}",
        ]
        if metadata.get("rerank_score") is not None:
            header_parts.append(f"rerank_score={metadata.get('rerank_score'):.4f}")
        blocks.append(
            "[" + " | ".join(header_parts) + "]\n" + document.page_content
        )
    return "\n\n".join(blocks)


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")

    question = state["question"]
    documents = state.get("documents", [])
    retrieval_queries = state.get("retrieval_queries", [state.get("rewritten_question", question)])
    route_strategy = state.get("route_strategy", "multi_hop")

    context = _format_context(documents)
    if not context and route_strategy == "no_retrieval":
        context = "No retrieval context used. Answer from model knowledge conservatively."
    generation = generation_chain.invoke(
        {
            "context": context,
            "question": question,
            "route_strategy": route_strategy,
            "retrieval_queries": "\n".join(retrieval_queries),
        }
    )

    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "retrieval_queries": retrieval_queries,
        "route_strategy": route_strategy,
    }

from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")

    question = state["question"]
    documents = state.get("documents", [])
    retrieval_queries = state.get("retrieval_queries", [state.get("rewritten_question", question)])
    route_strategy = state.get("route_strategy", "multi_hop")

    context = "\n\n".join(document.page_content for document in documents)
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

from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")

    question = state["question"]
    documents = state.get("documents", [])
    retrieval_queries = state.get("retrieval_queries", [state.get("rewritten_question", question)])

    context = "\n\n".join(document.page_content for document in documents)
    generation = generation_chain.invoke(
        {
            "context": context,
            "question": question,
            "retrieval_queries": "\n".join(retrieval_queries),
        }
    )

    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "retrieval_queries": retrieval_queries,
    }

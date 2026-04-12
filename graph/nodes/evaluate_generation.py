from typing import Any, Dict

from graph.consts import GENERATE, MAX_GENERATION_RETRIES, REWRITE_QUERY, WEBSEARCH
from graph.evaluation import evaluate_generation
from graph.state import GraphState


def evaluate_generation_node(state: GraphState) -> Dict[str, Any]:
    print("---EVALUATE GENERATION---")

    evaluation = evaluate_generation(
        question=state["question"],
        generation=state["generation"],
        documents=state.get("documents", []),
    )

    next_action = "end"
    retry_count = state.get("retry_count", 0)

    if not evaluation["grounded"]:
        if retry_count < MAX_GENERATION_RETRIES:
            next_action = GENERATE
            retry_count += 1
        else:
            next_action = WEBSEARCH
    elif not evaluation["addresses_question"]:
        if retry_count < MAX_GENERATION_RETRIES:
            next_action = WEBSEARCH if state.get("route") == "websearch" else REWRITE_QUERY
            retry_count += 1
        else:
            next_action = WEBSEARCH

    return {
        "evaluation": evaluation,
        "next_action": next_action,
        "retry_count": retry_count,
    }

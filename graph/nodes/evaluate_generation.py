from typing import Any, Dict

from graph.consts import (
    GENERATE,
    MAX_GENERATION_RETRIES,
    MULTI_HOP,
    NO_RETRIEVAL,
    REWRITE_QUERY,
    SINGLE_STEP,
    WEBSEARCH,
)
from graph.evaluation import evaluate_generation
from graph.state import GraphState


def evaluate_generation_node(state: GraphState) -> Dict[str, Any]:
    print("---EVALUATE GENERATION---")

    evaluation = evaluate_generation(
        question=state["question"],
        generation=state["generation"],
        documents=state.get("documents", []),
        route_strategy=state.get("route_strategy"),
    )

    next_action = "end"
    retry_count = state.get("retry_count", 0)
    forced_route_strategy = ""
    route_strategy = state.get("route_strategy", MULTI_HOP)

    if route_strategy == NO_RETRIEVAL and not evaluation["addresses_question"]:
        next_action = REWRITE_QUERY
        retry_count += 1
        forced_route_strategy = SINGLE_STEP
    elif not evaluation["grounded"]:
        if retry_count < MAX_GENERATION_RETRIES:
            if route_strategy == SINGLE_STEP:
                next_action = REWRITE_QUERY
                forced_route_strategy = MULTI_HOP
            else:
                next_action = GENERATE
            retry_count += 1
        else:
            next_action = WEBSEARCH
    elif not evaluation["addresses_question"]:
        if retry_count < MAX_GENERATION_RETRIES:
            if route_strategy == SINGLE_STEP:
                next_action = REWRITE_QUERY
                forced_route_strategy = MULTI_HOP
            else:
                next_action = WEBSEARCH if state.get("route") == "websearch" else REWRITE_QUERY
            retry_count += 1
        else:
            next_action = WEBSEARCH

    return {
        "evaluation": evaluation,
        "next_action": next_action,
        "retry_count": retry_count,
        "forced_route_strategy": forced_route_strategy,
    }

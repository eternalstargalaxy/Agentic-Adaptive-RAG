from typing import Any, Dict

from graph.chains.query_rewriter import query_rewriter
from graph.chains.router import RouteQuery, question_router
from graph.state import GraphState


def rewrite_query(state: GraphState) -> Dict[str, Any]:
    print("---REWRITE QUESTION---")

    question = state["question"]
    route: RouteQuery = question_router.invoke({"question": question})
    rewrite_plan = query_rewriter.invoke({"question": question})

    return {
        "question": question,
        "rewritten_question": rewrite_plan.rewritten_question,
        "retrieval_queries": rewrite_plan.retrieval_queries or [question],
        "search_query": rewrite_plan.rewritten_question,
        "route": route.datasource,
        "retrieval_round": 0,
        "retry_count": state.get("retry_count", 0),
        "documents": state.get("documents", []),
        "sub_queries": [],
        "use_hyde": rewrite_plan.use_hyde,
    }

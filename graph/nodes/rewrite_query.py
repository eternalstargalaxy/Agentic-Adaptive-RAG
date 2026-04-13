from typing import Any, Dict

from graph.corpus_pipeline import describe_corpus_manifest
from graph.corpus_profiles import get_active_corpus_profile
from graph.chains.query_rewriter import query_rewriter
from graph.chains.router import RouteQuery, question_router
from graph.consts import LLM, MULTI_HOP, NO_RETRIEVAL, SINGLE_STEP, VECTORSTORE
from graph.risk_guardrails import assess_medical_risk
from graph.state import GraphState


def rewrite_query(state: GraphState) -> Dict[str, Any]:
    print("---REWRITE QUESTION---")

    question = state["question"]
    profile = get_active_corpus_profile()
    forced_route_strategy = state.get("forced_route_strategy")
    if forced_route_strategy:
        route_strategy = forced_route_strategy
        route_rationale = "Escalated from a previous route decision."
    else:
        route: RouteQuery = question_router.invoke({"question": question})
        route_strategy = route.route_strategy
        route_rationale = route.rationale
    rewrite_plan = query_rewriter.invoke({"question": question})
    risk_assessment = assess_medical_risk(
        question=question,
        rewritten_question=rewrite_plan.rewritten_question,
        route_strategy=route_strategy,
    )
    if risk_assessment.prohibit_no_retrieval and route_strategy == NO_RETRIEVAL:
        route_strategy = risk_assessment.adjusted_route_strategy
        route_rationale = (
            f"{route_rationale} Risk guardrail applied: {risk_assessment.rationale}"
        )
    route_target = LLM if route_strategy == NO_RETRIEVAL else VECTORSTORE
    route_history = list(state.get("route_history", []))
    if not route_history or route_history[-1] != route_strategy:
        route_history.append(route_strategy)
    manifest_summary = describe_corpus_manifest(profile)

    return {
        "question": question,
        "rewritten_question": rewrite_plan.rewritten_question,
        "retrieval_queries": rewrite_plan.retrieval_queries or [question],
        "search_query": rewrite_plan.rewritten_question,
        "route": route_target,
        "route_strategy": route_strategy,
        "forced_route_strategy": "",
        "corpus_profile": profile.name,
        "corpus_version": profile.corpus_version,
        "corpus_manifest_path": manifest_summary.get("manifest_path", "") or "",
        "preferred_search_tools": list(profile.preferred_search_tools),
        "retrieval_round": 0,
        "retry_count": state.get("retry_count", 0),
        "documents": state.get("documents", []),
        "sub_queries": [],
        "use_hyde": rewrite_plan.use_hyde if route_strategy in {SINGLE_STEP, MULTI_HOP} else False,
        "route_rationale": route_rationale,
        "route_history": route_history,
        "risk_level": risk_assessment.risk_level,
        "risk_rationale": risk_assessment.rationale,
        "risk_signals": risk_assessment.matched_signals,
    }

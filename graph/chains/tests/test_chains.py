from graph.consts import GENERATE, RETRIEVE, WEBSEARCH
from graph.graph import route_after_document_grading, route_after_evaluation
from ingestion import normalize_queries


def test_normalize_queries_removes_empty_and_duplicates() -> None:
    assert normalize_queries([" agent memory ", "", "agent memory", "rag"]) == [
        "agent memory",
        "rag",
    ]


def test_route_after_document_grading_prefers_second_retrieval_round() -> None:
    state = {"next_action": RETRIEVE}
    assert route_after_document_grading(state) == RETRIEVE


def test_route_after_document_grading_can_fallback_to_websearch() -> None:
    state = {"next_action": WEBSEARCH}
    assert route_after_document_grading(state) == WEBSEARCH


def test_route_after_evaluation_accepts_finished_answer() -> None:
    state = {
        "next_action": "end",
        "evaluation": {"grounded": True, "addresses_question": True},
    }
    assert route_after_evaluation(state) == "__end__"


def test_route_after_evaluation_supports_regeneration() -> None:
    state = {"next_action": GENERATE, "evaluation": {}}
    assert route_after_evaluation(state) == GENERATE

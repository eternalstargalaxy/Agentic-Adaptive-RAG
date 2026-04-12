from graph.corpus_profiles import get_active_corpus_profile
from graph.retrieval_metrics import mrr_at_k, ndcg_at_k, recall_at_k, summarize_metrics


def test_medical_profile_is_default() -> None:
    assert get_active_corpus_profile().name == "medical_demo"


def test_recall_at_k() -> None:
    assert recall_at_k([1, 0, 1, 0], 2) == 0.5


def test_mrr_at_k() -> None:
    assert mrr_at_k([0, 1, 0], 3) == 0.5


def test_ndcg_at_k_is_bounded() -> None:
    value = ndcg_at_k([1, 0, 1], 3)
    assert 0 <= value <= 1


def test_summarize_metrics() -> None:
    summary = summarize_metrics(
        [
            {"recall@k": 0.5, "ndcg@k": 0.6, "mrr@k": 0.5},
            {"recall@k": 1.0, "ndcg@k": 0.8, "mrr@k": 1.0},
        ]
    )
    assert summary["recall@k"] == 0.75

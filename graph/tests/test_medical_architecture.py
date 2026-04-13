from graph.consts import NO_RETRIEVAL, SINGLE_STEP
from graph.corpus_pipeline import describe_corpus_manifest
from graph.corpus_profiles import MEDICAL_DEMO_PROFILE
from graph.risk_guardrails import assess_medical_risk


def test_high_risk_medical_question_cannot_stay_no_retrieval() -> None:
    assessment = assess_medical_risk(
        question="房颤合并心衰时，华法林和阿司匹林能不能联用？",
        rewritten_question="房颤患者合并心衰时华法林与阿司匹林联用的相互作用和禁忌",
        route_strategy=NO_RETRIEVAL,
    )
    assert assessment.adjusted_route_strategy == SINGLE_STEP
    assert assessment.risk_level == "high"


def test_low_risk_definition_question_can_keep_no_retrieval() -> None:
    assessment = assess_medical_risk(
        question="HbA1c 是什么？",
        rewritten_question="HbA1c 的定义",
        route_strategy=NO_RETRIEVAL,
    )
    assert assessment.adjusted_route_strategy == NO_RETRIEVAL


def test_medical_manifest_splits_serving_and_eval_assets() -> None:
    summary = describe_corpus_manifest(MEDICAL_DEMO_PROFILE)
    asset_names = {asset["name"] for asset in summary["assets"]}
    usages = {asset["usage"] for asset in summary["assets"]}
    assert "guideline_pages" in asset_names
    assert "pubmed_abstracts" in asset_names
    assert "nfcorpus_eval_registry" in asset_names
    assert "serve" in usages
    assert "eval" in usages

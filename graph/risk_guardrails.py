from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

from graph.consts import NO_RETRIEVAL, SINGLE_STEP


@dataclass(frozen=True)
class RiskAssessment:
    risk_level: str
    matched_signals: List[str]
    rationale: str
    adjusted_route_strategy: str

    @property
    def prohibit_no_retrieval(self) -> bool:
        return self.adjusted_route_strategy != NO_RETRIEVAL


HIGH_RISK_SIGNAL_LIBRARY = {
    "drug_interaction": (
        "drug interaction",
        "interaction",
        "contraindication",
        "多药联用",
        "联用",
        "相互作用",
        "禁忌",
        "冲突",
    ),
    "dosage_or_medication_plan": (
        "dose",
        "dosage",
        "prescription",
        "medication plan",
        "how much",
        "用量",
        "剂量",
        "处方",
        "吃多少",
        "怎么用药",
    ),
    "diagnosis_or_treatment_decision": (
        "treatment",
        "therapy",
        "management",
        "diagnosis",
        "differential",
        "诊断",
        "治疗",
        "处理",
        "方案",
        "鉴别",
    ),
    "special_population": (
        "pregnan",
        "breastfeeding",
        "child",
        "elderly",
        "renal failure",
        "liver failure",
        "妊娠",
        "孕妇",
        "哺乳",
        "儿童",
        "老年",
        "肾功能",
        "肝功能",
    ),
    "emergency_symptom": (
        "chest pain",
        "shortness of breath",
        "stroke",
        "sepsis",
        "emergency",
        "胸痛",
        "呼吸困难",
        "卒中",
        "脓毒症",
        "急诊",
    ),
    "high_stakes_drug": (
        "warfarin",
        "heparin",
        "insulin",
        "chemotherapy",
        "anticoag",
        "华法林",
        "肝素",
        "胰岛素",
        "化疗",
        "抗凝",
    ),
}

MEDICAL_ENTITY_PATTERNS: Sequence[str] = (
    "diabetes",
    "hypertension",
    "heart failure",
    "copd",
    "pneumonia",
    "atrial fibrillation",
    "糖尿病",
    "高血压",
    "心衰",
    "心力衰竭",
    "慢阻肺",
    "肺炎",
    "房颤",
    "冠心病",
)

MEDICATION_ENTITY_PATTERNS: Sequence[str] = (
    "metformin",
    "insulin",
    "aspirin",
    "warfarin",
    "clopidogrel",
    "metoprolol",
    "阿司匹林",
    "华法林",
    "氯吡格雷",
    "二甲双胍",
    "胰岛素",
    "美托洛尔",
)


def _contains_pattern(text: str, patterns: Sequence[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def _count_matches(text: str, patterns: Sequence[str]) -> int:
    return sum(1 for pattern in patterns if pattern in text)


def assess_medical_risk(question: str, route_strategy: str, rewritten_question: str = "") -> RiskAssessment:
    text = f"{question}\n{rewritten_question}".lower()
    matched_signals: List[str] = []

    for signal_name, patterns in HIGH_RISK_SIGNAL_LIBRARY.items():
        if _contains_pattern(text, patterns):
            matched_signals.append(signal_name)

    medication_count = _count_matches(text, MEDICATION_ENTITY_PATTERNS)
    disease_count = _count_matches(text, MEDICAL_ENTITY_PATTERNS)
    has_multi_factor_complexity = medication_count >= 2 or disease_count >= 2
    if has_multi_factor_complexity:
        matched_signals.append("multi_entity_medical_context")

    if route_strategy != NO_RETRIEVAL:
        return RiskAssessment(
            risk_level="guarded" if matched_signals else "low",
            matched_signals=matched_signals,
            rationale=(
                "The original router already selected a retrieval-backed path."
                if matched_signals
                else "No additional risk guardrail escalation was needed."
            ),
            adjusted_route_strategy=route_strategy,
        )

    if matched_signals:
        rationale = (
            "High-risk medical signals were detected "
            f"({', '.join(matched_signals)}), so No Retrieval is disabled."
        )
        return RiskAssessment(
            risk_level="high",
            matched_signals=matched_signals,
            rationale=rationale,
            adjusted_route_strategy=SINGLE_STEP,
        )

    # Short definition-style or general explanation questions can still pass through.
    if re.search(r"\bwhat is\b|是什么|定义|meaning|含义", text):
        return RiskAssessment(
            risk_level="low",
            matched_signals=[],
            rationale="Definition-style question without high-risk medical signals.",
            adjusted_route_strategy=route_strategy,
        )

    return RiskAssessment(
        risk_level="low",
        matched_signals=[],
        rationale="No high-risk medical signal detected by the route guardrail.",
        adjusted_route_strategy=route_strategy,
    )

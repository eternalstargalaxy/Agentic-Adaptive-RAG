from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple


@dataclass(frozen=True)
class CorpusProfile:
    name: str
    display_name: str
    description: str
    routing_topics: Tuple[str, ...]
    preferred_search_tools: Tuple[str, ...]
    urls: Tuple[str, ...]
    benchmark_focus: Tuple[str, ...]


GENERAL_AI_PROFILE = CorpusProfile(
    name="general_ai",
    display_name="通用 AI Agent RAG",
    description=(
        "面向 AI agents、prompt engineering、对抗攻击和 agentic RAG 设计模式的"
        "通用技术语料。"
    ),
    routing_topics=(
        "AI agent",
        "agent memory",
        "prompt engineering",
        "adversarial attack",
        "agentic RAG",
    ),
    preferred_search_tools=("search_web_general",),
    urls=(
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ),
    benchmark_focus=("Recall@5", "NDCG@10", "faithfulness"),
)


MEDICAL_DEMO_PROFILE = CorpusProfile(
    name="medical_demo",
    display_name="医学检索增强 RAG",
    description=(
        "面向医学问答和临床知识检索的启动语料。当前版本优先接入高可信公开医疗页面，"
        "为后续 PubMed、nfcorpus 和医学问答评测打底。"
    ),
    routing_topics=(
        "疾病",
        "症状",
        "检查项",
        "药物",
        "治疗方案",
        "医学缩写",
        "临床指南",
    ),
    preferred_search_tools=("search_pubmed", "search_medical_web"),
    urls=(
        "https://medlineplus.gov/diabetes.html",
        "https://medlineplus.gov/highbloodpressure.html",
        "https://medlineplus.gov/pneumonia.html",
        "https://medlineplus.gov/heartfailure.html",
        "https://medlineplus.gov/copd.html",
    ),
    benchmark_focus=("Recall@5", "NDCG@10", "MRR@10", "faithfulness"),
)


CORPUS_PROFILES: Dict[str, CorpusProfile] = {
    GENERAL_AI_PROFILE.name: GENERAL_AI_PROFILE,
    MEDICAL_DEMO_PROFILE.name: MEDICAL_DEMO_PROFILE,
}


def list_available_profiles() -> Tuple[str, ...]:
    return tuple(CORPUS_PROFILES.keys())


@lru_cache(maxsize=1)
def get_active_corpus_profile() -> CorpusProfile:
    profile_name = os.getenv("RAG_CORPUS_PROFILE", MEDICAL_DEMO_PROFILE.name).strip()
    return CORPUS_PROFILES.get(profile_name, MEDICAL_DEMO_PROFILE)

from textwrap import dedent
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from graph.consts import MULTI_HOP, NO_RETRIEVAL, SINGLE_STEP
from graph.prompt_defaults import MEDICAL_SAFETY_POLICY, build_profile_prompt_context
from model import grader_model


class RouteQuery(BaseModel):
    """
    Route a user query by reasoning complexity.
    """

    route_strategy: Literal["no_retrieval", "single_step", "multi_hop"] = Field(
        ...,
        description="Choose the best route strategy for this question.",
    )
    rationale: str = Field(description="Short reason for the routing decision.")


structured_llm_router = grader_model.with_structured_output(RouteQuery)
profile_context = build_profile_prompt_context()

system = dedent(
    f"""
    You are the routing controller for a medical adaptive RAG workflow.

    {profile_context}

    Your job is to choose the lowest-cost route that is still safe and reliable.
    Bias toward retrieval for domain-specific medical questions.

    {MEDICAL_SAFETY_POLICY}

    Available routes:

    1. {NO_RETRIEVAL}
    - Use only for stable, low-risk, definition-style, or general background questions.
    - Good examples: "HbA1c 是什么？", "COPD 的中文全称是什么？"
    - Do not use for treatment choice, diagnosis, differential diagnosis, drug interaction,
      contraindication, dosage, special population, emergency symptoms, or multi-condition planning.

    2. {SINGLE_STEP}
    - Use for straightforward factual medical questions that are likely answerable with one local
      retrieval pass over guidelines or curated abstracts.
    - Good examples: common symptoms, risk factors, lifestyle management, lab meaning,
      and high-level disease care principles already covered by the corpus.
    - This should be the default for simple domain questions.

    3. {MULTI_HOP}
    - Use for complex, compositional, or high-stakes medical questions that require combining
      multiple evidence fragments or verifying across sources.
    - Choose this for multi-comorbidity management, drug interaction reasoning, trade-off analysis,
      ambiguous symptom interpretation, comparative treatment questions, or cases likely to need
      web evidence in addition to the local corpus.

    Representative examples:
    - "2型糖尿病常见症状有哪些？" -> {SINGLE_STEP}
    - "高血压患者首先要做哪些生活方式管理？" -> {SINGLE_STEP}
    - "房颤合并心衰和慢阻肺时，抗凝与控率治疗要注意什么？" -> {MULTI_HOP}
    - "华法林和阿司匹林能不能联用？" -> {MULTI_HOP}

    Output requirements:
    - Choose exactly one route_strategy.
    - The rationale should be one short sentence.
    - Use the user's language for the rationale when possible.
    - Do not reveal hidden chain-of-thought.
    """
).strip()

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "User question:\n{question}\n\n"
            "Choose the best route strategy for this question.",
        ),
    ]
)

question_router = route_prompt | structured_llm_router

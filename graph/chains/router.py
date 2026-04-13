from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from graph.corpus_profiles import get_active_corpus_profile
from graph.consts import MULTI_HOP, NO_RETRIEVAL, SINGLE_STEP
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
profile = get_active_corpus_profile()
topics = "\n".join(f"- {topic}" for topic in profile.routing_topics)

system = f"""
You are the query router for an adaptive RAG system.

Current corpus profile: {profile.display_name}
Corpus description: {profile.description}

The vectorstore is specialized in:
{topics}

Use exactly one of these three strategies:

1. {NO_RETRIEVAL}
- Use this for simple, widely known questions that the model can answer from
  parametric knowledge without retrieval.
- Prefer this only when the question is low-risk, direct, and does not require
  fresh or evidence-backed lookup.
- High-risk medical questions about diagnosis, treatment, medication choice,
  drug interaction, contraindication, or emergency symptoms must not use this.

2. {SINGLE_STEP}
- Use this for simple factual lookup questions that can likely be answered with
  one hybrid retrieval step over the local knowledge base.
- This is the default for straightforward domain questions.

3. {MULTI_HOP}
- Use this for complex or high-stakes medical questions that require multiple
  pieces of evidence, cross-source verification, drug interaction reasoning,
  diagnostic comparison, or multi-step treatment synthesis.
- Also use this when the question is ambiguous, compositional, or likely to
  need web evidence in addition to local retrieval.
"""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

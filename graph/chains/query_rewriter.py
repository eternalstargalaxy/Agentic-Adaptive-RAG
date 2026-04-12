from typing import List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from graph.corpus_profiles import get_active_corpus_profile
from model import grader_model


class QueryRewritePlan(BaseModel):
    rewritten_question: str = Field(
        description="A clearer version of the original user question."
    )
    retrieval_queries: List[str] = Field(
        description="One to three focused retrieval queries for local search."
    )
    use_hyde: bool = Field(
        description="Whether a hypothetical answer should be generated for retrieval."
    )


structured_rewriter = grader_model.with_structured_output(QueryRewritePlan)
profile = get_active_corpus_profile()
topics = ", ".join(profile.routing_topics)

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You prepare questions for retrieval. Rewrite the user question so it is "
            "clear, domain-specific, and search-friendly. Return up to three retrieval "
            "queries. Use HyDE only when the question is short, ambiguous, or lacks keywords. "
            f"The current corpus profile is {profile.display_name}, and the retrieval topics "
            f"mainly cover: {topics}. Preserve key medical entities, abbreviations, lab names, "
            "and drug names when they appear.",
        ),
        ("human", "{question}"),
    ]
)

query_rewriter = rewrite_prompt | structured_rewriter

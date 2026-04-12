from typing import List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

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

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You prepare questions for retrieval. Rewrite the user question so it is "
            "clear, domain-specific, and search-friendly. Return up to three retrieval "
            "queries. Use HyDE only when the question is short, ambiguous, or lacks keywords.",
        ),
        ("human", "{question}"),
    ]
)

query_rewriter = rewrite_prompt | structured_rewriter

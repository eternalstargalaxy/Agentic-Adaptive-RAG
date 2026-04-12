from typing import List, Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from model import grader_model


class GapAnalysis(BaseModel):
    next_action: Literal["generate", "retrieve", "websearch"] = Field(
        description="Best next step for the workflow."
    )
    missing_information: str = Field(
        description="What is still missing after reviewing the current evidence."
    )
    sub_queries: List[str] = Field(
        description="Targeted follow-up retrieval queries that can fill the gaps."
    )
    search_query: str = Field(
        description="The best search query to use if web search is required."
    )


structured_gap_analyzer = grader_model.with_structured_output(GapAnalysis)

gap_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You inspect retrieved evidence for an adaptive RAG system. Decide whether "
            "the system should generate now, run one more local retrieval round, or "
            "fallback to web search. Prefer another local retrieval round before web "
            "search when the question is still within the local knowledge scope.",
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Current route:\n{route}\n\n"
            "Retrieval round:\n{retrieval_round}\n\n"
            "Relevant document count:\n{relevant_doc_count}\n\n"
            "Current evidence:\n{documents}",
        ),
    ]
)

gap_analyzer = gap_prompt | structured_gap_analyzer

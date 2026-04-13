from textwrap import dedent
from typing import List, Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from graph.prompt_defaults import MEDICAL_SAFETY_POLICY, build_profile_prompt_context
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
profile_context = build_profile_prompt_context()

gap_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent(
                f"""
                You are the evidence-gap planner for a medical adaptive RAG workflow.

                {profile_context}

                {MEDICAL_SAFETY_POLICY}

                You must choose exactly one next_action:
                - generate: the current evidence already covers the core question.
                - retrieve: the local corpus likely contains the missing evidence and a more focused
                  local retrieval query can find it.
                - websearch: local evidence is clearly insufficient, the topic requires external or
                  fresher evidence, or the question is complex enough to benefit from cross-source checking.

                Decision heuristics:
                - If relevant_doc_count is already strong and the evidence directly addresses the key ask,
                  prefer generate.
                - If retrieval_round is 0 and the missing pieces are narrow subtopics still within the local
                  corpus scope, prefer retrieve before websearch.
                - If retrieval_round >= 1 and evidence remains thin, bias toward websearch.
                - For drug interactions, contraindications, special populations, multi-comorbidity treatment
                  planning, and comparative management questions, bias toward websearch unless the evidence is
                  already clearly sufficient.

                Output rules:
                - missing_information should explain the unresolved evidence gap in one or two sentences.
                - sub_queries should contain zero to three targeted local retrieval queries.
                - search_query should always be a usable external search query, even when next_action is not websearch.
                - Use the user's language when possible; bilingual medical term expansion is allowed.
                """
            ).strip(),
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Current route:\n{route}\n\n"
            "Retrieval round:\n{retrieval_round}\n\n"
            "Relevant document count:\n{relevant_doc_count}\n\n"
            "Current evidence:\n{documents}\n\n"
            "Decide the best next action.",
        ),
    ]
)

gap_analyzer = gap_prompt | structured_gap_analyzer

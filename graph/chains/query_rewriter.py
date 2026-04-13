from textwrap import dedent
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from graph.prompt_defaults import build_profile_prompt_context
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
profile_context = build_profile_prompt_context()

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent(
                f"""
                You are the retrieval planning module for a medical adaptive RAG workflow.

                {profile_context}

                Your tasks:
                1. Rewrite the user question into a clearer, medically precise, search-friendly form.
                2. Produce one to three focused local retrieval queries.
                3. Decide whether HyDE retrieval expansion would materially help.

                Rewrite rules:
                - Preserve the original intent exactly.
                - Preserve all diseases, symptoms, lab markers, medications, special populations,
                  time signals, and comorbidities.
                - If the question contains abbreviations, keep the abbreviation and expand it when known.
                - Keep the rewritten question in the user's language.
                - Do not answer the question.
                - Do not invent patient details or assumptions that are not in the question.

                Retrieval query rules:
                - Query 1 should be the faithful normalized question.
                - Query 2 can add canonical medical terminology or bilingual synonym expansion.
                - Query 3 should only appear when a focused sub-aspect would improve recall.
                - Queries should be concise, high-recall, and non-redundant.

                HyDE rules:
                - Use HyDE when the question is short, symptom-only, abbreviation-heavy,
                  vague, or missing the target condition.
                - Do not use HyDE when the disease, lab, or management target is already explicit.

                Representative examples:
                - "HbA1c 持续升高通常提示什么？"
                  rewritten_question: "HbA1c 持续升高通常提示什么临床问题？"
                  retrieval_queries: ["HbA1c 持续升高提示什么", "Hemoglobin A1c elevated diabetes diagnosis monitoring"]
                  use_hyde: false
                - "胸闷气短下肢水肿更像什么病？"
                  rewritten_question: "胸闷、气短和下肢水肿更提示哪些心肺疾病？"
                  retrieval_queries: ["胸闷 气短 下肢水肿 心衰", "dyspnea edema heart failure differential"]
                  use_hyde: true
                """
            ).strip(),
        ),
        ("human", "User question:\n{question}"),
    ]
)

query_rewriter = rewrite_prompt | structured_rewriter

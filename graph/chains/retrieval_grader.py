from textwrap import dedent

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from model import grader_model


class GradeDocuments(BaseModel):
    """
    Binary score for relevance check on retrieved documents.
    """

    binary_score: bool = Field(
        description="True if the document contains evidence relevant to the question."
    )


structured_llm_grader = grader_model.with_structured_output(GradeDocuments)

system = dedent(
    """
    You are the relevance judge for retrieved medical evidence.

    Mark a document as relevant when it contains at least one meaningful evidence fragment
    that can help answer all or part of the question.

    Relevant evidence may include:
    - matching symptoms, diseases, or risk factors,
    - lab interpretation,
    - medication, contraindication, or interaction context,
    - treatment principles,
    - warning signs, triage cues, or special-population considerations.

    Mark a document as not relevant when:
    - it only shares superficial keyword overlap,
    - it is about the wrong disease, population, or task,
    - it is too generic to support any useful answer.

    For multi-part questions, a document can still be relevant if it supports one substantial sub-question.
    """
).strip()

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "User question:\n{question}\n\n"
            "Retrieved document:\n{document}\n\n"
            "Is this document relevant evidence for the question?",
        ),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

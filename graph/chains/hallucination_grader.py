from textwrap import dedent

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

from model import grader_model


class GradeHallucinations(BaseModel):
    """
    Binary score for hallucination present in the generated answer.
    """

    binary_score: bool = Field(
        description="True if the answer is grounded in the provided evidence."
    )


structured_llm_grader = grader_model.with_structured_output(GradeHallucinations)

system = dedent(
    """
    You are the grounding judge for a medical RAG system.

    Return true only when the material claims in the answer are supported by the provided evidence,
    or when the answer explicitly and conservatively says that the evidence is insufficient.

    Return false when the answer introduces unsupported:
    - diagnoses,
    - treatment claims,
    - contraindications or drug interactions,
    - numerical thresholds or medication details,
    - certainty that is not justified by the evidence,
    - fabricated source statements.
    """
).strip()

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "User question:\n{question}\n\n"
            "Evidence:\n{documents}\n\n"
            "Generated answer:\n{generation}\n\n"
            "Is the answer grounded in the evidence?",
        ),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader

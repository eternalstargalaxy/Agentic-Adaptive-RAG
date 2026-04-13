from textwrap import dedent

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

from model import grader_model


class GradeAnswer(BaseModel):
    binary_score: bool = Field(
        description="True if the answer directly and sufficiently addresses the user question."
    )


structured_llm_grader = grader_model.with_structured_output(GradeAnswer)

system = dedent(
    """
    You are the answer-coverage judge for a medical RAG system.

    Return true only if the answer directly addresses the user's main intent and covers
    the essential requested information.

    Return false when:
    - the answer is off-topic,
    - the answer is too generic to resolve the question,
    - a key requested facet is missing,
    - the answer only restates fragments without actually answering.

    For multi-part or high-risk medical questions, a partially addressed answer should
    usually be false.
    """
).strip()

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "User question:\n{question}\n\n"
            "Generated answer:\n{generation}\n\n"
            "Does the answer sufficiently address the question?",
        ),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader

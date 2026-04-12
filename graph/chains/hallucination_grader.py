from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

from model import grader_model


class GradeHallucinations(BaseModel):
    """
    Binary score for hallucination present in the generated answer.
    """

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'."
    )


structured_llm_grader = grader_model.with_structured_output(GradeHallucinations)

system = """
You are a grader assessing whether an LLM generation is grounded in a set of retrieved facts.
Give a binary score 'yes' or 'no'. 'Yes' means the answer is supported by the facts.
"""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts:\n\n{documents}\n\nLLM generation:\n{generation}"),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader

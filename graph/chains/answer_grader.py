from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

from model import grader_model


class GradeAnswer(BaseModel):
    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'."
    )


structured_llm_grader = grader_model.with_structured_output(GradeAnswer)

system = """
You are a grader assessing whether an answer addresses and resolves a question.
Give a binary score 'yes' or 'no'. 'Yes' means the answer addresses the question.
"""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question:\n\n{question}\n\nLLM generation:\n{generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader

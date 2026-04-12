from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from model import grader_model


class GradeDocuments(BaseModel):
    """
    Binary score for relevance check on retrieved documents.
    """

    binary_score: str = Field(
        description="Documents are relevant to the question, answer with 'yes' or 'no'."
    )


structured_llm_grader = grader_model.with_structured_output(GradeDocuments)

system = """
You are a grader assessing whether a retrieved document is relevant to a user question.
If the document contains keywords, facts, or semantic meaning that can help answer the
question, grade it as relevant. Give a binary score: 'yes' or 'no'.
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document:\n\n{document}\n\nUser question:\n{question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

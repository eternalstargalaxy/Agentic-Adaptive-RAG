from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from model import grader_model


class RouteQuery(BaseModel):
    """
    Route a user query to the most relevant datasource.
    """

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Choose whether the question should use local retrieval or web search.",
    )


structured_llm_router = grader_model.with_structured_output(RouteQuery)

system = """
You route user questions to either a local vectorstore or web search.

The vectorstore is specialized in:
- AI agents and agent memory
- prompt engineering
- adversarial attacks on LLMs
- agentic RAG system design patterns

Route to the vectorstore when the question can likely be answered from this
specialized knowledge base. Route to websearch when the user asks for current
events, fresh facts, or information outside that scope.
"""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

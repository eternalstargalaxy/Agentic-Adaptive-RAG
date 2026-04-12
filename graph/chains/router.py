from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from graph.corpus_profiles import get_active_corpus_profile
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
profile = get_active_corpus_profile()
topics = "\n".join(f"- {topic}" for topic in profile.routing_topics)

system = f"""
You route user questions to either a local vectorstore or web search.

Current corpus profile: {profile.display_name}
Corpus description: {profile.description}

The vectorstore is specialized in:
{topics}

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

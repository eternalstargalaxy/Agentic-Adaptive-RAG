from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from model import llm_model


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful RAG assistant. Use the provided context first, "
            "synthesize across sources, and be explicit when the evidence is incomplete. "
            "Prefer concise, factual answers. If the route strategy is no_retrieval, "
            "you may answer from parametric knowledge but should stay conservative.",
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Route strategy:\n{route_strategy}\n\n"
            "Retrieval queries used:\n{retrieval_queries}\n\n"
            "Context:\n{context}\n\n"
            "Answer the question using only supported information when possible.",
        ),
    ]
)

generation_chain = prompt | llm_model | StrOutputParser()

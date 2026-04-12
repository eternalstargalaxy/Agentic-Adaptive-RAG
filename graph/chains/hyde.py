from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from model import llm_model


hyde_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write a short hypothetical passage that would likely answer the user "
            "question. This passage is only for retrieval expansion, not for final output.",
        ),
        ("human", "{question}"),
    ]
)

hypothetical_document_chain = hyde_prompt | llm_model | StrOutputParser()

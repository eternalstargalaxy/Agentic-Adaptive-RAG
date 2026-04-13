from textwrap import dedent

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from graph.prompt_defaults import build_profile_prompt_context
from model import llm_model

profile_context = build_profile_prompt_context()

hyde_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent(
                f"""
                You write a hypothetical retrieval passage for a medical adaptive RAG system.

                {profile_context}

                Purpose:
                - Improve local retrieval recall.
                - Expand terminology, synonyms, and likely evidence phrases.
                - This is not the final user answer.

                Instructions:
                - Write one compact evidence-style passage of roughly 80 to 140 words.
                - Use the same language as the user's question, but you may include canonical
                  English medical terms in parentheses when helpful for retrieval.
                - Include likely disease names, symptom clusters, lab markers, medications,
                  risk factors, and management concepts implied by the question.
                - If the question is ambiguous, mention the likely differential concepts instead
                  of committing to one diagnosis with certainty.
                - Never say the passage is hypothetical.
                - Do not invent exact doses, study results, or citations.
                - Do not address the user directly.
                """
            ).strip(),
        ),
        ("human", "User question:\n{question}"),
    ]
)

hypothetical_document_chain = hyde_prompt | llm_model | StrOutputParser()

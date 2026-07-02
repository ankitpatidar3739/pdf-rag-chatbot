"""

Two-stage pipeline
Stage 1  Condense question
    (chat_history + follow-up) → standalone question
    Required so FAISS retrieval works correctly in multi-turn chat.
    Skipped when there is no prior history (saves one API call).

Stage 2  Answer generation
    standalone question → FAISS retriever → top-k chunks → LLM → answer
"""

from __future__ import annotations

import os
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


# ─── Prompts ───────────────────────────────────────────────────────────────────

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    "Given the chat history and a follow-up question, rephrase the follow-up "
    "into a standalone question that can be understood without the history.\n"
    "If the question is already standalone, return it unchanged.\n\n"
    "Chat History:\n{chat_history}\n\n"
    "Follow-up Question: {question}\n\n"
    "Standalone Question:"
)

QA_PROMPT = ChatPromptTemplate.from_template(
    "You are a precise document assistant. Answer using ONLY the context below.\n"
    "Do not use any external knowledge.\n\n"
    "Guidelines:\n"
    "- Be specific and reference details from the context.\n"
    "- If the answer is absent, say: \"I couldn't find information about this "
    "in the uploaded documents.\"\n"
    "- Use bullet points for lists.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _format_docs(docs: list[Document]) -> str:
    """Join retrieved chunks with a clear separator for the LLM context."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def _format_chat_history(history: list[dict]) -> str:
    """Convert Streamlit message dicts to a plain-text history string."""
    if not history:
        return "No previous conversation."
    lines = [
        f"{'Human' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history
    ]
    return "\n".join(lines)


# ─── Public API ────────────────────────────────────────────────────────────────

def build_rag_chain(
    vector_store: FAISS,
    model_name: str = "llama-3.1-8b-instant",
    top_k: int = 4,
) -> dict[str, Any]:
    
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set.")

    llm = ChatGroq(
        model=model_name,
        temperature=0.2,
        max_tokens=1024,
        groq_api_key=groq_api_key,
        timeout=30,      # fail fast instead of hanging on network issues
        max_retries=2,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    condense_chain = CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()

    return {"llm": llm, "retriever": retriever, "condense_chain": condense_chain}


def get_answer(
    chain_dict: dict[str, Any],
    question: str,
    chat_history: list[dict],
) -> tuple[str, list[Document]]:
    
    llm: ChatGroq = chain_dict["llm"]
    retriever = chain_dict["retriever"]
    condense_chain = chain_dict["condense_chain"]

    # Stage 1: condense only when there is prior context
    if chat_history:
        standalone_question: str = condense_chain.invoke({
            "chat_history": _format_chat_history(chat_history),
            "question": question,
        })
    else:
        standalone_question = question

    # Stage 2: retrieve → generate
    source_docs: list[Document] = retriever.invoke(standalone_question)
    answer: str = (QA_PROMPT | llm | StrOutputParser()).invoke({
        "context": _format_docs(source_docs),
        "question": standalone_question,
    })

    return answer, source_docs

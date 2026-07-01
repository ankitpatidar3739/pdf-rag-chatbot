"""
src/utils.py

Shared helper functions used by both Streamlit (app.py) and FastAPI
(backend/main.py).  No framework-specific imports here.
"""

from __future__ import annotations

from langchain_core.documents import Document


def format_sources(source_docs: list[Document]) -> list[dict]:
    """
    Convert LangChain Document objects into serialisable dicts for display.

    Deduplicates by (filename, page) so the same page is not cited twice.
    Snippets are trimmed to 250 characters for readability.

    Returns:
        [{"file": str, "page": int, "snippet": str}, ...]
    """
    seen: set[tuple] = set()
    formatted: list[dict] = []

    for doc in source_docs:
        meta = doc.metadata
        file_name: str = meta.get("source_file", meta.get("source", "Unknown"))
        # PyPDFLoader uses 0-based page numbers; convert to 1-based for display
        page_num: int = int(meta.get("page", 0)) + 1

        key = (file_name, page_num)
        if key in seen:
            continue
        seen.add(key)

        snippet: str = doc.page_content.strip().replace("\n", " ")
        if len(snippet) > 250:
            snippet = snippet[:247] + "..."

        formatted.append({"file": file_name, "page": page_num, "snippet": snippet})

    return formatted

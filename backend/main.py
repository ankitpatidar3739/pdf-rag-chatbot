"""
backend/main.py

FastAPI demo layer for DocuChat.

This module is intentionally SEPARATE from the Streamlit app.
Streamlit talks directly to src/ exactly as before — this file
does not power the UI at all.  It exists to demonstrate FastAPI
knowledge and can be explored interactively via Swagger at:

    http://127.0.0.1:8000/docs

Run with:
    uvicorn backend.main:app --reload
"""

from __future__ import annotations

import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.pdf_processor import process_pdfs as _process_pdfs
from src.rag_chain import build_rag_chain, get_answer
from src.utils import format_sources


# ─── App instance ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="DocuChat API",
    description=(
        "REST interface for the DocuChat RAG pipeline.\n\n"
        "The Streamlit UI (app.py) is completely independent — both use "
        "the same src/ modules directly.  This API is for demonstration only."
    ),
    version="1.0.0",
)


# ─── In-process state ──────────────────────────────────────────────────────────
# A real service would use Redis or a database.
# For this demo a module-level dict is clear and sufficient.
_state: dict = {
    "vector_store": None,
    "rag_chain": None,
    "file_names": [],
    "total_chunks": 0,
}


# ─── Pydantic models ───────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    """Request body for POST /ask."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The question to ask about the uploaded documents.",
        examples=["What is the main argument of this paper?"],
    )
    chat_history: list[dict] = Field(
        default_factory=list,
        description=(
            "Previous turns as a list of "
            '{"role": "user"|"assistant", "content": "..."} dicts.'
        ),
    )


class SourceChunk(BaseModel):
    """A single retrieved source chunk attached to an answer."""
    file: str
    page: int
    snippet: str


class AskResponse(BaseModel):
    """Response body for POST /ask."""
    question: str
    answer: str
    sources: list[SourceChunk]


class IngestResponse(BaseModel):
    """Response body for POST /ingest."""
    message: str
    files_processed: list[str]
    total_chunks: int


class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: str
    documents_loaded: bool
    loaded_files: list[str]
    total_chunks: int


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Utility"],
)
def health_check() -> HealthResponse:
    """
    Returns service status and whether documents are currently loaded.
    Useful to verify the API is running before sending questions.
    """
    return HealthResponse(
        status="ok",
        documents_loaded=_state["vector_store"] is not None,
        loaded_files=_state["file_names"],
        total_chunks=_state["total_chunks"],
    )


@app.get(
    "/models",
    summary="List available free-tier Groq models",
    tags=["Utility"],
)
def list_models() -> JSONResponse:
    """Returns the Groq models supported on the free tier."""
    models = [
        {
            "id": "llama-3.1-8b-instant",
            "description": "Fast, generous rate-limits — recommended default",
            "context_window": 131072,
        },
        {
            "id": "meta-llama/llama-4-scout-17b-16e-instruct",
            "description": "Newer model, preview, still free",
            "context_window": 131072,
        },
        {
            "id": "qwen/qwen3-32b",
            "description": "Stronger reasoning, preview, free",
            "context_window": 131072,
        },
    ]
    return JSONResponse(content={"models": models})


@app.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Upload and index PDF files",
    tags=["RAG Pipeline"],
)
async def ingest_pdfs(
    files: list[UploadFile] = File(..., description="One or more PDF files to index."),
    chunk_size: int = Query(600, ge=200, le=1500, description="Characters per chunk."),
    model_name: str = Query("llama-3.1-8b-instant", description="Groq model to use."),
    top_k: int = Query(4, ge=1, le=10, description="Chunks to retrieve per question."),
) -> IngestResponse:
    """
    Accepts PDF uploads, chunks them, builds a FAISS vector index,
    and stores a ready-to-use RAG chain in memory.

    Demonstrates: async endpoint, file uploads, query params with
    validation constraints, HTTPException, response model.
    """
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(
            status_code=400,
            detail="GROQ_API_KEY environment variable is not set.",
        )

    for f in files:
        if not (f.filename or "").lower().endswith(".pdf"):
            raise HTTPException(
                status_code=422,
                detail=f"'{f.filename}' is not a PDF. Only .pdf files are accepted.",
            )

    # Adapt FastAPI UploadFile to the interface process_pdfs expects.
    # Streamlit's UploadedFile exposes .name and .getvalue(); we mirror that.
    class _Adapter:
        def __init__(self, name: str, data: bytes) -> None:
            self.name = name
            self._data = data

        def getvalue(self) -> bytes:
            return self._data

    adapted: list[_Adapter] = []
    for f in files:
        adapted.append(_Adapter(f.filename, await f.read()))

    try:
        vector_store, total_chunks, file_names = _process_pdfs(
            adapted, chunk_size=chunk_size
        )
        rag_chain = build_rag_chain(
            vector_store, model_name=model_name, top_k=top_k
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    _state.update({
        "vector_store": vector_store,
        "rag_chain": rag_chain,
        "file_names": file_names,
        "total_chunks": total_chunks,
    })

    return IngestResponse(
        message=f"Successfully indexed {len(file_names)} file(s).",
        files_processed=file_names,
        total_chunks=total_chunks,
    )


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question about the indexed documents",
    tags=["RAG Pipeline"],
)
def ask_question(body: QuestionRequest) -> AskResponse:
    """
    Runs the full two-stage RAG pipeline and returns an answer
    with page-level source citations.

    Demonstrates: POST with Pydantic body, response model,
    404 for missing state, reuse of the same src/ logic as Streamlit.
    """
    if _state["rag_chain"] is None:
        raise HTTPException(
            status_code=404,
            detail="No documents loaded. Call POST /ingest first.",
        )

    try:
        answer, source_docs = get_answer(
            _state["rag_chain"],
            body.question,
            body.chat_history,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {exc}")

    return AskResponse(
        question=body.question,
        answer=answer,
        sources=[SourceChunk(**s) for s in format_sources(source_docs)],
    )


@app.delete(
    "/reset",
    summary="Clear the in-memory index",
    tags=["Utility"],
)
def reset() -> JSONResponse:
    """
    Clears all loaded documents and the RAG chain from memory.
    Demonstrates a DELETE endpoint and in-memory state management.
    """
    _state.update({
        "vector_store": None,
        "rag_chain": None,
        "file_names": [],
        "total_chunks": 0,
    })
    return JSONResponse(
        content={"message": "Index cleared. Upload new PDFs to /ingest."}
    )

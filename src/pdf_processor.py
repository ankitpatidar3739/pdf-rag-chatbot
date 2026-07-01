"""
src/pdf_processor.py

Handles PDF ingestion: reading, splitting into chunks, embedding,
and building a FAISS in-memory vector store.

Embedding model: all-MiniLM-L6-v2 (runs locally, no API key needed).
"""

from __future__ import annotations

import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the sentence-transformer embedding model.
    HuggingFaceEmbeddings caches the model after the first load.
    """
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def process_pdfs(
    uploaded_files: list,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> tuple[FAISS, int, list[str]]:
    """
    Ingest a list of file-like objects, build and return a FAISS index.

    Each file must expose:
        .name       — original filename (str)
        .getvalue() — raw PDF bytes

    This interface is satisfied by both Streamlit's UploadedFile and
    the _Adapter wrapper in backend/main.py, so the same function
    serves both entry points without modification.

    Returns:
        vector_store  — FAISS index ready for similarity search
        total_chunks  — number of text chunks indexed
        file_names    — list of original filenames processed
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_documents = []
    file_names: list[str] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for uploaded_file in uploaded_files:
            tmp_path = f"{tmp_dir}/{uploaded_file.name}"
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            pages = PyPDFLoader(tmp_path).load()

            for page in pages:
                page.metadata["source_file"] = uploaded_file.name

            all_documents.extend(splitter.split_documents(pages))
            file_names.append(uploaded_file.name)

    if not all_documents:
        raise ValueError(
            "No text could be extracted from the uploaded PDF(s). "
            "Scanned image-only PDFs require OCR preprocessing."
        )

    vector_store = FAISS.from_documents(all_documents, _get_embeddings())
    return vector_store, len(all_documents), file_names

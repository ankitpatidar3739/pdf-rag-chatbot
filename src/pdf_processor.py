
from __future__ import annotations

import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def _get_embeddings() -> HuggingFaceEmbeddings:
    
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

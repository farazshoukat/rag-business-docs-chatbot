"""Ingests an uploaded document into a session-specific Chroma collection."""

import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import CHROMA_PERSIST_DIR

_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings


def load_document(file_path: str):
    if file_path.endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    return TextLoader(file_path, encoding="utf-8").load()


def ingest_file(file_path: str, session_id: str):
    """Ingest a single file into a Chroma collection scoped to session_id."""
    docs = load_document(file_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=f"session_{session_id}",
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return len(chunks)


if __name__ == "__main__":
    # Manual test: ingest the sample FAQ doc as session "demo"
    count = ingest_file("sample_docs/company_faq.txt", "demo")
    print(f"Ingested {count} chunk(s) into session 'demo'.")
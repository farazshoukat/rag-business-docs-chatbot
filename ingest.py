"""Loads docs from sample_docs/, chunks them, embeds them, and stores in Chroma."""

import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import CHROMA_PERSIST_DIR, SAMPLE_DOCS_DIR


def load_documents():
    docs = []
    for filename in os.listdir(SAMPLE_DOCS_DIR):
        path = os.path.join(SAMPLE_DOCS_DIR, filename)
        if filename.endswith(".txt"):
            docs.extend(TextLoader(path, encoding="utf-8").load())
        elif filename.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
    return docs


def build_vectorstore():
    docs = load_documents()
    print(f"Loaded {len(docs)} document(s).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunk(s).")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print(f"Vector store saved to '{CHROMA_PERSIST_DIR}'.")
    return vectorstore


if __name__ == "__main__":
    build_vectorstore()
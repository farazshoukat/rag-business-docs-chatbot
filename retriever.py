"""Retriever: fetches relevant chunks from the Chroma vector store for a query."""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import CHROMA_PERSIST_DIR


def get_retriever(k: int = 3):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})
"""Retriever: fetches relevant chunks from a session-specific Chroma collection."""

from langchain_chroma import Chroma

from config import CHROMA_PERSIST_DIR
from ingest import get_embeddings


def get_retriever(session_id: str, k: int = 6):
    vectorstore = Chroma(
        collection_name=f"session_{session_id}",
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=get_embeddings(),
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})
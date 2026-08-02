"""RAG chain: combines retrieval with the Groq LLM to answer questions."""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import GROQ_API_KEY
from retriever import get_retriever

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the context provided below.
Give complete, detailed answers using everything relevant in the context — don't summarize down to one line if more detail is available.
If the answer isn't in the context, say you don't have that information — never make something up.

Context:
{context}
"""

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.2,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(session_id: str):
    retriever = get_retriever(session_id)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
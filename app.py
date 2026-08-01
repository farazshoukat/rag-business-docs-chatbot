"""Simple CLI test interface for the RAG chatbot."""

from chain import build_chain


def main():
    print("RAG Chatbot ready. Type 'exit' to quit.\n")
    rag_chain = build_chain()

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        answer = rag_chain.invoke(question)
        print("Bot:", answer, "\n")


if __name__ == "__main__":
    main()
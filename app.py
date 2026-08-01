"""Flask API for the RAG chatbot — used by the website/demo."""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from chain import build_chain

app = Flask(__name__)
CORS(app)

rag_chain = build_chain()


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    answer = rag_chain.invoke(question)
    return jsonify({"answer": answer})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
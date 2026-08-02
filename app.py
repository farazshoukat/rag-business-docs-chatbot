"""Flask API for the RAG chatbot — supports per-session document upload."""

import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS

from ingest import ingest_file
from chain import build_chain

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Cache built chains per session so we don't rebuild on every message
_chains = {}


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    session_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")
    file.save(file_path)

    try:
        chunk_count = ingest_file(file_path, session_id)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Failed to process file: {exc}"}), 500

    return jsonify({"session_id": session_id, "chunks": chunk_count})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    session_id = data.get("session_id", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400
    if not session_id:
        return jsonify({"error": "No session_id provided"}), 400

    if session_id not in _chains:
        _chains[session_id] = build_chain(session_id)

    answer = _chains[session_id].invoke(question)
    return jsonify({"answer": answer})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
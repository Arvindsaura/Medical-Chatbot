"""
Medical Chatbot — Flask Web App
CT Scan & Pulmonary Nodule Detection Specialist
Backend powered by Pinecone + Groq LLaMA + LangChain
"""

import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# ── Load environment variables ─────────────────────────────────────
load_dotenv()

# ── Import your existing helper functions ──────────────────────────
from src.helper import (
    download_embeddings,
    setup_pinecone,
    load_llm,
    create_rag_chain
)

# ── Flask App Setup ─────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret")

# ── Load AI Components ONCE (same as CLI initialization) ───────────
print("🔄 Initializing Medical AI System...")

print("🔹 Loading embedding model...")
embedding = download_embeddings()

print("🔹 Connecting to Pinecone...")
vector_store = setup_pinecone(
    index_name="medical-chatbot",
    embedding=embedding
)

print("🔹 Loading LLM (Groq)...")
llm = load_llm()

print("🔹 Building RAG chain...")
rag_chain = create_rag_chain(vector_store, llm)

print("✅ Medical AI System Ready.\n")


# ── Routes ──────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    """Render chat UI"""
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    """Handle chat query from frontend"""
    try:
        data = request.get_json()
        user_query = data.get("query", "").strip()

        if not user_query:
            return jsonify({"answer": "Please enter a valid query."}), 400

        # Invoke RAG chain
        response = rag_chain.invoke({"input": user_query})
        answer = response.get("answer", "No response generated.")

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({
            "answer": "An internal server error occurred.",
            "error": str(e)
        }), 500


# ── Run App ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False)
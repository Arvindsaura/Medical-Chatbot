# -------------------------------
# Imports
# -------------------------------
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings

from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from pinecone import Pinecone
from pinecone import ServerlessSpec


# -------------------------------
# 1. Load PDF Files
# -------------------------------
def load_pdf_files(data_path):
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


# -------------------------------
# 2. Split Documents (improved chunking)
# -------------------------------
def text_split(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    return texts


# -------------------------------
# 3. Download Embeddings
# -------------------------------
def download_embeddings():
    # FastEmbed runs BAAI/bge-small-en-v1.5 locally via ONNX runtime.
    # Much lighter than sentence-transformers (no PyTorch), and no outbound
    # HTTP calls needed — works on Render's free tier (512MB RAM).
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return embeddings


# -------------------------------
# 4. Setup Pinecone
# -------------------------------
def setup_pinecone(index_name, embedding):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    vector_store = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding
    )

    return vector_store


# -------------------------------
# 5. Load LLM (Google Gemini)
# -------------------------------


def load_llm():
    llm = ChatOpenAI(
        model="llama-3.1-8b-instant",
        openai_api_key=os.getenv("GROQ_API_KEY"),
        openai_api_base="https://api.groq.com/openai/v1",
        temperature=0.4,
        max_tokens=800,
        max_retries=0,
    )
    return llm

# -------------------------------
# 6. Create RAG Chain (specialized for CT / pulmonary nodules)
# -------------------------------
from langchain_core.runnables import RunnableLambda

def create_rag_chain(vector_store, llm):

    # MMR retriever: balances relevance + diversity across retrieved chunks
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 6,
            "lambda_mult": 0.7,
        }
    )

    # Truncate each retrieved chunk to avoid HTTP 413 (request too large)
    MAX_CHARS_PER_CHUNK = 600

    def truncate_docs(inputs):
        docs = inputs.get("context", [])
        for doc in docs:
            if len(doc.page_content) > MAX_CHARS_PER_CHUNK:
                doc.page_content = doc.page_content[:MAX_CHARS_PER_CHUNK] + "..."
        return inputs

    template = """You are a friendly, knowledgeable medical AI assistant. Your knowledge comes from comprehensive medical literature and textbooks.

BEHAVIOR RULES:

1. **Casual greetings** ("hi", "hello", "how are you", etc.): Respond naturally and briefly like a friendly assistant. Example: "Hello! How can I help you today? Feel free to ask me any health or medical questions."
2. **Medical questions**: Use the retrieved context first; supplement with general medical knowledge if needed.
3. **Non-medical, non-greeting inputs**: Politely steer back toward health topics in one short sentence.
4. Keep responses concise and easy to understand. Do NOT show your reasoning or thinking steps.
5. For medical answers only: add a brief disclaimer that responses are informational and not a substitute for professional advice.
6. **Follow-up questions**: If the user refers to something mentioned earlier (e.g., "the above", "that condition", "those vegetables"), use the conversation history below to understand what they mean.

Previous conversation:
{chat_history}

Context from medical literature:
{context}

User:
{input}

Assistant:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "input", "chat_history"]
    )

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    return rag_chain

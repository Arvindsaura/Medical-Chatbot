# -------------------------------
# Imports
# -------------------------------
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

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
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
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
        model="llama-3.3-70b-versatile",
        openai_api_key=os.getenv("GROQ_API_KEY"),
        openai_api_base="https://api.groq.com/openai/v1",
        temperature=0.2,
        max_tokens=800,
    )
    return llm

# -------------------------------
# 6. Create RAG Chain (specialized for CT / pulmonary nodules)
# -------------------------------
def create_rag_chain(vector_store, llm):

    # MMR retriever: balances relevance + diversity across retrieved chunks
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 10,
            "lambda_mult": 0.7,
        }
    )

    template = """You are a specialist AI radiology assistant focused exclusively on **CT scan analysis and pulmonary nodule detection**.

Your expertise includes:
- Interpreting CT scan findings related to lung nodules (solid, sub-solid, ground-glass opacity)
- Lung-RADS classification and risk stratification
- Fleischner Society guidelines for incidental pulmonary nodule management
- Nodule morphology analysis (size, shape, margins, density, calcification patterns)
- Differential diagnosis of pulmonary nodules (benign vs malignant indicators)
- Follow-up imaging recommendations based on nodule characteristics
- TNM staging considerations when malignancy is suspected
- Associated findings (lymphadenopathy, pleural effusion, emphysema)

RULES:

1. Prioritize the retrieved context as the primary source of truth.
2. If the retrieved context is insufficient BUT the question is still within pulmonary CT imaging scope, you may use general medical radiology knowledge.
3. Clearly distinguish your reasoning using:
   - "Based on retrieved context:"
   - "Based on general radiology knowledge:"
4. If the question is unrelated to CT scans or lung imaging, redirect briefly in one sentence.
5. Provide structured, concise, clinical responses.
6. Avoid unnecessary repetition.
7. Always include a short informational disclaimer.

Context:
{context}

Patient Query / Clinical Question:
{input}

Specialist Assessment:
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "input"]
    )

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    return rag_chain

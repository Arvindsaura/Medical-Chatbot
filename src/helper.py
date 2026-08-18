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
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
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
    # FastEmbed uses a lightweight ONNX runtime (~67MB model, no PyTorch).
    # Works on Render free tier (512MB RAM).
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return embeddings


# -------------------------------
# 4. Setup Pinecone
# -------------------------------
def setup_pinecone(index_name, embedding):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    if index_name not in [idx.name for idx in pc.list_indexes()]:
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
        model="groq/compound",
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

    def format_docs(docs):
        """Format retrieved docs into a single string, truncated per chunk."""
        parts = []
        for doc in docs:
            content = doc.page_content
            if len(content) > MAX_CHARS_PER_CHUNK:
                content = content[:MAX_CHARS_PER_CHUNK] + "..."
            parts.append(content)
        return "\n\n".join(parts)

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

    # Build a manual chain that correctly passes ALL three keys to the prompt.
    # create_retrieval_chain only forwards {input} and {context}, dropping
    # {chat_history} — which causes a KeyError in the PromptTemplate.
    def run_rag(inputs: dict) -> dict:
        query = inputs["input"]
        chat_history = inputs.get("chat_history", "")

        # Retrieve relevant docs
        docs = retriever.invoke(query)
        context_str = format_docs(docs)

        # Build and invoke the prompt → LLM
        filled_prompt = prompt.format(
            input=query,
            context=context_str,
            chat_history=chat_history
        )
        result = llm.invoke(filled_prompt)

        # result is an AIMessage; extract text
        answer = result.content if hasattr(result, "content") else str(result)
        return {"answer": answer, "context": docs}

    return RunnableLambda(run_rag)

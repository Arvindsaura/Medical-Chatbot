"""
Debug script — prints the FULL exception so we know exactly what is failing.
"""
import os, traceback
from dotenv import load_dotenv
load_dotenv()

from src.helper import download_embeddings, setup_pinecone, load_llm, create_rag_chain

print("Loading embeddings...")
embedding = download_embeddings()

print("Connecting to Pinecone...")
vector_store = setup_pinecone(index_name="medical-chatbot", embedding=embedding)

print("Loading LLM...")
llm = load_llm()

print("Building RAG chain...")
rag_chain = create_rag_chain(vector_store, llm)

print("Sending test query...\n")
try:
    response = rag_chain.invoke({"input": "hi"})
    print("SUCCESS:", response)
except Exception as e:
    print("=== FULL ERROR ===")
    traceback.print_exc()
    print("=== ERROR MESSAGE ===")
    print(str(e))

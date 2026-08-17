# Medical Chatbot

A specialized AI-powered Medical Chatbot focused on CT Scan & Pulmonary Nodule Detection, built using Flask, LangChain, Pinecone, and Groq (LLaMA).

## Project Summary

This project is a web-based conversational AI designed to assist with medical queries, specifically specializing in pulmonary nodule detection and CT scan interpretations. It utilizes Retrieval-Augmented Generation (RAG) to provide accurate and context-aware responses based on a provided knowledge base.

**Key Technologies:**
- **Backend Framework:** Flask
- **LLM Engine:** Groq (LLaMA models)
- **Vector Database:** Pinecone
- **Orchestration:** LangChain
- **Embeddings:** HuggingFace / Sentence Transformers
- **Frontend:** HTML, CSS (Modern UI), JavaScript

## Features
- Interactive Web Chat Interface
- Fast and accurate retrieval from a specialized medical knowledge base
- Context-aware RAG implementation using LangChain
- Vector similarity search with Pinecone

## Setup Instructions

### Prerequisites
- Python 3.8+
- [Pinecone](https://www.pinecone.io/) Account & API Key
- [Groq](https://groq.com/) Account & API Key

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Arvindsaura/Medical-Chatbot.git
   cd Medical-Chatbot
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   PINECONE_API_KEY=your_pinecone_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   SECRET_KEY=your_flask_secret_key_here
   ```

5. **Initialize the Vector Store (First Run Only):**
   Run the data processing and vector indexing script to populate Pinecone:
   ```bash
   python store_index.py
   ```

### Running the Application

Start the Flask server:
```bash
python app.py
```
The application will be accessible at `http://localhost:5000` (or `http://127.0.0.1:5000`).

## Project Structure
- `app.py`: Main Flask application handling routing and API endpoints.
- `store_index.py`: Script to process data and store embeddings in Pinecone.
- `src/helper.py`: Contains core logic for LLM setup, RAG chain creation, and vector store connection.
- `templates/`: HTML templates for the chat interface.
- `static/`: CSS and JS assets for the frontend.
- `data/`: Raw medical data documents (PDFs, etc.).

## License
MIT License

# 📢 Marketing Research Agent

This project implements an **Agentic RAG (Retrieval-Augmented Generation)** system using **LangGraph** and **FastAPI** to assist with **marketing research and ad copy generation**. It features a multi-step, dynamic workflow that intelligently decides whether to retrieve information from its knowledge base, generate a response, or creatively rewrite ad text.

---

## ✨ Features
- ⚡ **FastAPI Backend** → Scalable API serving the agent's functionality  
- 🧠 **LangGraph Agent** → Stateful, multi-step agent for complex decision-making  
- 📚 **RAG System** → FAISS vector store retrieves and answers questions from the indexed knowledge base  
- ✅ **Dynamic Document Grader** → LLM ensures only relevant documents are used  
- 📄 **PDF Uploads** → Expand the knowledge base with new PDF documents  
- 💻 **Interactive UI** → Simple web interface for chatting with the agent  

---

## 🚀 Getting Started

### 1. Prerequisites
- Python **3.8+**  
- Git  
- A text editor (VS Code, Sublime, etc.)  

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 3. Set Up a Virtual Environment
```bash
python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install fastapi uvicorn python-dotenv langchain_core langchain_groq langgraph langchain_huggingface langchain_community langchain_text_splitters pydantic
```

(Optional) Export requirements:
```bash
pip freeze > requirements.txt
```

### 5. API Key Configuration
The application requires a GROQ API key.

Create a `.env` file in the project root:
```env
GROQ_API_KEY="your_groq_api_key_here"
```

### 6. Run the Server
From the project’s root directory:
```bash
uvicorn app:app --reload
```

The server will now be running at:

🌐 UI → http://127.0.0.1:8000  
📖 API Docs → http://127.0.0.1:8000/docs  

---

## 🌐 Usage
- **Web UI** → Chat with the agent through the browser.  
- **API Docs** → Test endpoints like `/chat` and `/upload_pdf`.  
- **PDF Upload** → Expand the agent’s knowledge base with your own documents.  

---

## 🏛️ Project Architecture
At the core of this project is a **LangGraph state machine** that powers the agent’s decision-making. The agent can:

- Retrieve relevant knowledge  
- Evaluate document relevance  
- Generate context-aware responses  
- Rewrite ad copy dynamically  

This design enables a **more powerful and flexible RAG workflow** compared to standard pipelines.

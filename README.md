Technical Write-up: Agentic RAG for Marketing Ad Copy
This project implements a LangGraph-based agentic RAG system designed to assist with marketing ad copy and research. The solution is built with a FastAPI backend to expose its functionality via a web API, ensuring it's a deployable and scalable service. The core architecture leverages LangGraph to create a stateful, multi-step reasoning workflow.

Architecture and Tools
The system's architecture is centered around LangGraph, which defines a cyclical graph of nodes representing different steps in the agent's reasoning process. The nodes are:

agent: The primary reasoning node. It uses a tool-calling LLM to decide the next step—either to use a tool or provide a direct response. For initial user queries, it can trigger the retrieve node via the retriever_tool. For follow-up conversational turns, it provides a concise, direct answer.

retrieve: A ToolNode that executes the retriever_tool, fetching relevant documents from a FAISS vector store.

grade_documents: A conditional node that acts as a router. It uses a dedicated LLM to evaluate the relevance of the retrieved documents to the user's question, directing the flow to either generate or rewrite.

generate: Creates a final answer by synthesizing information from the retrieved documents and the user's query, adhering to a specific marketing-oriented prompt.

rewrite: An alternative path for when no relevant documents are found. It uses the LLM to rewrite and enhance the original ad copy, suggesting different tones and platform optimizations.

The project uses LangChain components like WebBaseLoader, PyPDFLoader, FAISS, and RecursiveCharacterTextSplitter for data loading, chunking, and vector storage. The LLMs are provided by ChatGroq, with specific models like Gemma2-9b-It and llama-3.3-70b-versatile used for different tasks, demonstrating model specialization within the agent's workflow. The entire system is served via FastAPI, with endpoints for chatting (/chat) and for uploading and indexing new PDF documents (/upload_pdf).

Use of Agentic RAG
This solution is a prime example of Agentic RAG. It goes beyond simple "retrieve and generate" by incorporating decision-making into the retrieval process itself. The agent doesn't just retrieve documents; it first attempts to decide if a tool is needed, then uses a separate, specialized "grader" LLM to evaluate the relevance of the retrieved information. This multi-step, dynamic flow is more robust than a static RAG pipeline.

This approach helps with complex, multi-step reasoning by:

Routing: The grade_documents node acts as a router, ensuring the system either retrieves information from its knowledge base or, if the query is unrelated, creatively rewrites the ad copy. This prevents the LLM from trying to answer with irrelevant context, thereby improving response quality and reducing hallucinations.

Improving Precision and Recall: By pre-loading and allowing the addition of new documents (via PDF uploads), the system's knowledge base can grow. The agent's ability to precisely grade the relevance of retrieved chunks improves precision, as it only uses documents it deems useful for generation. The tool-use component ensures the system has a high recall of its available knowledge, as it actively searches for information rather than relying on a static context window.

Knowledge Graph Integration
While the current implementation uses a traditional vector store (FAISS), it could be significantly enhanced by integrating a Knowledge Graph (KG).

A KG could represent entities and their relationships, such as:

Entities: Marketing concepts (e.g., "SEO," "PPC," "Social Media Marketing"), Ad Platforms ("Instagram," "LinkedIn," "Twitter"), Creative Types ("Video Ad," "Carousel," "Static Image"), and User Intent ("Awareness," "Lead Generation," "Conversion").

Relationships: SEO is_a_topic_of Digital Marketing, Instagram is_a_platform_for Visual Ads, Video Ad drives Awareness.

Example:
If a user asks, "How can I improve my marketing on Instagram?", a KG could be used to:

Identify Instagram as an entity related to Social Media Marketing.

Find relationships linking Instagram to Creative Types like "Reels" and "Stories" and to User Intent like "Brand Awareness."

The agent could then use this structured knowledge to formulate a more targeted and contextually rich response, even before or in conjunction with retrieving unstructured text from the FAISS vector store. This would allow the agent to reason about related concepts more effectively than a simple semantic search alone.

Evaluation Strategy
Evaluating this agent's performance requires a multi-faceted approach.

Relevance & Hallucination Rate (Manual):

Method: A human evaluator would rate the agent's responses on a scale (e.g., 1-5) for relevance to the original query and factuality.

Metrics: Percentage of hallucination-free responses and average relevance score. This is crucial for verifying the grade_documents node is working correctly.

ROUGE Score for Summaries (Automated):

Method: For queries that result in a summary (e.g., in the generate node), compare the generated answer against a set of human-written reference answers using ROUGE metrics (e.g., ROUGE-L).

Metrics: ROUGE-L to measure the longest common subsequence, indicating content overlap and fluency.

F1 Score for Extraction (Automated):

Method: For specific queries requiring data extraction (e.g., "What are the key stats for PPC ads?"), compare the agent's extracted information against a ground truth set of facts.

Metrics: F1 score (the harmonic mean of precision and recall) to measure the accuracy of extracted data.

Tool-Use Accuracy (Automated/Manual):

Method: Test cases would verify that the agent correctly calls the retriever_tool when appropriate and decides to rewrite when a query is out of scope.

Metrics: Percentage of correct tool-use decisions.

Pattern Recognition and Improvement Loop
This agent can be adapted to improve over time by incorporating memory and a feedback loop.

Memory Modules: The current AgentState in LangGraph includes a messages list, which serves as a basic form of short-term memory. This allows the agent to maintain context within a single conversation turn. For long-term adaptation, a persistent memory module (e.g., using a database) could store successful prompt-response pairs.

Feedback Loop: A human-in-the-loop or automated feedback mechanism could be implemented. For example, a user could provide a thumbs-up or thumbs-down on a response. Bad responses could trigger a prompt refinement loop, where the system automatically revises the prompt used in the generate or rewrite nodes to avoid similar mistakes. The revised prompt, along with the corrected response, would then be stored in a "knowledge base of failures" to prevent recurrence. This could also be used to fine-tune the grade_documents LLM over time.

1. Prerequisites
First, make sure you have the following installed:

Python 3.8+: The project is built with Python.

Git: To clone the repository from GitHub.

A text editor (like VS Code or Sublime Text).

Steps to Run the Project Locally
To run this FastAPI project on your local machine, follow these steps.

1. Prerequisites
First, make sure you have the following installed:

Python 3.8+: The project is built with Python.

Git: To clone the repository from GitHub.

A text editor (like VS Code or Sublime Text).

2. Get the Code
Open your terminal or command prompt and clone the repository you created on GitHub. Replace the URL with your project's URL.

Bash

git clone https://github.com/your-username/your-repo-name.git
Bash

cd your-repo-name

3. Set Up the Environment
It is highly recommended to use a virtual environment to manage dependencies.

Create a virtual environment:

Bash

python -m venv .venv
Activate the virtual environment:

On Windows:

Bash

.venv\Scripts\activate


Install dependencies: After activating the environment, install the required Python packages. You'll need to create a requirements.txt file if you haven't already.

Bash

pip install fastapi uvicorn python-dotenv langchain_core langchain_groq langgraph langchain_huggingface langchain_community langchain_text_splitters pydantic
Once these are installed, you can generate your requirements.txt file for future use:

Bash

pip freeze > requirements.txt
Now, to install dependencies in the future, you would use:

Bash

pip install -r requirements.txt


Steps to Run the Project Locally
To run this FastAPI project on your local machine, follow these steps.

1. Prerequisites
First, make sure you have the following installed:

Python 3.8+: The project is built with Python.

Git: To clone the repository from GitHub.

A text editor (like VS Code or Sublime Text).

2. Get the Code
Open your terminal or command prompt and clone the repository you created on GitHub. Replace the URL with your project's URL.

Bash

git clone https://github.com/your-username/your-repo-name.git
Next, navigate into the project directory:

Bash

cd your-repo-name
3. Set Up the Environment
It is highly recommended to use a virtual environment to manage dependencies.

Create a virtual environment:

Bash

python -m venv .venv
Activate the virtual environment:

On Windows:

Bash

.venv\Scripts\activate
On macOS/Linux:

Bash

source .venv/bin/activate
Install dependencies: After activating the environment, install the required Python packages. You'll need to create a requirements.txt file if you haven't already.

Bash

pip install fastapi uvicorn python-dotenv langchain_core langchain_groq langgraph langchain_huggingface langchain_community langchain_text_splitters pydantic
Once these are installed, you can generate your requirements.txt file for future use:

Bash

pip freeze > requirements.txt
Now, to install dependencies in the future, you would use:

Bash

pip install -r requirements.txt
4. Configure API Key
The application requires a GROQ API key.

Create a file named .env in your project's root directory.

Add your API key to this file in the following format:

GROQ_API_KEY="your_groq_api_key_here"

5. Run the Application
Finally, start the FastAPI server using uvicorn. The reload flag will automatically restart the server when you make code changes.

Bash

uvicorn app:app --reload
The server will now be running on your local machine. You can access the application by opening your web browser and navigating to:

UI: http://127.0.0.1:8000

API Docs: http://127.0.0.1:8000/docs (for testing the API endpoints like /chat and /upload_pdf)

# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from pathlib import Path
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated, Sequence, TypedDict, Literal
import os
import warnings

# LangGraph / LangChain
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

warnings.filterwarnings("ignore")
load_dotenv()

app = FastAPI()

# CORS (open for dev; tighten for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === LLM + Embeddings ===
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("Warning: GROQ_API_KEY not set in environment.")

# base llm used for general tasks (you can change model_name as required)
llm = ChatGroq(model_name="Gemma2-9b-It", temperature=0, streaming=True)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# === RAG base: preload a blog post (cached) ===
CACHE_FILE = "cached_blog.txt"
if Path(CACHE_FILE).exists():
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        raw_content = f.read()
else:
    docs = WebBaseLoader(
        "https://blog.hubspot.com/marketing"    
    ).load()
    raw_content = docs[0].page_content
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        f.write(raw_content)

# === Text splitting & vectorstore setup ===
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=100, chunk_overlap=25)
doc_splits = text_splitter.split_documents([Document(page_content=raw_content)])

VECTOR_DIR = "faiss_index"
if os.path.exists(f"{VECTOR_DIR}/index.faiss"):
    vectorstore = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_documents(doc_splits, embedding=embeddings)
    vectorstore.save_local(VECTOR_DIR)

# Retriever tool for LangGraph
retriever = vectorstore.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "this is related ai agents blogs",
)
tools = [retriever_tool]

# LangGraph Agent State typed dict
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Grader output schema
class GradeOutput(BaseModel):
    binary_score: str = Field(description="'yes' or 'no'")

def grade_documents(state) -> Literal["generate", "rewrite"]:
    print("=== [NODE: RETRIEVE] ===")
    # Use model_name param (not `model`)
    model = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0).with_structured_output(GradeOutput)
    messages = state["messages"]

    # Extract likely question and context safely
    question = None
    context = None
    for m in messages:
        if isinstance(m, HumanMessage):
            if not question:
                question = m.content
    # context: last message content
    context = messages[-1].content if messages else ""

    prompt = PromptTemplate(
        template="""You are a grader checking if a marketing blog is relevant to the user's research.
Document:
{context}
Question: {question}
Is the document relevant? Answer 'yes' or 'no'.""",
        input_variables=["context", "question"]
    )

    result = (prompt | model).invoke({"context": context, "question": question})
    # result is model structured output with binary_score
    if getattr(result, "binary_score", "") == "yes":
        print("=== [DECISION: DOCS RELEVANT] ===")
        return "generate"
    else:
        print("=== [DECISION: DOCS NOT RELEVANT] ===")
        return "rewrite"

def agent(state):
    print("=== [NODE: AGENT] ===")
    messages = state["messages"]
    # instantiate model with correct kwarg
    llm_local = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

    # follow-up (concise) behavior if more than one message
    if len(messages) > 1:
        last_message = messages[-1]
        question = last_message.content
        prompt = PromptTemplate(
            template="""You are a marketing assistant.
Only answer the question directly in one sentence or less.
Do NOT explain, expand, reflect, or rephrase.
Question: {question}""",
            input_variables=["question"]
        )
        chain = prompt | llm_local
        response = chain.invoke({"question": question})
        # ensure we return an AIMessage
        content = response.content if hasattr(response, "content") else str(response)
        return {"messages": [AIMessage(content=content)]}
    else:
        # first-time message: allow tools
        llm_with_tool = llm.bind_tools(tools)
        response = llm_with_tool.invoke(messages)
        # response may be a message-like object; ensure proper structure
        if isinstance(response, AIMessage):
            return {"messages": [response]}
        else:
            content = getattr(response, "content", str(response))
            return {"messages": [AIMessage(content=content)]}

def rewrite(state):
    print("=== [NODE: REWRITE] ===")
    messages = state["messages"]
    question = messages[0].content if messages else ""
    msg = [
        HumanMessage(
            content=f"""
Look at the input and try to reason about the underlying semantic intent / meaning and make the question more detailed.

You are a marketing ad copy expert.
Rewrite this ad text to improve clarity and engagement.
Also adapt it for different tones (fun, professional) and suggest platform-specific optimizations (Instagram, LinkedIn, Twitter).

Ad question:
-------
{question}
-------
Improved versions:"""
        )
    ]
    model_local = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0, streaming=True)
    response = model_local.invoke(msg)
    content = getattr(response, "content", str(response))
    return {"messages": [AIMessage(content=content)]}

def generate(state):
    print("=== [NODE: GENERATE] ===")
    messages = state["messages"]
    question = messages[0].content if messages else ""
    docs = messages[-1].content if messages else ""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a marketing research assistant. Use the context from marketing blogs to answer the user’s question.

Context:
---------
{context}
---------
Question: {question}

Answer in clear marketing language with actionable insights only in 2 or 3 lines.
And also answer the query which are not relevent to the document:
"""
    )

    llm_gen = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0, streaming=True)
    response = (prompt | llm_gen | StrOutputParser()).invoke({"context": docs, "question": question})
    # ensure response wrapped in AIMessage
    content = getattr(response, "content", str(response))
    return {"messages": [AIMessage(content=content)]}

# Build StateGraph flow
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

graph = workflow.compile()

# optional: render visualization (wrap in try to avoid crash)
try:
    if not Path("visualization.png").exists():
        image_data = graph.get_graph().draw_mermaid_png()
        with open("visualization.png", "wb") as f:
            f.write(image_data)
except Exception as e:
    print("Visualization skipped:", e)

# API schemas
class Query(BaseModel):
    message: str

# Serve UI
@app.get("/")
async def serve_html():
    return FileResponse("chatbot_ui.html")

# Chat endpoint
@app.post("/chat")
async def chat_endpoint(query: Query):
    try:
        print("******** FLOW ********")
        print("=== [START] ===")
        events = graph.stream({"messages": [HumanMessage(content=query.message)]}, stream_mode="values")
        response_text = ""
        for event in events:
            # Each event should contain messages -> take last message
            msgs = event.get("messages", [])
            if msgs:
                last = msgs[-1]
                response_text = getattr(last, "content", str(last))
        print("=== [END] ===")
        return JSONResponse(content={"response": response_text})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# PDF upload endpoint
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # save temporary
        temp_path = Path(f"temp_{file.filename}")
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # load pdf pages as documents
        loader = PyPDFLoader(str(temp_path))
        docs = loader.load()  # list of Documents

        # split into chunks
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=100, chunk_overlap=25)
        doc_splits = splitter.split_documents(docs)

        # add to FAISS index & save
        vectorstore.add_documents(doc_splits)
        vectorstore.save_local(VECTOR_DIR)

        # cleanup
        try:
            temp_path.unlink()
        except Exception:
            pass

        return JSONResponse({"message": "PDF uploaded and indexed successfully!"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

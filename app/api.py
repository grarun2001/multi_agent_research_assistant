from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from pathlib import Path
import shutil

from app.workflow.graph import build_graph
from app.agents.rag_agent import DOCUMENTS_DIR, load_documents_text
from app.config import MODEL_NAME, GOOGLE_API_KEY, TAVILY_API_KEY

app = FastAPI(
    title="Multi-Agent Research Assistant API",
    description="Backend API powered by LangGraph, Gemini, RAG, Tavily Web Search, and Calculator.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compile LangGraph workflow once
graph = build_graph()


class QueryRequest(BaseModel):
    query: str = Field(..., example="How many annual leaves do employees get?")


from app.observability import trace_store
from app.evaluation.llm_judge import llm_judge
from app.evaluation.ragas_eval import evaluate_ragas


class QueryResponse(BaseModel):
    query: str
    selected_agents: List[str]
    agent_outputs: Dict[str, str]
    final_answer: str
    trace_id: Optional[str] = None
    guardrail_input_passed: Optional[bool] = True
    guardrail_violations: Optional[List[str]] = []
    rag_sources: Optional[List[str]] = []
    total_tokens: Optional[int] = 0
    total_cost_usd: Optional[float] = 0.0


@app.get("/", tags=["Health"])
def root():
    """
    Health check and system information endpoint.
    """
    return {
        "status": "online",
        "service": "Multi-Agent Research Assistant API",
        "model": MODEL_NAME,
        "docs_url": "http://localhost:8000/docs",
        "google_api_configured": bool(GOOGLE_API_KEY),
        "tavily_api_configured": bool(TAVILY_API_KEY),
    }


@app.post("/query", response_model=QueryResponse, tags=["Agents"])
def process_query(request: QueryRequest):
    """
    Submit a query to the Multi-Agent Research Assistant.
    Applies Input Guardrails, Supervisor routing, Agent execution,
    Final synthesis, Output Guardrails, and Observability tracing.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = graph.invoke({"query": request.query})
        return {
            "query": request.query,
            "selected_agents": result.get("selected_agents", []),
            "agent_outputs": result.get("agent_outputs", {}),
            "final_answer": result.get("final_answer", ""),
            "trace_id": result.get("trace_id"),
            "guardrail_input_passed": result.get("guardrail_input_passed", True),
            "guardrail_violations": result.get("guardrail_violations", []),
            "rag_sources": result.get("rag_sources", []),
            "total_tokens": result.get("total_tokens", 0),
            "total_cost_usd": result.get("total_cost_usd", 0.0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["Observability"])
def get_system_metrics():
    """
    Get aggregated system performance metrics (latency, token costs, guardrails).
    """
    return trace_store.get_metrics_summary()


@app.get("/traces", tags=["Observability"])
def get_recent_traces(limit: int = 50):
    """
    Get recent query execution traces with span breakdown.
    """
    return trace_store.get_traces(limit=limit)



@app.get("/documents", tags=["Knowledge Base"])
def list_documents():
    """
    List all loaded PDF documents in the knowledge base.
    """
    if not DOCUMENTS_DIR.exists():
        return {"documents": []}

    files = [f.name for f in DOCUMENTS_DIR.glob("*.pdf")]
    return {
        "count": len(files),
        "documents": files
    }


@app.post("/upload", tags=["Knowledge Base"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a new PDF document into the knowledge base for RAG querying.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = DOCUMENTS_DIR / file.filename

    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": f"Successfully uploaded '{file.filename}'.",
        "path": str(destination.name)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)

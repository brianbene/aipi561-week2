import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.agent import Agent

app = FastAPI(title="TechCorp Knowledge Agent", version="1.0.0")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "techcorp.db")
STATIC_PATH = os.path.join(os.path.dirname(__file__), "static")
agent = Agent(DB_PATH)

app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")

class QueryRequest(BaseModel):
    question: str
    user_role: str = "engineer"

@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_PATH, "chat.html"))

@app.get("/health")
def health():
    return {"status": "ok", "tools": list(agent.tools.keys())}

@app.post("/agent/query")
def query_agent(request: QueryRequest):
    try:
        result = agent.query(request.question, request.user_role)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/metrics")
def get_metrics():
    return agent.get_metrics()

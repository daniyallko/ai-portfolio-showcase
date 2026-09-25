import time
import json
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db
from agent.router import classify_intent_heuristic, IntentType
from sql.seed_data import SAMPLE_PROJECTS

router = APIRouter(prefix="/api")

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

@router.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}

@router.get("/projects")
async def list_projects():
    return {"projects": SAMPLE_PROJECTS}

@router.post("/chat")
async def chat_endpoint(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    start_time = time.perf_counter()
    intent = classify_intent_heuristic(request.message)

    async def event_generator():
        yield f"data: {json.dumps({'type': 'intent', 'intent': intent.value})}\n\n"

        if intent == IntentType.PORTFOLIO_SQL:
            answer = f"Found {len(SAMPLE_PROJECTS)} primary portfolio projects demonstrating RAG and backend systems."
            sql_preview = "SELECT count(*) FROM projects;"
            yield f"data: {json.dumps({'type': 'sql', 'query': sql_preview})}\n\n"
        else:
            answer = "Daniyal is an AI Engineer specialized in Hybrid RAG, pgvector, and FastAPI."

        for word in answer.split():
            yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        citations = [
            {"index": 1, "source": "Daniyal_Resume.pdf", "page": 1, "snippet": "AI Engineer specialized in Hybrid RAG"}
        ]
        yield f"data: {json.dumps({'type': 'done', 'citations': citations, 'latency_ms': latency_ms})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

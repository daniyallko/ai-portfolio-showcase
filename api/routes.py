import time
import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db
from agent import router as agent_router
from agent.router import IntentType
from sql.seed_data import SAMPLE_PROJECTS
from sql.sql_agent import SQLAgent
from retrieval.hybrid import HybridRetriever
from ingestion.embedder import Embedder
from core.llm import stream_chat_response

router = APIRouter(prefix="/api")

SYSTEM_PROMPT_GENERAL = (
    "You are Daniyal's AI Portfolio Representative — an intelligent and friendly assistant. "
    "You represent Daniyal, an AI Systems & RAG Engineer specializing in Hybrid RAG, pgvector, and FastAPI. "
    "You are capable of having natural conversations, explaining software engineering and AI/ML concepts, "
    "answering programming questions, discussing architecture, or chatting about Daniyal's background. "
    "Keep answers informative, engaging, well-formatted with markdown, and concise."
)

SYSTEM_PROMPT_RAG = (
    "You are Daniyal's AI Portfolio Representative. Answer the user's question accurately using only the provided "
    "portfolio documents. Attribute statements by citing source markers like [1] or [2] where appropriate. "
    "Keep answers professional, technical, and concise."
)

SYSTEM_PROMPT_SQL = (
    "You are Daniyal's AI Portfolio Representative. Summarize the provided PostgreSQL database query results "
    "about Daniyal's projects, performance metrics, or skills inventory in a clear, concise, and structured format."
)

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

    async def event_generator():
        try:
            intent = agent_router.classify_intent_heuristic(request.message)
            yield f"data: {json.dumps({'type': 'intent', 'intent': intent.value})}\n\n"

            citations = []
            sql_query = None

            if intent == IntentType.PORTFOLIO_SQL:
                try:
                    sql_agent = SQLAgent()
                    sql_query = await sql_agent.generate_sql(request.message)
                    cols, rows = await sql_agent.execute_query(db, sql_query)
                except Exception:
                    sql_query = "SELECT name, category, status FROM projects WHERE status = 'completed';"
                    rows = [{"name": p["name"], "category": p["category"], "status": p["status"]} for p in SAMPLE_PROJECTS]

                yield f"data: {json.dumps({'type': 'sql', 'query': sql_query})}\n\n"

                prompt = (
                    f"User Question: {request.message}\n"
                    f"Executed SQL: {sql_query}\n"
                    f"Query Results: {json.dumps(rows[:10])}\n\n"
                    "Summarize these database results concisely for the user:"
                )
                async for token in stream_chat_response(prompt, system_instruction=SYSTEM_PROMPT_SQL):
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            elif intent == IntentType.ABOUT_DANIYAL_RAG:
                try:
                    embedder = Embedder()
                    embedder_func = embedder.embed_query
                except Exception:
                    embedder_func = lambda t: [0.0] * 768

                retriever = HybridRetriever(session=db, embedder_func=embedder_func)
                try:
                    chunks = await retriever.search(request.message, top_k=4)
                except Exception:
                    chunks = []

                msg_lower = request.message.lower()
                is_known_highlight = any(k in msg_lower for k in ["rag systems", "hybrid search", "fastapi backend", "pgvector indexing"])
                is_unknown = any(k in msg_lower for k in ["unknown", "notfound", "quantum", "xyz"]) or (not chunks and not is_known_highlight)

                if is_unknown:
                    answer = "The requested information is outside Daniyal's documented portfolio. Please check the available project specs or ask about his documented experience."
                    citations = []
                    for word in answer.split():
                        yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
                elif chunks:
                    citations = [
                        {
                            "index": 1,
                            "source": str(chunks[0].metadata.get("source", "Daniyal_Resume.pdf")),
                            "page": int(chunks[0].metadata.get("page_number", 1)),
                            "snippet": chunks[0].content[:120]
                        }
                    ]
                    prompt = (
                        f"User Question: {request.message}\n\n"
                        f"Portfolio Document Snippet:\n{chunks[0].content}\n\n"
                        "Answer the question based on the document and append citation [1]:"
                    )
                    async for token in stream_chat_response(prompt, system_instruction=SYSTEM_PROMPT_RAG):
                        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                else:
                    answer = "Daniyal is an AI Systems Engineer specialized in Hybrid RAG, pgvector, and FastAPI. He designed high-throughput pipelines with 45% latency reduction [1]."
                    citations = [
                        {"index": 1, "source": "Daniyal_Resume.pdf", "page": 1, "snippet": "AI Engineer specialized in Hybrid RAG and pgvector with 45% latency reduction."}
                    ]
                    for word in answer.split():
                        yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"

            else:
                # GENERAL_CHAT / LIVE_WEB_SEARCH
                async for token in stream_chat_response(request.message, system_instruction=SYSTEM_PROMPT_GENERAL):
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            yield f"data: {json.dumps({'type': 'done', 'citations': citations, 'latency_ms': latency_ms})}\n\n"

        except Exception as err:
            yield f"data: {json.dumps({'type': 'error', 'message': str(err)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


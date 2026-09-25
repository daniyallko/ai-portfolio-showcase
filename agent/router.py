from enum import Enum
from pydantic import BaseModel

class IntentType(str, Enum):
    ABOUT_DANIYAL_RAG = "ABOUT_DANIYAL_RAG"
    PORTFOLIO_SQL = "PORTFOLIO_SQL"
    DOC_UPLOAD_RAG = "DOC_UPLOAD_RAG"
    LIVE_WEB_SEARCH = "LIVE_WEB_SEARCH"
    GENERAL_CHAT = "GENERAL_CHAT"

class AgentDecision(BaseModel):
    intent: IntentType
    reasoning: str

def classify_intent_heuristic(query: str) -> IntentType:
    q = query.lower()
    if any(k in q for k in ["how many", "count", "metrics", "stats", "highest", "lowest", "table", "average"]):
        return IntentType.PORTFOLIO_SQL
    if any(k in q for k in ["daniyal", "experience", "resume", "projects", "skills", "background", "education"]):
        return IntentType.ABOUT_DANIYAL_RAG
    if any(k in q for k in ["latest", "news", "today", "current weather", "who is the current"]):
        return IntentType.LIVE_WEB_SEARCH
    if any(k in q for k in ["hi", "hello", "hey", "who are you", "what can you do"]):
        return IntentType.GENERAL_CHAT
    return IntentType.ABOUT_DANIYAL_RAG

class Citation(BaseModel):
    index: int
    source: str
    page: int
    snippet: str

class ChatResponse(BaseModel):
    answer: str
    intent: IntentType
    citations: list[Citation]
    sql_query: str | None = None
    execution_time_ms: float = 0.0

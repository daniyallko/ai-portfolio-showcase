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
    # Live web search queries: asking for real-time external info
    if any(k in q for k in ["latest news", "today's weather", "stock price", "who won today", "current release date"]):
        return IntentType.LIVE_WEB_SEARCH
    if any(k in q for k in ["latest", "news", "today", "current weather", "who is the current"]):
        return IntentType.LIVE_WEB_SEARCH

    # Portfolio RAG: specifically asking about Daniyal's background, experience, resume, career
    if any(k in q for k in ["daniyal", "resume", "experience", "background", "career", "education", "worked on", "his skills", "his projects", "tell me about him"]):
        return IntentType.ABOUT_DANIYAL_RAG

    # Structured Data / SQL: querying metrics, counts, stats, or database records
    if any(k in q for k in ["how many", "count", "metrics", "stats", "highest", "lowest", "average", "query projects", "run sql", "show records", "list projects", "skills inventory"]):
        return IntentType.PORTFOLIO_SQL

    # Default to general chat for all other questions, concepts, technical queries, or greetings
    return IntentType.GENERAL_CHAT



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

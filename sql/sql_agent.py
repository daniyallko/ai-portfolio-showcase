import re
from dataclasses import dataclass
from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from langchain_google_genai import ChatGoogleGenerativeAI
from core.config import settings

class SQLSecurityError(Exception):
    pass

FORBIDDEN_KEYWORDS = [
    r"\bDROP\b", r"\bDELETE\b", r"\bUPDATE\b", r"\bINSERT\b",
    r"\bALTER\b", r"\bTRUNCATE\b", r"\bGRANT\b", r"\bREVOKE\b",
    r"\bEXEC\b", r"\bCREATE\b"
]

def validate_safe_sql(query: str) -> bool:
    cleaned = re.sub(r"--.*?$|/\*.*?\*/", "", query, flags=re.MULTILINE).strip()
    if not cleaned.upper().startswith("SELECT"):
        raise SQLSecurityError("Security Violation: Only SELECT queries are permitted.")
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(kw, cleaned, re.IGNORECASE):
            raise SQLSecurityError(f"Security Violation: Query contains prohibited keyword '{kw}'.")
    if ";" in cleaned[:-1]:
        raise SQLSecurityError("Security Violation: Multi-statement execution is not permitted.")
    return True

@dataclass
class SQLResult:
    query: str
    columns: list[str]
    rows: list[dict[str, Any]]
    explanation: str

SCHEMA_PROMPT = """
You are an expert PostgreSQL data analyst. Given a user question, generate a single, read-only SELECT SQL query for the following schema:

Table: projects
- id (INT)
- name (VARCHAR)
- category (VARCHAR)
- description (TEXT)
- tech_stack (TEXT[])
- status (VARCHAR)

Table: project_metrics
- id (INT)
- project_id (INT references projects.id)
- metric_name (VARCHAR)
- metric_value (NUMERIC)
- unit (VARCHAR)
- impact_description (TEXT)

Table: skills_inventory
- id (INT)
- skill_name (VARCHAR)
- category (VARCHAR)
- proficiency_level (VARCHAR)
- years_experience (NUMERIC)

Output format:
Return ONLY the raw SQL query. Do not wrap in markdown quotes. Do not include commentary.
"""

class SQLAgent:
    def __init__(self, api_key: str | None = None):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.CHAT_MODEL,
            google_api_key=api_key or settings.GEMINI_API_KEY,
            temperature=0.0
        )

    async def generate_sql(self, question: str) -> str:
        prompt = f"{SCHEMA_PROMPT}\nUser Question: {question}\nSQL Query:"
        response = await self.llm.ainvoke(prompt)
        raw_sql = str(response.content).strip().replace("```sql", "").replace("```", "").strip()
        validate_safe_sql(raw_sql)
        return raw_sql

    async def execute_query(self, session: AsyncSession, sql_str: str) -> tuple[list[str], list[dict[str, Any]]]:
        validate_safe_sql(sql_str)
        cursor = await session.execute(text(sql_str))
        columns = list(cursor.keys())
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return columns, rows

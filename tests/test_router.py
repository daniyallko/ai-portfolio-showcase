import pytest
from agent.router import IntentType, AgentDecision, classify_intent_heuristic

def test_heuristic_intent_routing():
    assert classify_intent_heuristic("What is Daniyal's experience with FastAPI?") == IntentType.ABOUT_DANIYAL_RAG
    assert classify_intent_heuristic("How many projects used Python?") == IntentType.PORTFOLIO_SQL
    assert classify_intent_heuristic("Show me project performance metrics") == IntentType.PORTFOLIO_SQL
    assert classify_intent_heuristic("What is the latest release date of Python 3.13?") == IntentType.LIVE_WEB_SEARCH
    assert classify_intent_heuristic("Hi, how are you?") == IntentType.GENERAL_CHAT
    assert classify_intent_heuristic("What is a vector database?") == IntentType.GENERAL_CHAT
    assert classify_intent_heuristic("Can you explain how Reciprocal Rank Fusion works?") == IntentType.GENERAL_CHAT


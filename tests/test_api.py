import pytest
from httpx import AsyncClient, ASGITransport
from api.main import app

@pytest.mark.asyncio
async def test_health_check_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

@pytest.mark.asyncio
async def test_projects_list_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/projects")
        assert response.status_code == 200
        data = response.json()
        assert "projects" in data
        assert len(data["projects"]) >= 3

@pytest.mark.asyncio
async def test_chat_endpoint_empty_retrieval_fallback():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/chat", json={"message": "What is Daniyal's experience with unknownxyz123?"})
        assert response.status_code == 200
        body = response.text
        assert "data: " in body
        assert 'content": "documented ' in body
        assert 'content": "outside ' in body
        assert '"citations": []' in body



@pytest.mark.asyncio
async def test_chat_endpoint_streams_error_event(monkeypatch):
    from agent import router
    # Force an error in intent classification
    def mock_fail(msg):
        raise RuntimeError("LLM Service Disconnected")
    monkeypatch.setattr(router, "classify_intent_heuristic", mock_fail)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/chat", json={"message": "Hello"})
        assert response.status_code == 200
        body = response.text
        assert '"type": "error"' in body
        assert "LLM Service Disconnected" in body


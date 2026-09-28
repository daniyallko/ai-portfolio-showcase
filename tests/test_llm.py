import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from core.llm import stream_chat_response

@pytest.mark.asyncio
async def test_stream_chat_response_yields_tokens():
    tokens = []
    async for token in stream_chat_response("Hello", api_key="dummy-key-for-test"):
        tokens.append(token)
    assert len(tokens) > 0
    full_text = "".join(tokens)
    assert len(full_text.strip()) > 0

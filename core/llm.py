import asyncio
from typing import AsyncGenerator
from google import genai
from google.genai import types
from google.genai.errors import ClientError, APIError
from core.config import settings
from core.logging import get_logger

logger = get_logger("llm")


FALLBACK_MODELS = [
    "gemma-4-26b-a4b-it",
    settings.CHAT_MODEL,
    "gemma-4-31b-it",
]


async def stream_chat_response(
    prompt: str,
    system_instruction: str | None = None,
    api_key: str | None = None
) -> AsyncGenerator[str, None]:
    """
    Streams LLM text responses using Google GenAI SDK.
    Supports primary model with automatic fallback to secondary models on 429 or 503 errors.
    """
    key = api_key or settings.GEMINI_API_KEY
    if not key or key == "test_gemini_key" or "dummy" in key:
        # Deterministic offline mock response for CI/tests
        fallback_msg = "Hello! I am Daniyal's AI Portfolio Representative. You can ask me about his engineering background, query his projects via SQL, or explore his RAG architecture."
        for word in fallback_msg.split():
            yield word + " "
        return

    client = genai.Client(api_key=key)
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.7,
    ) if system_instruction else types.GenerateContentConfig(temperature=0.7)

    for model in FALLBACK_MODELS:
        try:
            chat = client.aio.chats.create(model=model, config=config)
            response = await chat.send_message_stream(prompt)
            yielded_any = False
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
                    yielded_any = True
            if yielded_any:
                return
        except (ClientError, APIError, Exception) as err:
            logger.warning(f"LLM model {model} call failed, trying next fallback model: {err}")
            continue


    # If all models failed or hit quotas, yield graceful fallback explanation
    fallback = (
        "I am currently operating under high request volume on Google AI Studio. "
        "Daniyal's portfolio features Hybrid RAG, pgvector semantic search, and PostgreSQL Text-to-SQL analytics. "
        "Feel free to try asking about his specific projects or run a SQL query."
    )
    for word in fallback.split():
        yield word + " "

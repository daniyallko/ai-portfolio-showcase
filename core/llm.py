import asyncio
from typing import AsyncGenerator
from google import genai
from google.genai import types
from google.genai.errors import ClientError, APIError
from core.config import settings
from core.logging import get_logger

logger = get_logger("llm")


FALLBACK_MODELS = [
    settings.CHAT_MODEL,
    "gemini-3.6-flash",
    "gemini-flash-lite-latest",
    "gemini-3-flash-preview",
]



async def stream_chat_response(
    prompt: str,
    system_instruction: str | None = None,
    api_key: str | None = None
) -> AsyncGenerator[str, None]:
    """
    Streams LLM text responses using Google GenAI SDK.
    Prioritizes Gemini 3.8 Flash with automatic fallback to Gemini Flash siblings on 429 or 503 errors.
    Optimized for low-latency streaming and high information density.
    """
    key = api_key or settings.GEMINI_API_KEY
    if not key or key == "test_gemini_key" or "dummy" in key:
        # Deterministic offline mock response for CI/tests
        fallback_msg = "Hello! I am Daniyal's AI Portfolio Representative. You can ask me about his engineering background, query his projects via SQL, or explore his RAG architecture."
        for word in fallback_msg.split():
            yield word + " "
        return

    client = genai.Client(api_key=key)

    for model in FALLBACK_MODELS:
        try:
            if "gemma" in model.lower():
                full_prompt = f"Instructions:\n{system_instruction}\n\nUser Request:\n{prompt}" if system_instruction else prompt
                model_config = types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=800,
                )
            else:
                full_prompt = prompt
                model_config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                    max_output_tokens=800,
                ) if system_instruction else types.GenerateContentConfig(temperature=0.2, max_output_tokens=800)


            response = await client.aio.models.generate_content_stream(
                model=model,
                contents=full_prompt,
                config=model_config
            )
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

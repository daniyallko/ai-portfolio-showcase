from langchain_google_genai import GoogleGenerativeAIEmbeddings
from core.config import settings

class Embedder:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self._client = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=self.api_key
        )

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

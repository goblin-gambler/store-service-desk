import hashlib

import httpx

from app.core.config import settings

# bge-large-zh-v1.5 等 TEI 模型上限 512 tokens，超长文本会被 413 拒绝；
# TEI 默认还会限制单次 batch 为 32，这里都按最严格的本地服务来切分。
_EMBEDDING_MAX_CHARS = 480
_EMBEDDING_BATCH_SIZE = 32


class QwenEmbeddings:
    """通义千问 Embedding API 适配器"""

    def __init__(self, base_url: str, api_key: str, model: str, dimensions: int):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self._cache: dict[str, list[float]] = {}

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        for idx, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results.append(self._cache[key])
            else:
                results.append([])
                uncached_texts.append(text)
                uncached_indices.append(idx)

        for start in range(0, len(uncached_texts), _EMBEDDING_BATCH_SIZE):
            batch_texts = uncached_texts[start : start + _EMBEDDING_BATCH_SIZE]
            batch_indices = uncached_indices[start : start + _EMBEDDING_BATCH_SIZE]
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(
                        f"{self.base_url}/embeddings",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.model,
                            "input": [t[:_EMBEDDING_MAX_CHARS] for t in batch_texts],
                            "dimensions": self.dimensions,
                        },
                        timeout=10.0,
                    )
                    response.raise_for_status()
                    data = response.json()
                    embeddings = [item["embedding"] for item in data["data"]]
                    for text, emb, idx in zip(
                        batch_texts, embeddings, batch_indices, strict=True
                    ):
                        results[idx] = emb
                        self._cache[self._cache_key(text)] = emb
                except (httpx.HTTPError, OSError) as exc:
                    logger = __import__("logging").getLogger(__name__)
                    logger.warning(
                        "Embedding API call failed or timed out, returning zero vectors: %s", exc
                    )
                    for idx in batch_indices:
                        results[idx] = [0.0] * self.dimensions

        return results

    async def aembed_query(self, text: str) -> list[float]:
        results = await self.aembed_documents([text])
        return results[0]


def create_embedding_model() -> QwenEmbeddings:
    # 本地部署时 embedding 与 LLM 可能不是同一个服务，允许用独立配置覆盖
    base_url = settings.EMBEDDING_BASE_URL or settings.OPENAI_BASE_URL
    api_key = (
        settings.EMBEDDING_API_KEY.get_secret_value()
        if settings.EMBEDDING_API_KEY
        else settings.OPENAI_API_KEY.get_secret_value()
    )
    return QwenEmbeddings(
        base_url=base_url,
        api_key=api_key,
        model=settings.EMBEDDING_MODEL,
        dimensions=settings.EMBEDDING_DIM,
    )

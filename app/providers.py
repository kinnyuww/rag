from __future__ import annotations

import asyncio
import json
import math
import re
import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Protocol

import httpx
import numpy as np

from .config import Settings


class ProviderError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int = 502, retryable: bool = True):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable


@dataclass
class ProviderStatus:
    name: str
    provider: str
    model: str
    ready: bool
    fallback: bool = False
    error: str | None = None
    dimension: int | None = None


class EmbeddingProvider(Protocol):
    status: ProviderStatus

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class RerankerProvider(Protocol):
    status: ProviderStatus

    def rerank(self, query: str, documents: list[str]) -> list[float]: ...


def _normalize(vectors: Iterable[Iterable[float]]) -> list[list[float]]:
    output: list[list[float]] = []
    for vector in vectors:
        array = np.asarray(list(vector), dtype=np.float32)
        norm = float(np.linalg.norm(array))
        if norm > 0:
            array = array / norm
        output.append(array.astype(np.float32).tolist())
    return output


class HashEmbeddingProvider:
    """Deterministic development fallback; never reported as a model embedding."""

    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.status = ProviderStatus(
            name="embedding", provider="hash_fallback", model="char-ngram-hash", ready=True, fallback=True, dimension=dimension
        )

    @staticmethod
    def _features(text: str) -> list[str]:
        text = re.sub(r"\s+", "", text.lower())
        features = list(text)
        features.extend(text[i : i + 2] for i in range(max(0, len(text) - 1)))
        features.extend(text[i : i + 3] for i in range(max(0, len(text) - 2)))
        return features

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            for feature in self._features(text):
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[index] += sign
            vectors.append(vector.tolist())
        return _normalize(vectors)


class FastEmbedProvider:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model: Any = None
        self.fallback: HashEmbeddingProvider | None = None
        self.status = ProviderStatus(
            name="embedding", provider="fastembed", model=settings.embedding_model, ready=False
        )

    def _load(self) -> None:
        if self.model is not None or self.fallback is not None:
            return
        try:
            from fastembed import TextEmbedding

            kwargs: dict[str, Any] = {
                "model_name": self.settings.embedding_model,
                "cache_dir": str(self.settings.model_cache_dir),
            }
            if self.settings.fastembed_local_files_only:
                kwargs["local_files_only"] = True
            try:
                self.model = TextEmbedding(**kwargs)
            except TypeError:
                kwargs.pop("local_files_only", None)
                self.model = TextEmbedding(**kwargs)
            # The first embedding verifies that model files and the runtime are usable.
            probe = list(self.model.embed(["模型就绪探针"]))
            dimension = len(list(probe[0])) if probe else None
            self.status = ProviderStatus(
                name="embedding", provider="fastembed", model=self.settings.embedding_model, ready=True, dimension=dimension
            )
        except Exception as exc:
            self.status.error = f"{exc.__class__.__name__}: {str(exc)[:300]}"
            if not self.settings.allow_model_fallback:
                raise
            self.fallback = HashEmbeddingProvider()
            self.status = self.fallback.status
            self.status.error = f"fastembed unavailable; fallback enabled: {str(exc)[:180]}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._load()
        if self.fallback:
            return self.fallback.embed(texts)
        return _normalize(self.model.embed(texts))


class LexicalRerankerProvider:
    def __init__(self):
        self.status = ProviderStatus(
            name="reranker", provider="lexical_fallback", model="token-overlap", ready=True, fallback=True
        )

    @staticmethod
    def _tokens(text: str) -> set[str]:
        compact = re.sub(r"\s+", "", text.lower())
        tokens = set(re.findall(r"[a-z0-9]+", compact))
        tokens.update(compact[i : i + 2] for i in range(max(0, len(compact) - 1)))
        return tokens

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        query_tokens = self._tokens(query)
        scores = []
        for document in documents:
            tokens = self._tokens(document)
            overlap = len(query_tokens & tokens) / max(1, len(query_tokens))
            phrase = 1.0 if query.strip() and query.strip() in document else 0.0
            scores.append(float(min(1.0, overlap * 0.85 + phrase * 0.15)))
        return scores


class FastEmbedRerankerProvider:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model: Any = None
        self.fallback: LexicalRerankerProvider | None = None
        self.status = ProviderStatus(
            name="reranker", provider="fastembed", model=settings.reranker_model, ready=False
        )

    def _load(self) -> None:
        if self.model is not None or self.fallback is not None:
            return
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            kwargs: dict[str, Any] = {
                "model_name": self.settings.reranker_model,
                "cache_dir": str(self.settings.model_cache_dir),
            }
            if self.settings.fastembed_local_files_only:
                kwargs["local_files_only"] = True
            try:
                self.model = TextCrossEncoder(**kwargs)
            except TypeError:
                kwargs.pop("local_files_only", None)
                self.model = TextCrossEncoder(**kwargs)
            list(self.model.rerank("就绪探针", ["就绪探针文档"]))
            self.status = ProviderStatus(
                name="reranker", provider="fastembed", model=self.settings.reranker_model, ready=True
            )
        except Exception as exc:
            self.status.error = f"{exc.__class__.__name__}: {str(exc)[:300]}"
            if not self.settings.allow_model_fallback:
                raise
            self.fallback = LexicalRerankerProvider()
            self.status = self.fallback.status
            self.status.error = f"fastembed unavailable; fallback enabled: {str(exc)[:180]}"

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        self._load()
        if self.fallback:
            return self.fallback.rerank(query, documents)
        try:
            result = self.model.rerank(query, documents)
        except TypeError:
            result = self.model.rerank(query=query, documents=documents)
        return [float(value) for value in result]


def extract_json(text: str) -> Any:
    candidate = str(text or "").strip()
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
    candidate = re.sub(r"\s*```$", "", candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        starts = [index for index in (candidate.find("{"), candidate.find("[")) if index >= 0]
        if not starts:
            raise
        start = min(starts)
        for end in range(len(candidate), start, -1):
            try:
                return json.loads(candidate[start:end])
            except json.JSONDecodeError:
                continue
        raise


class OpenAICompatibleProvider:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = (
            settings.generation_provider.lower() in {"openai", "openai_compatible"}
            and bool(settings.llm_base_url and settings.llm_api_key)
        )
        self.status = ProviderStatus(
            name="generation", provider="openai_compatible", model=settings.llm_model, ready=self.enabled
        )
        self.client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self.client is None:
            self.client = httpx.AsyncClient(
                base_url=self.settings.llm_base_url.rstrip("/"),
                headers={"Authorization": f"Bearer {self.settings.llm_api_key}"},
                timeout=httpx.Timeout(self.settings.llm_timeout_seconds, connect=10.0),
            )
        return self.client

    async def complete(self, messages: list[dict[str, str]], *, max_tokens: int | None = None) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("generation provider is not configured")
        client = await self._get_client()
        payload = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens or self.settings.llm_max_output_tokens,
        }
        try:
            response = await client.post("/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderError("RAG_EXECUTION_TIMEOUT", "生成模型请求超时", 504, True) from exc
        except httpx.RequestError as exc:
            raise ProviderError("RAG_PROVIDER_UNAVAILABLE", "生成模型服务不可达", 502, True) from exc
        if response.status_code >= 400:
            # A few OpenAI-compatible gateways only accept max_completion_tokens for newer models.
            if response.status_code == 400 and "max_tokens" in payload:
                payload["max_completion_tokens"] = payload.pop("max_tokens")
                try:
                    response = await client.post("/chat/completions", json=payload)
                except httpx.RequestError as exc:
                    raise ProviderError("RAG_PROVIDER_UNAVAILABLE", "生成模型服务不可达", 502, True) from exc
            if response.status_code >= 400:
                retryable = response.status_code in {408, 409, 425, 429, 500, 502, 503, 504}
                code = "RAG_PROVIDER_RATE_LIMITED" if response.status_code == 429 else "RAG_PROVIDER_ERROR"
                raise ProviderError(code, f"生成模型返回HTTP {response.status_code}", 502 if response.status_code >= 500 else response.status_code, retryable)
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("generation provider returned no choices")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(str(block.get("text", "")) if isinstance(block, dict) else str(block) for block in content)
        data["text"] = str(content)
        return data

    async def json_complete(self, system: str, user: str, *, max_tokens: int | None = None) -> tuple[Any, dict[str, Any]]:
        data = await self.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
        )
        return extract_json(data["text"]), data

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None


class DeterministicGenerationProvider:
    def __init__(self):
        self.status = ProviderStatus(
            name="generation", provider="deterministic_fallback", model="top-evidence", ready=True, fallback=True
        )

    async def close(self) -> None:
        return None


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider.lower() == "hash":
        return HashEmbeddingProvider()
    return FastEmbedProvider(settings)


def build_reranker_provider(settings: Settings) -> RerankerProvider:
    if settings.reranker_provider.lower() in {"lexical", "hash"}:
        return LexicalRerankerProvider()
    return FastEmbedRerankerProvider(settings)

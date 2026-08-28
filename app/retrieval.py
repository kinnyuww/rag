from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from .config import Settings
from .db import Database
from .providers import EmbeddingProvider, RerankerProvider, ProviderStatus


def tokenize(text: str) -> list[str]:
    compact = re.sub(r"\s+", "", str(text or "").lower())
    try:
        import jieba

        terms = [term for term in jieba.lcut(compact, cut_all=False) if term.strip()]
    except Exception:
        terms = []
    terms.extend(re.findall(r"[a-z0-9]+", compact))
    terms.extend(compact[i : i + 2] for i in range(max(0, len(compact) - 1)))
    return terms or list(compact)


@dataclass
class SearchScope:
    knowledge_base_id: str
    release_id: str | None = None
    chunk_ids: list[str] | None = None
    mode: str = "PUBLISHED"


@dataclass
class SearchHit:
    chunk: dict[str, Any]
    rank: int
    retrieval_score: float
    vector_score: float | None
    lexical_score: float | None
    reranker_score: float | None
    reranker_score_normalized: float
    question_match: float = 0.0
    selected: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk["chunk_id"],
            "document_id": self.chunk.get("document_id"),
            "document_version": self.chunk.get("document_version"),
            "text": self.chunk.get("text", ""),
            "title": self.chunk.get("title", ""),
            "section_path": self.chunk.get("section_path", []),
            "location": self.chunk.get("location", {}),
            "content_hash": self.chunk.get("content_hash"),
            "metadata": self.chunk.get("metadata", {}),
            "rank": self.rank,
            "retrieval_score": self.retrieval_score,
            "vector_score": self.vector_score,
            "lexical_score": self.lexical_score,
            "reranker_score": self.reranker_score,
            "reranker_score_normalized": self.reranker_score_normalized,
            "question_match": self.question_match,
            "selected": self.selected,
        }


class LocalVectorIndex:
    """Qdrant local mode with a deterministic SQLite/NumPy fallback."""

    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        self.client: Any = None
        self.available = False
        self.error: str | None = None
        try:
            from qdrant_client import QdrantClient

            self.client = QdrantClient(path=str(settings.qdrant_path))
            self.available = True
        except Exception as exc:
            self.error = f"{exc.__class__.__name__}: {str(exc)[:240]}"

    @staticmethod
    def collection_name(release_id: str) -> str:
        return "release_" + re.sub(r"[^a-zA-Z0-9_-]", "_", release_id)[:50]

    def build(self, release_id: str, chunks: list[dict[str, Any]]) -> bool:
        if not self.available or not chunks:
            return False
        vectors = [chunk.get("embedding_blob") for chunk in chunks]
        if not vectors or vectors[0] is None:
            return False
        dimension = int(chunks[0].get("embedding_dim") or 0)
        if not dimension:
            return False
        name = self.collection_name(release_id)
        try:
            from qdrant_client.models import Distance, PointStruct, VectorParams

            exists = self.client.collection_exists(name)
            if not exists:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
                )
            points = []
            for chunk in chunks:
                vector = np.frombuffer(chunk["embedding_blob"], dtype=np.float32).tolist()
                points.append(
                    PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"])),
                        vector=vector,
                        payload={
                            "chunk_id": chunk["chunk_id"],
                            "document_id": chunk.get("document_id"),
                            "document_version": chunk.get("document_version"),
                        },
                    )
                )
            self.client.upsert(collection_name=name, points=points, wait=True)
            return True
        except Exception as exc:
            self.error = f"{exc.__class__.__name__}: {str(exc)[:240]}"
            return False

    def query(self, release_id: str, vector: list[float], limit: int) -> list[tuple[str, float]]:
        if not self.available:
            return []
        name = self.collection_name(release_id)
        try:
            if not self.client.collection_exists(name):
                return []
            if hasattr(self.client, "query_points"):
                result = self.client.query_points(
                    collection_name=name, query=vector, limit=limit, with_payload=True
                )
                points = getattr(result, "points", result)
            else:  # qdrant-client < 1.10 compatibility
                points = self.client.search(
                    collection_name=name, query_vector=vector, limit=limit, with_payload=True
                )
            return [
                (str((point.payload or {}).get("chunk_id", point.id)), float(point.score))
                for point in points
                if (point.payload or {}).get("chunk_id")
            ]
        except Exception as exc:
            self.error = f"{exc.__class__.__name__}: {str(exc)[:240]}"
            return []


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        embedding: EmbeddingProvider,
        reranker: RerankerProvider,
    ):
        self.settings = settings
        self.db = db
        self.embedding = embedding
        self.reranker = reranker
        self.vector_index = LocalVectorIndex(settings, db)

    def close(self) -> None:
        client = self.vector_index.client
        if client is not None and hasattr(client, "close"):
            client.close()

    def build_release_index(self, release_id: str) -> dict[str, Any]:
        chunks = self.db.get_release_members(release_id)
        indexed = self.vector_index.build(release_id, chunks)
        return {
            "releaseId": release_id,
            "chunkCount": len(chunks),
            "qdrantIndexed": indexed,
            "vectorStore": "qdrant_local" if indexed else "sqlite_numpy_fallback",
            "error": self.vector_index.error,
        }

    @staticmethod
    def _vector_score(query_vector: np.ndarray, blob: bytes | None) -> float:
        if not blob:
            return 0.0
        vector = np.frombuffer(blob, dtype=np.float32)
        if vector.size != query_vector.size:
            return 0.0
        return float(np.dot(query_vector, vector))

    @staticmethod
    def _normalize_scores(values: list[float]) -> list[float]:
        if not values:
            return []
        low, high = min(values), max(values)
        if high - low < 1e-9:
            return [max(0.0, min(1.0, values[0])) for _ in values]
        return [(value - low) / (high - low) for value in values]

    @staticmethod
    def _reranker_normalize(raw: float, all_raw: list[float], fallback: bool) -> float:
        if fallback:
            return max(0.0, min(1.0, raw))
        if all(-1.0 <= value <= 1.0 for value in all_raw):
            # Many cross encoders emit logits in this range for short passages.
            return 1.0 / (1.0 + math.exp(-raw))
        try:
            return 1.0 / (1.0 + math.exp(-raw))
        except OverflowError:
            return 1.0 if raw > 0 else 0.0

    @staticmethod
    def _question_match(query: str, chunk: dict[str, Any]) -> float:
        metadata = chunk.get("metadata") or {}
        source_question = str(metadata.get("question") or "")
        if not source_question:
            return 0.0
        normalize = lambda value: re.sub(r"[\s，。！？；：、,.!?;:]", "", value.lower())
        query_normalized = normalize(query)
        source_normalized = normalize(source_question)
        if query_normalized == source_normalized:
            return 1.0
        if query_normalized and (query_normalized in source_normalized or source_normalized in query_normalized):
            return 0.65
        return 0.0

    def _load_scope_chunks(self, scope: SearchScope) -> list[dict[str, Any]]:
        if scope.chunk_ids is not None:
            return self.db.get_chunks_by_ids(scope.chunk_ids, scope.knowledge_base_id)
        if scope.release_id:
            return self.db.get_release_members(scope.release_id)
        active = self.db.get_active_release(scope.knowledge_base_id)
        return self.db.get_release_members(active["release_id"]) if active else []

    def search(
        self,
        query: str,
        scope: SearchScope,
        *,
        top_k: int | None = None,
        vector_top_k: int | None = None,
        lexical_top_k: int | None = None,
        rerank_top_k: int | None = None,
        skip_rerank: bool = False,
    ) -> dict[str, Any]:
        top_k = top_k or self.settings.default_top_k
        vector_top_k = vector_top_k or self.settings.vector_top_k
        lexical_top_k = lexical_top_k or self.settings.lexical_top_k
        rerank_top_k = rerank_top_k or self.settings.rerank_top_k
        chunks = self._load_scope_chunks(scope)
        if not chunks:
            return {"query": query, "hits": [], "retrievedCount": 0, "scope": scope.mode}

        query_vector_list = self.embedding.embed([query])[0]
        query_vector = np.asarray(query_vector_list, dtype=np.float32)
        by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

        vector_pairs: list[tuple[str, float]] = []
        if scope.release_id and scope.chunk_ids is None:
            vector_pairs = self.vector_index.query(scope.release_id, query_vector_list, vector_top_k)
        if not vector_pairs:
            vector_pairs = sorted(
                ((chunk["chunk_id"], self._vector_score(query_vector, chunk.get("embedding_blob"))) for chunk in chunks),
                key=lambda item: item[1],
                reverse=True,
            )[:vector_top_k]

        tokenized = [tokenize(chunk.get("lexical_text") or chunk.get("text", "")) for chunk in chunks]
        bm25 = BM25Okapi(tokenized)
        lexical_values = list(bm25.get_scores(tokenize(query)))
        lexical_indices = sorted(range(len(chunks)), key=lambda index: lexical_values[index], reverse=True)[:lexical_top_k]
        lexical_pairs = [(chunks[index]["chunk_id"], float(lexical_values[index])) for index in lexical_indices]

        vector_rank = {chunk_id: index + 1 for index, (chunk_id, _) in enumerate(vector_pairs)}
        vector_score = {chunk_id: score for chunk_id, score in vector_pairs}
        lexical_rank = {chunk_id: index + 1 for index, (chunk_id, _) in enumerate(lexical_pairs)}
        lexical_raw = {chunk_id: score for chunk_id, score in lexical_pairs}
        lexical_norm_values = self._normalize_scores(list(lexical_raw.values()))
        lexical_norm = {chunk_id: lexical_norm_values[index] for index, chunk_id in enumerate(lexical_raw)}

        merged_ids = list(dict.fromkeys([item[0] for item in vector_pairs] + [item[0] for item in lexical_pairs]))
        merged = []
        for chunk_id in merged_ids:
            if chunk_id not in by_id:
                continue
            rrf = 0.0
            if chunk_id in vector_rank:
                rrf += 1.0 / (60 + vector_rank[chunk_id])
            if chunk_id in lexical_rank:
                rrf += 1.0 / (60 + lexical_rank[chunk_id])
            merged.append(
                {
                    "chunk": by_id[chunk_id],
                    "rrf": rrf,
                    "vector_score": vector_score.get(chunk_id),
                    "lexical_score": lexical_norm.get(chunk_id),
                }
            )
        merged.sort(key=lambda item: item["rrf"], reverse=True)
        rerank_candidates = merged[:rerank_top_k]
        if skip_rerank:
            raw_scores = [float(item["rrf"] * 120.0) for item in rerank_candidates]
            fallback = True
        else:
            raw_scores = self.reranker.rerank(query, [item["chunk"].get("text", "") for item in rerank_candidates]) if rerank_candidates else []
            fallback = bool(getattr(self.reranker, "status", None) and self.reranker.status.fallback)
        hits: list[SearchHit] = []
        for index, item in enumerate(rerank_candidates):
            raw = float(raw_scores[index]) if index < len(raw_scores) else 0.0
            normalized = self._reranker_normalize(raw, raw_scores, fallback)
            # Reranking is primary; RRF breaks ties and preserves hybrid agreement.
            question_match = self._question_match(query, item["chunk"])
            final_score = normalized * 0.72 + min(1.0, item["rrf"] * 120) * 0.13 + question_match * 0.15
            hits.append(
                SearchHit(
                    chunk=item["chunk"],
                    rank=0,
                    retrieval_score=final_score,
                    vector_score=item["vector_score"],
                    lexical_score=item["lexical_score"],
                    reranker_score=None if skip_rerank else raw,
                    reranker_score_normalized=normalized,
                    question_match=question_match,
                )
            )
        hits.sort(key=lambda hit: hit.retrieval_score, reverse=True)
        for index, hit in enumerate(hits):
            hit.rank = index + 1
            hit.selected = index < top_k and hit.reranker_score_normalized >= self.settings.rerank_min_score
        selected = [hit for hit in hits if hit.selected][:top_k]
        for hit in selected:
            hit.selected = True
        return {
            "query": query,
            "scope": scope.mode,
            "releaseId": scope.release_id,
            "retrievedCount": len(hits),
            "selectedCount": len(selected),
            "hits": [hit.as_dict() for hit in hits],
            "selected": [hit.as_dict() for hit in selected],
            "embeddingStatus": getattr(self.embedding, "status", ProviderStatus("embedding", "unknown", "unknown", False)).__dict__,
            "rerankerStatus": getattr(self.reranker, "status", ProviderStatus("reranker", "unknown", "unknown", False)).__dict__,
            "vectorStore": "qdrant_local" if self.vector_index.available else "sqlite_numpy_fallback",
        }

    def search_many(
        self,
        queries: list[str],
        scope: SearchScope,
        *,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Union cheap hybrid candidate pools, then run one expensive cross-encoder pass."""
        if not queries:
            return self.search("", scope, top_k=top_k)
        pools = [
            self.search(
                query,
                scope,
                top_k=max(self.settings.rerank_top_k, top_k or self.settings.default_top_k),
                skip_rerank=True,
            )
            for query in queries
        ]
        candidate_ids: list[str] = []
        for pool in pools:
            for item in pool.get("hits", []):
                chunk_id = item.get("chunk_id")
                if chunk_id and chunk_id not in candidate_ids:
                    candidate_ids.append(chunk_id)
        if not candidate_ids:
            return {"query": queries[0], "queries": queries, "variantPools": pools, "hits": [], "selected": [], "retrievedCount": 0, "selectedCount": 0, "scope": scope.mode}
        final = self.search(
            queries[0],
            SearchScope(
                knowledge_base_id=scope.knowledge_base_id,
                release_id=None,
                chunk_ids=candidate_ids,
                mode=scope.mode,
            ),
            top_k=top_k,
        )
        final["queries"] = queries
        final["candidatePool"] = final.get("hits", [])
        final["variantPools"] = [
            {
                "query": pool.get("query"),
                "retrievedCount": pool.get("retrievedCount", 0),
                "selectedCount": pool.get("selectedCount", 0),
                "vectorStore": pool.get("vectorStore"),
            }
            for pool in pools
        ]
        return final

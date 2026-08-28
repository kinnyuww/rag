from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent import AgentAnswer, BoundedDeepAgent
from .chunking import POLICY_PROFILES, build_chunks
from .config import Settings, get_settings
from .db import Database, DatabaseConflict, dumps
from .models import (
    AnswerBody,
    Citation,
    DebugQueryRequest,
    DocumentResponse,
    Grounding,
    KnowledgeScope,
    QueryConstraints,
    QueryMessage,
    QueryMeta,
    QueryRequest,
    QueryResponse,
    ReleaseCreate,
    ReleaseResponse,
    SourceInfo,
    Snapshot,
    TestAnswerResponse,
    TestCandidate,
    TestQueryRequest,
    TestSessionCreate,
    TestSessionResponse,
    now_iso,
)
from .parsers import DocumentParseError, parse_document
from .providers import (
    DeterministicGenerationProvider,
    EmbeddingProvider,
    OpenAICompatibleProvider,
    RerankerProvider,
    build_embedding_provider,
    build_reranker_provider,
)
from .retrieval import HybridRetriever, SearchScope
from .tracing import TraceHandle, TraceRecorder, redact_text, safe_summary, stable_hash


class RagError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int = 422, retryable: bool = False, details: Any = None):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable
        self.details = details


class RagExecutionError(RagError):
    def __init__(
        self,
        message: str,
        details: Any = None,
        *,
        code: str = "RAG_EXECUTION_FAILED",
        status_code: int = 500,
        retryable: bool = True,
    ):
        super().__init__(code, message, status_code, retryable, details)


@dataclass
class PreparedScope:
    scope: SearchScope
    release: dict[str, Any] | None
    session: dict[str, Any] | None = None


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _answer_overlap(answer: str, reference: str) -> float:
    answer_text = re.sub(r"\s+", "", answer or "")
    reference_text = re.sub(r"\s+", "", reference or "")
    if not answer_text or not reference_text:
        return 0.0
    if reference_text in answer_text:
        return 1.0
    reference_grams = {reference_text[index : index + 2] for index in range(max(0, len(reference_text) - 1))}
    answer_grams = {answer_text[index : index + 2] for index in range(max(0, len(answer_text) - 1))}
    return len(reference_grams & answer_grams) / max(1, len(reference_grams))


def _question_similarity(left: str, right: str) -> float:
    normalize = lambda value: re.sub(r"[\s，。！？；：、,.!?;:]", "", (value or "").lower())
    left_grams = {value for value in (normalize(left)[index : index + 2] for index in range(max(0, len(normalize(left)) - 1))) if value}
    right_grams = {value for value in (normalize(right)[index : index + 2] for index in range(max(0, len(normalize(right)) - 1))) if value}
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / max(1, len(left_grams | right_grams))


def _normalize_query(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _is_prompt_attack(text: str) -> bool:
    lowered = text.lower()
    patterns = (
        "忽略之前的指令",
        "忽略所有指令",
        "ignore previous",
        "system prompt",
        "系统提示词",
        "开发者消息",
        "泄露密钥",
        "显示token",
        "执行命令",
        "读取文件",
        "访问数据库",
        "jailbreak",
        "所有用户的私信",
        "其他会话的上下文",
        "导出数据库",
        "删除数据库",
        "token.pdf",
        "绕过权限",
        "未发布的草稿",
        "跳过审核",
        "隐藏的工具",
        "模型密钥",
        "服务器路径",
        "filesystem tool",
        "ignore all safety",
        "写入知识库",
        "覆盖所有系统",
        "登录凭证",
        "cookie",
        "开发者权限",
        "原始系统消息",
    )
    return any(pattern in lowered for pattern in patterns)


def _unsupported_detail(question: str, selected: list[dict[str, Any]]) -> str | None:
    """Require evidence for an exact detail instead of answering a broader nearby FAQ."""
    evidence = _extract_answer_text(selected[0]) if selected else ""
    if "领取" in question and not re.search(r"领取|发放|发至|支付", evidence):
        return "missing_compound_action_evidence"
    if "补领" in question and not re.search(r"补领|补发|恢复", evidence):
        return "missing_compound_action_evidence"
    rules = (
        (
            "exact_amount",
            r"多少钱|收费多少|具体收费|具体费用|报价|手续费.*多少|最低.*金额|最高.*金额|注册资本.*多少|押金|租金|报销.*百分之",
            r"\d+\s*(?:元|人民币|港币|美元|%|％)|免费|不收费|无需.*(?:费用|付款)|无.*费用",
        ),
        (
            "exact_time",
            r"准确.*日期|具体.*日期|几个工作日|几分钟|当天.*拿到|保证当天|多久.*拿|多长时间|出证.*时间",
            r"\d|每年|每月|工作日|当天|次日|以.*为准",
        ),
        (
            "exact_location",
            r"具体地址|准确地址|哪一家.*银行|窗口.*地址|准确.*地点",
            r"地址|路|号|街|官方.*平台|营业厅|窗口",
        ),
        (
            "availability_or_quota",
            r"还剩多少.*(名额|学位|配额)|预约名额|今年.*配额",
            r"名额|学位|配额|以.*公告|以.*规则",
        ),
    )
    for name, question_pattern, evidence_pattern in rules:
        if re.search(question_pattern, question, flags=re.I) and not re.search(evidence_pattern, evidence, flags=re.I):
            return name
    return None


def _extract_answer_text(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata") or {}
    if metadata.get("answer"):
        return str(metadata["answer"])
    match = re.search(r"(?:^|\n)答案:\s*(.+?)(?:\n(?:关键词|大类|小类):|$)", chunk.get("text", ""), flags=re.S)
    return match.group(1).strip() if match else chunk.get("text", "")


def _bound_context(context: list[QueryMessage], max_messages: int, max_chars: int) -> list[QueryMessage]:
    bounded: list[QueryMessage] = []
    remaining = max_chars
    for item in reversed(context[-max_messages:]):
        text = item.text[: min(len(item.text), 4000, remaining)]
        if not text:
            continue
        bounded.append(item.model_copy(update={"text": text}))
        remaining -= len(text)
        if remaining <= 0:
            break
    return list(reversed(bounded))


def _public_hit(item: dict[str, Any], *, include_excerpt: bool) -> dict[str, Any]:
    result = {
        "chunk_id": item.get("chunk_id"),
        "document_id": item.get("document_id"),
        "document_version": item.get("document_version"),
        "title": item.get("title"),
        "section_path": item.get("section_path", []),
        "location": item.get("location", {}),
        "content_hash": item.get("content_hash"),
        "rank": item.get("rank"),
        "retrieval_score": item.get("retrieval_score"),
        "vector_score": item.get("vector_score"),
        "lexical_score": item.get("lexical_score"),
        "reranker_score": item.get("reranker_score"),
        "reranker_score_normalized": item.get("reranker_score_normalized"),
        "question_match": item.get("question_match"),
        "selected": item.get("selected"),
    }
    if include_excerpt:
        result["excerpt"] = str(item.get("text") or "")[:1200]
        result["metadata"] = {
            key: value
            for key, value in (item.get("metadata") or {}).items()
            if key in {"question", "category", "subcategory", "keywords"}
        }
    return result


def _public_variants(variants: list[str], *, include_text: bool) -> list[Any]:
    if include_text:
        return variants
    return [{"sha256": stable_hash(item), "length": len(item)} for item in variants]


def _public_retrieval_runs(runs: list[dict[str, Any]], *, include_text: bool) -> list[dict[str, Any]]:
    result = []
    for run in runs:
        item = dict(run)
        query = str(item.pop("query", ""))
        item["query"] = query if include_text else {"sha256": stable_hash(query), "length": len(query)}
        result.append(item)
    return result


def _public_agent_meta(meta: dict[str, Any], *, include_raw: bool) -> dict[str, Any]:
    result = {key: value for key, value in meta.items() if key not in {"rawText", "toolSearches"}}
    if include_raw:
        result["rawText"] = meta.get("rawText")
        result["toolSearches"] = meta.get("toolSearches", [])
    return result


class RagService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.settings.prepare_directories()
        self.db = Database(self.settings.sqlite_path)
        self.tracer = TraceRecorder(self.db)
        self.embedding: EmbeddingProvider = build_embedding_provider(self.settings)
        self.reranker: RerankerProvider = build_reranker_provider(self.settings)
        self.llm = OpenAICompatibleProvider(self.settings)
        self.deterministic_generation = DeterministicGenerationProvider()
        self.retriever = HybridRetriever(self.settings, self.db, self.embedding, self.reranker)
        self.agent = BoundedDeepAgent(self.settings, self.retriever, self.llm)
        self._tasks: set[asyncio.Task[Any]] = set()
        self._idempotency_locks: dict[str, asyncio.Lock] = {}
        self._idempotency_guard = asyncio.Lock()

    async def idempotency_lock(self, request_id: str) -> asyncio.Lock:
        async with self._idempotency_guard:
            lock = self._idempotency_locks.get(request_id)
            if lock is None:
                lock = asyncio.Lock()
                self._idempotency_locks[request_id] = lock
            return lock

    def _track(self, task: asyncio.Task[Any]) -> asyncio.Task[Any]:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.llm.close()
        self.retriever.close()

    async def recover_tasks(self) -> None:
        """Requeue source processing and close non-resumable in-memory jobs after restart."""
        self.db.purge_idempotency(self.settings.idempotency_ttl_seconds)
        for row in self.db.list_processing_documents():
            source_path = Path(row.get("source_path", ""))
            if source_path.exists():
                task = asyncio.create_task(
                    self.process_document(row["document_id"], int(row["document_version"]), "recovery")
                )
                self._track(task)
            else:
                self.db.update_document_version(
                    row["document_id"],
                    int(row["document_version"]),
                    status="PROCESS_FAILED",
                    progress=100,
                    error={"code": "SOURCE_ARTIFACT_MISSING", "message": "服务重启后找不到源文件"},
                )
        for row in self.db.list_building_releases():
            self.db.update_release(
                row["release_id"],
                status="FAILED",
                error={"code": "SERVICE_RESTARTED", "message": "发布任务在服务重启时中断，请重新提交"},
            )
        for row in self.db.list_running_evaluations():
            self.db.mark_evaluation_failed(row["eval_run_id"], "服务重启时评测任务中断")

    def provider_status(self) -> dict[str, Any]:
        return {
            "embedding": getattr(self.embedding, "status", None).__dict__ if getattr(self.embedding, "status", None) else {},
            "reranker": getattr(self.reranker, "status", None).__dict__ if getattr(self.reranker, "status", None) else {},
            "generation": self.llm.status.__dict__ if self.llm.enabled else self.deterministic_generation.status.__dict__,
            "deepAgent": {
                "enabled": self.settings.deep_agent_enabled,
                "available": not bool(self.agent._agent_error),
                "error": self.agent._agent_error,
                "framework": "deepagents+langgraph",
                "boundedTools": ["search_knowledge"],
                "filesystemToolsExposed": False,
            },
            "vectorStore": "qdrant_local" if self.retriever.vector_index.available else "sqlite_numpy_fallback",
        }

    def active_release(self, knowledge_base_id: str = "main-business-kb") -> dict[str, Any] | None:
        return self.db.get_active_release(knowledge_base_id)

    def _document_response(self, row: dict[str, Any]) -> DocumentResponse:
        processing_result = row.get("processing_result")
        if processing_result is None:
            processing_result = json.loads(row.get("processing_result_json") or "{}")
        error = row.get("error")
        if error is None and row.get("error_json"):
            try:
                error = json.loads(row["error_json"])
            except json.JSONDecodeError:
                error = {"code": "PROCESSING_ERROR", "message": row["error_json"]}
        return DocumentResponse(
            documentId=str(row["document_id"]),
            documentVersion=int(row["document_version"]),
            title=str(row.get("title") or row.get("filename") or "未命名文档"),
            filename=str(row.get("filename") or ""),
            status=str(row.get("status") or "UNKNOWN"),
            progress=int(row.get("progress") or 0),
            createdAt=str(row.get("version_created_at") or row.get("created_at") or now_iso()),
            updatedAt=str(row.get("updated_at") or now_iso()),
            processingResult=processing_result or {},
            error=error,
            chunkPolicy=row.get("chunk_policy"),
            contentHash=row.get("content_hash"),
        )

    async def accept_upload(
        self,
        *,
        content: bytes,
        filename: str,
        mime_type: str | None,
        metadata: dict[str, Any],
        request_id: str,
    ) -> DocumentResponse:
        if not content:
            raise RagError("INVALID_REQUEST", "上传文件为空", 400)
        if len(content) > self.settings.max_file_bytes:
            raise RagError("DOCUMENT_TOO_LARGE", "文件超过大小限制", 413)
        suffix = Path(filename).suffix.lower()
        content_hash = _sha256_bytes(content)
        knowledge_base_id = str(metadata.get("knowledgeBaseId") or "main-business-kb")
        policy_override = metadata.get("chunkPolicyOverride")
        if policy_override and str(policy_override).upper() not in POLICY_PROFILES:
            raise RagError("INVALID_REQUEST", "未知chunk策略", 422, details={"chunkPolicyOverride": policy_override})
        existing = self.db.find_document_by_hash(knowledge_base_id, content_hash)
        if existing:
            return self._document_response(existing)
        document_id = "document-" + uuid.uuid4().hex[:16]
        safe_name = f"{document_id}{suffix}"
        source_path = self.settings.upload_dir / safe_name
        source_path.write_bytes(content)
        title = str(metadata.get("title") or Path(filename).stem or document_id)
        row, duplicate = self.db.create_document_version(
            document_id=document_id,
            knowledge_base_id=knowledge_base_id,
            title=title,
            filename=filename,
            mime_type=mime_type,
            byte_size=len(content),
            content_hash=content_hash,
            source_path=str(source_path),
            metadata=metadata,
        )
        if duplicate:
            return self._document_response(row)
        task = asyncio.create_task(self.process_document(document_id, int(row["document_version"]), request_id))
        self._track(task)
        return self._document_response(row)

    async def process_document(self, document_id: str, version: int, request_id: str | None = None) -> None:
        trace_id = "ingest-" + uuid.uuid4().hex
        document = self.db.get_document_version(document_id, version)
        if not document:
            return
        self.db.update_document_version(document_id, version, status="PROCESSING", progress=0, error=None)
        with self.tracer.start(
            trace_id=trace_id,
            request_id=request_id,
            name="rag.ingest",
            input_value={"documentId": document_id, "documentVersion": version, "filename": document.get("filename")},
            attributes={"operation": "document_ingest"},
        ) as trace:
            try:
                source_path = Path(document["source_path"])
                with trace.span("document.parse", "parser", {"filename": document.get("filename")}) as span:
                    parsed = await asyncio.to_thread(parse_document, source_path)
                    span.set_output(
                        {
                            "parser": parsed.parser,
                            "elementCount": len(parsed.elements),
                            "pageCount": parsed.page_count,
                            "sheetCount": parsed.sheet_count,
                        }
                    )
                self.db.update_document_version(document_id, version, progress=35, parser=parsed.parser)
                with trace.span("chunking.select_and_build", "chunking", {"parser": parsed.parser}) as span:
                    policy, chunks = await asyncio.to_thread(
                        build_chunks,
                        parsed,
                        document_id=document_id,
                        document_version=version,
                        policy_override=(json.loads(document.get("metadata_json") or "{}").get("chunkPolicyOverride")),
                    )
                    span.set_output({"policy": policy, "chunkCount": len(chunks)})
                self.db.update_document_version(document_id, version, progress=55, chunk_policy=policy)
                with trace.span("embedding.index", "embedding", {"chunkCount": len(chunks)}) as span:
                    vectors = await asyncio.to_thread(self.embedding.embed, [chunk["text"] for chunk in chunks])
                    for chunk, vector in zip(chunks, vectors):
                        import numpy as np

                        array = np.asarray(vector, dtype=np.float32)
                        chunk["embedding_blob"] = array.tobytes()
                        chunk["embedding_dim"] = int(array.size)
                    self.db.insert_chunks(chunks)
                    span.set_output(
                        {
                            "chunkCount": len(chunks),
                            "dimension": len(vectors[0]) if vectors else 0,
                            "provider": self.embedding.status.provider,
                            "fallback": self.embedding.status.fallback,
                        }
                    )
                result = {
                    "pageCount": parsed.page_count,
                    "sheetCount": parsed.sheet_count,
                    "elementCount": len(parsed.elements),
                    "chunkCount": len(chunks),
                    "chunkPolicy": policy,
                    "embeddingProvider": self.embedding.status.provider,
                    "embeddingModel": self.embedding.status.model,
                }
                self.db.update_document_version(
                    document_id,
                    version,
                    status="READY_FOR_TEST",
                    progress=100,
                    parser=parsed.parser,
                    chunk_policy=policy,
                    processing_result=result,
                    error=None,
                )
                trace.finish(result)
            except DocumentParseError as exc:
                error = {"code": exc.code, "message": redact_text(str(exc), 500)}
                self.db.update_document_version(document_id, version, status="PROCESS_FAILED", progress=100, error=error)
                trace.finish(error, "ERROR")
            except Exception as exc:
                error = {"code": "DOCUMENT_PROCESSING_FAILED", "message": redact_text(str(exc), 500)}
                self.db.update_document_version(document_id, version, status="PROCESS_FAILED", progress=100, error=error)
                trace.finish(error, "ERROR")

    def get_document(self, document_id: str, version: int | None = None) -> DocumentResponse:
        row = self.db.get_document_version(document_id, version)
        if not row:
            raise RagError("RESOURCE_NOT_FOUND", "文档不存在", 404)
        return self._document_response(row)

    def list_documents(self, knowledge_base_id: str = "main-business-kb") -> list[DocumentResponse]:
        return [self._document_response(row) for row in self.db.list_documents(knowledge_base_id)]

    def document_chunks(self, document_id: str, version: int | None = None) -> list[dict[str, Any]]:
        if not self.db.get_document_version(document_id, version):
            raise RagError("RESOURCE_NOT_FOUND", "文档不存在", 404)
        chunks = self.db.get_chunks_for_document(document_id, version)
        result = []
        for chunk in chunks:
            item = {key: value for key, value in chunk.items() if key not in {"embedding_blob", "embedding_dim"}}
            item["excerpt"] = item.get("text", "")[:1000]
            result.append(item)
        return result

    def _prepare_scope(
        self,
        *,
        knowledge_base_id: str,
        release_id: str | None = None,
        test_session_id: str | None = None,
    ) -> PreparedScope:
        if test_session_id:
            session = self.db.get_test_session(test_session_id)
            if not session:
                raise RagError("RESOURCE_NOT_FOUND", "测试会话不存在", 404)
            chunk_ids: list[str] = []
            base_release_id = session.get("base_release_id") if session.get("mode") != "SINGLE_DOCUMENT" else None
            release = self.db.get_release(base_release_id) if base_release_id else None
            if release and release.get("knowledge_base_id") != knowledge_base_id:
                raise RagError("INVALID_REQUEST", "基础知识版本不属于目标知识库", 422)
            if release:
                chunk_ids.extend(item["chunk_id"] for item in self.db.get_release_members(base_release_id))
            for candidate in session.get("candidate_documents", []):
                candidate_row = self.db.get_document_version(candidate["documentId"], int(candidate["documentVersion"]))
                if not candidate_row or candidate_row.get("knowledge_base_id") != knowledge_base_id:
                    raise RagError("INVALID_REQUEST", "候选文档不属于目标知识库", 422, details=candidate)
                chunk_ids.extend(
                    item["chunk_id"]
                    for item in self.db.get_chunks_for_document(candidate["documentId"], int(candidate["documentVersion"]))
                )
            chunk_ids = list(dict.fromkeys(chunk_ids))
            if not chunk_ids:
                raise RagError("RAG_NOT_READY", "测试范围没有可检索内容", 503, True)
            if release is None:
                release = {
                    "release_id": None,
                    "knowledge_version": f"test-{test_session_id}",
                }
            return PreparedScope(SearchScope(knowledge_base_id, chunk_ids=chunk_ids, mode="TEST"), release, session)
        if release_id:
            release = self.db.get_release(release_id)
            if not release or release.get("status") not in {"PUBLISHED", "ROLLED_BACK"}:
                raise RagError("RESOURCE_NOT_FOUND", "正式知识版本不存在", 404)
            if release.get("knowledge_base_id") != knowledge_base_id:
                raise RagError("INVALID_REQUEST", "知识版本不属于目标知识库", 422)
        else:
            release = self.db.get_active_release(knowledge_base_id)
        if not release:
            raise RagError("RAG_NOT_READY", "没有已发布知识版本", 503, True)
        return PreparedScope(SearchScope(knowledge_base_id, release_id=release["release_id"], mode="PUBLISHED"), release)

    async def _rewrite(self, question: str, context: list[QueryMessage], trace: TraceHandle) -> list[str]:
        normalized = _normalize_query(question)
        variants = [normalized]
        if context:
            last_user = next((item.text for item in reversed(context) if item.role == "USER"), "")
            if last_user and len(normalized) < 18 and not any(mark in normalized for mark in "多少钱多少如何哪里吗？?"):
                variants.append(_normalize_query(f"{last_user} {normalized}"))
        # Add a punctuation-normalized variant without inventing domain facts.
        compact = re.sub(r"[，。！？；：、,.!?;:]", " ", normalized)
        compact = _normalize_query(compact)
        if compact and compact not in variants:
            variants.append(compact)
        if self.settings.query_rewrite_enabled and self.llm.enabled:
            with trace.span("query.rewrite.llm", "llm", {"question": normalized}) as span:
                try:
                    payload, data = await self.llm.json_complete(
                        "你是检索查询改写器。只做同义改写和必要的上下文补全，不回答问题，不添加原问题没有的事实。输出JSON对象 {\"queries\":[string]}，最多3条。",
                        json.dumps({"question": normalized, "context": [item.model_dump() for item in context[-4:]]}, ensure_ascii=False),
                        max_tokens=240,
                    )
                    generated = payload.get("queries", []) if isinstance(payload, dict) else []
                    for item in generated[:3]:
                        value = _normalize_query(str(item))
                        if value and value not in variants:
                            variants.append(value)
                    span.set_output({"queries": variants[:5], "provider": data.get("model", self.settings.llm_model)})
                except Exception as exc:
                    span.set_output({"fallback": True, "error": redact_text(str(exc), 300)})
        return variants[:5]

    async def _retrieve(
        self,
        variants: list[str],
        scope: SearchScope,
        trace: TraceHandle,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        with trace.span("retrieval.hybrid", "retriever", {"variants": variants, "scope": scope.mode}) as span:
            result = await asyncio.to_thread(self.retriever.search_many, variants, scope, top_k=top_k)
            candidates = result.get("hits", [])
            selected = result.get("selected", [])
            runs = result.get("variantPools", [])
            span.add_candidates(candidates[:40])
            span.set_output({"variantCount": len(variants), "candidateCount": len(candidates), "selectedCount": len(selected)})
        return selected, runs

    def _citation(self, hit: dict[str, Any], verification: str = "VERIFIED") -> Citation:
        location = hit.get("location") or {}
        return Citation(
            documentId=str(hit.get("document_id", "")),
            documentVersion=int(hit.get("document_version") or 1),
            chunkId=str(hit.get("chunk_id", "")),
            title=str(hit.get("title") or "未命名内容"),
            excerpt=str(hit.get("text", ""))[:1200],
            page=location.get("page"),
            sectionPath=list(hit.get("section_path") or []),
            sheet=location.get("sheet"),
            rowStart=location.get("rowStart") or location.get("row"),
            rowEnd=location.get("rowEnd") or location.get("row"),
            contentHash=hit.get("content_hash"),
            verificationStatus=verification,
        )

    def _evidence_gate(
        self,
        question: str,
        selected: list[dict[str, Any]],
        *,
        require_grounding: bool,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        if _is_prompt_attack(question):
            return False, "OUT_OF_SCOPE", {"blocked": True, "rule": "prompt_attack"}
        if not selected:
            return False, "NO_RELEVANT_KNOWLEDGE", {"blocked": True, "reason": "no_candidates"}
        top = float(selected[0].get("reranker_score_normalized") or 0)
        second = float(selected[1].get("reranker_score_normalized") or 0) if len(selected) > 1 else 0
        if require_grounding and top < self.settings.evidence_min_score:
            return False, "INSUFFICIENT_EVIDENCE", {"blocked": True, "topScore": top, "threshold": self.settings.evidence_min_score}
        # Only call this a conflict when the same normalized question has competing answers.
        # Different nearby FAQs are normal for a broad query and must not be rejected by default.
        competing: dict[str, set[str]] = {}
        for item in selected[:8]:
            raw_question = str((item.get("metadata") or {}).get("question") or "").strip()
            normalized_question = re.sub(r"[\s，。！？；：、,.!?;:]", "", raw_question.lower())
            if normalized_question:
                competing.setdefault(normalized_question, set()).add(
                    hashlib.sha256(_extract_answer_text(item).encode("utf-8")).hexdigest()[:12]
                )
        conflicting_questions = [key for key, values in competing.items() if len(values) > 1]
        if conflicting_questions and top - second < 0.04:
            return False, "KNOWLEDGE_CONFLICT", {"blocked": True, "conflictingQuestions": conflicting_questions[:3]}
        return True, None, {
            "blocked": False,
            "topScore": top,
            "scoreGap": max(0, top - second),
            "selectedCount": len(selected),
        }

    def _negative_block(self, question: str, selected: list[dict[str, Any]], knowledge_base_id: str) -> dict[str, Any] | None:
        normalized = re.sub(r"\s+", "", question.lower())
        selected_ids = {item.get("chunk_id") for item in selected}
        for case in self.db.list_negative_cases(knowledge_base_id):
            blocked_ids = set(case.get("blocked_chunk_ids", []))
            case_question = re.sub(r"\s+", "", str(case.get("question", "")).lower())
            if blocked_ids & selected_ids and (
                case_question == normalized
                or case_question in normalized
                or normalized in case_question
                or _question_similarity(question, str(case.get("question", ""))) >= 0.30
            ):
                return case
        return None

    @staticmethod
    def _confidence(selected: list[dict[str, Any]]) -> tuple[float, str]:
        if not selected:
            return 0.0, "LOW"
        top = _clamp(float(selected[0].get("reranker_score_normalized") or 0))
        second = _clamp(float(selected[1].get("reranker_score_normalized") or 0)) if len(selected) > 1 else 0
        gap = _clamp(top - second, 0, 1)
        agreement = sum(1 for item in selected[:3] if item.get("vector_score") is not None and item.get("lexical_score") is not None) / min(3, len(selected))
        score = _clamp(top * 0.7 + gap * 0.15 + agreement * 0.15)
        band = "HIGH" if score >= 0.75 else "MEDIUM" if score >= 0.5 else "LOW"
        return score, band

    async def _generate(
        self,
        *,
        question: str,
        context: list[QueryMessage],
        selected: list[dict[str, Any]],
        trace_id: str,
        scope: SearchScope,
        max_answer_chars: int,
        use_agent: bool,
        generate: bool = True,
    ) -> tuple[AgentAnswer, dict[str, Any]]:
        if not selected:
            return AgentAnswer(result="NO_ANSWER", reason_code="INSUFFICIENT_EVIDENCE"), {"framework": "none"}
        if not generate:
            top = selected[0]
            return AgentAnswer(
                result="ANSWERED",
                answer=_extract_answer_text(top)[:max_answer_chars],
                citation_ids=[top["chunk_id"]],
            ), {"framework": "deterministic-evaluation", "provider": "top-evidence", "model": "top-evidence"}
        context_text = "\n".join(f"{item.role}: {item.text}" for item in context[-6:])
        if use_agent:
            result = await self.agent.run(
                question=question,
                context_text=context_text,
                evidence=selected,
                trace_id=trace_id,
                max_answer_chars=max_answer_chars,
                scope=scope,
            )
        else:
            result = await self.agent.run(
                question=question,
                context_text=context_text,
                evidence=selected,
                trace_id=trace_id,
                max_answer_chars=max_answer_chars,
                scope=scope,
                use_agent=False,
            )
        if result.output is not None:
            return result.output, {
                "framework": result.framework,
                "provider": result.provider,
                "model": result.model,
                "error": result.error,
                "rawText": redact_text(result.raw_text, 1200),
                "toolSearches": [
                    {"query": item.get("query"), "selectedCount": len((item.get("result") or {}).get("selected", []))}
                    for item in result.tool_searches
                ],
            }
        if self.llm.enabled:
            raise RagExecutionError(
                result.error or "generation failed",
                code=result.error_code or "RAG_EXECUTION_FAILED",
                status_code=result.status_code or 500,
                retryable=result.retryable,
            )
        top = selected[0]
        return AgentAnswer(
            result="ANSWERED",
            answer=_extract_answer_text(top)[:max_answer_chars],
            citation_ids=[top["chunk_id"]],
        ), {"framework": "deterministic-fallback", "provider": "top-evidence", "model": "top-evidence"}

    async def run_query(
        self,
        *,
        question: str,
        snapshot: Snapshot,
        source: SourceInfo,
        knowledge_base_id: str,
        trace_id: str,
        request_id: str,
        context: list[QueryMessage] | None = None,
        release_id: str | None = None,
        test_session_id: str | None = None,
        max_answer_chars: int | None = None,
        use_rewrite: bool = True,
        use_agent: bool = True,
        top_k: int | None = None,
        generate: bool = True,
        debug: bool = False,
        require_grounding: bool = True,
    ) -> QueryResponse:
        started_ns = time.perf_counter_ns()
        question = _normalize_query(question)
        raw_context = context or []
        if any(_is_prompt_attack(item.text) for item in raw_context):
            raw_context = []
        context = _bound_context(raw_context, self.settings.max_context_messages, self.settings.max_context_chars)
        max_answer_chars = max_answer_chars or self.settings.max_answer_chars
        top_k = top_k or self.settings.default_top_k
        try:
            prepared = self._prepare_scope(
                knowledge_base_id=knowledge_base_id,
                release_id=release_id,
                test_session_id=test_session_id,
            )
        except RagError as exc:
            if not self.db.get_trace(trace_id):
                with self.tracer.start(
                    trace_id=trace_id,
                    request_id=request_id,
                    business_trace_id=trace_id,
                    name="rag.query",
                    input_value={"question": question, "scope": "UNRESOLVED"},
                    attributes={"knowledgeBaseId": knowledge_base_id},
                ) as failed_trace:
                    with failed_trace.span("scope.resolve", "guard", {"releaseId": release_id, "testSessionId": test_session_id}) as span:
                        span.set_output({"errorCode": exc.code})
                    failed_trace.finish({"errorCode": exc.code, "message": redact_text(str(exc), 300)}, "ERROR")
            raise
        existing_trace = self.db.get_trace(trace_id)
        if existing_trace:
            raise RagError("IDEMPOTENCY_CONFLICT", "traceId已经被使用，请为新运行生成新的traceId", 409)
        trace = self.tracer.start(
            trace_id=trace_id,
            request_id=request_id,
            business_trace_id=trace_id,
            name="rag.query",
            input_value={
                "question": question,
                "snapshot": snapshot.model_dump(),
                "source": source.model_dump(),
                "scope": prepared.scope.mode,
            },
            attributes={
                "knowledge_base_id": knowledge_base_id,
                "release_id": prepared.release.get("release_id") if prepared.release else None,
                "test_session_id": test_session_id,
            },
        )
        try:
            with trace:
                with trace.span("query.prepare", "workflow", {"question": question}) as span:
                    if len(question) > self.settings.max_query_chars:
                        raise RagError("INVALID_QUERY", "query.text超过长度限制", 413)
                    if _is_prompt_attack(question):
                        span.set_output({"blocked": True, "reason": "prompt_attack"})
                        result = self._no_answer(
                            request_id, trace_id, snapshot, knowledge_base_id, prepared, "OUT_OF_SCOPE", 0, [], {}, {}
                        )
                        result.meta.latencyMs = int((time.perf_counter_ns() - started_ns) / 1_000_000)
                        trace.finish(result.model_dump())
                        return result
                    span.set_output({"normalizedLength": len(question), "contextCount": len(context)})
                with trace.span("query.rewrite", "workflow", {"question": question}) as span:
                    variants = await self._rewrite(question, context, trace) if use_rewrite else [question]
                    span.set_output({"variants": variants, "count": len(variants)})
                selected, retrieval_runs = await self._retrieve(variants, prepared.scope, trace, top_k)
                with trace.span("evidence.gate", "guard", {"candidateCount": len(selected)}) as span:
                    allowed, reason, gate = self._evidence_gate(question, selected, require_grounding=require_grounding)
                    detail_gap = _unsupported_detail(question, selected)
                    if allowed and detail_gap:
                        allowed = False
                        reason = "INSUFFICIENT_EVIDENCE"
                        gate["detailGap"] = detail_gap
                    negative = self._negative_block(question, selected, knowledge_base_id)
                    if negative:
                        allowed = False
                        reason = "DISABLED_TEST_CASE"
                        gate["negativeCaseId"] = negative.get("negative_case_id")
                    span.set_output(gate)
                if not allowed:
                    response = self._no_answer(
                        request_id,
                        trace_id,
                        snapshot,
                        knowledge_base_id,
                        prepared,
                        reason or "INSUFFICIENT_EVIDENCE",
                        0,
                        selected,
                        {
                            "variants": _public_variants(variants, include_text=debug),
                            "retrievalRuns": _public_retrieval_runs(retrieval_runs, include_text=debug),
                            "evidenceGate": gate,
                            "selectedCandidates": [_public_hit(item, include_excerpt=debug) for item in selected],
                        },
                        {},
                        debug=debug,
                    )
                    response.meta.latencyMs = int((time.perf_counter_ns() - started_ns) / 1_000_000)
                    trace.finish(response.model_dump())
                    return response
                with trace.span("answer.generate", "agent", {"evidenceCount": len(selected)}) as span:
                    agent_output, agent_meta = await self._generate(
                        question=question,
                        context=context,
                        selected=selected,
                        trace_id=trace_id,
                        scope=prepared.scope,
                        max_answer_chars=max_answer_chars,
                        use_agent=use_agent,
                        generate=generate,
                    )
                    span.set_output({"result": agent_output.result, **agent_meta})
                valid_ids = {item["chunk_id"] for item in selected}
                citation_ids = [item for item in agent_output.citation_ids if item in valid_ids]
                invalid_citation_ids = [item for item in agent_output.citation_ids if item not in valid_ids]
                verification = "VERIFIED"
                if agent_output.result == "ANSWERED" and not citation_ids:
                    agent_output = AgentAnswer(
                        result="NO_ANSWER",
                        answer="",
                        citation_ids=[],
                        reason_code="MODEL_OUTPUT_UNGROUNDED",
                    )
                if agent_output.result.upper() != "ANSWERED" or not str(agent_output.answer or "").strip():
                    response = self._no_answer(
                        request_id,
                        trace_id,
                        snapshot,
                        knowledge_base_id,
                        prepared,
                        agent_output.reason_code or "INSUFFICIENT_EVIDENCE",
                        0,
                        selected,
                        {
                            "variants": _public_variants(variants, include_text=debug),
                            "retrievalRuns": _public_retrieval_runs(retrieval_runs, include_text=debug),
                            "evidenceGate": gate,
                            "selectedCandidates": [_public_hit(item, include_excerpt=debug) for item in selected],
                            "agent": _public_agent_meta(agent_meta, include_raw=debug),
                            "invalidCitationIds": invalid_citation_ids,
                        },
                        agent_meta,
                        debug=debug,
                    )
                    response.meta.latencyMs = int((time.perf_counter_ns() - started_ns) / 1_000_000)
                    trace.finish(response.model_dump())
                    return response
                selected_by_id = {item["chunk_id"]: item for item in selected}
                citations = [self._citation(selected_by_id[item], verification) for item in citation_ids]
                confidence, confidence_band = self._confidence(selected)
                grounding = Grounding(
                    confidence=confidence,
                    confidenceBand=confidence_band,
                    confidenceType="DIAGNOSTIC_NOT_CALIBRATED_PROBABILITY",
                    sourceReferences=citations,
                    evidenceCoverage=_clamp(len(citation_ids) / max(1, min(3, len(selected)))),
                    scores=[
                        {
                            "rerankerScore": item.get("reranker_score"),
                            "rerankerScoreNormalized": item.get("reranker_score_normalized"),
                            "retrievalScore": item.get("retrieval_score"),
                            "vectorScore": item.get("vector_score"),
                            "lexicalScore": item.get("lexical_score"),
                            "scoreType": self.reranker.status.provider,
                            "higherIsBetter": True,
                            "model": self.reranker.status.model,
                        }
                        for item in selected
                    ],
                )
                meta = QueryMeta(
                    knowledgeBaseId=knowledge_base_id,
                    releaseId=prepared.release.get("release_id") if prepared.release else None,
                    knowledgeVersion=prepared.release.get("knowledge_version") if prepared.release else None,
                    serviceVersion=self.settings.service_version,
                    traceId=trace_id,
                    embeddingProvider=self.embedding.status.provider,
                    rerankerProvider=self.reranker.status.provider,
                    generationProvider=agent_meta.get("provider"),
                    agentFramework=agent_meta.get("framework"),
                )
                answer_text = str(agent_output.answer).strip()[:max_answer_chars]
                response = QueryResponse(
                    contractVersion=self.settings.contract_version,
                    requestId=request_id,
                    traceId=trace_id,
                    snapshot=snapshot,
                    result="ANSWERED",
                    answer=AnswerBody(text=answer_text, format="PLAIN_TEXT"),
                    grounding=grounding,
                    meta=meta,
                    diagnostics={
                        "variants": _public_variants(variants, include_text=debug),
                        "retrievalRuns": _public_retrieval_runs(retrieval_runs, include_text=debug),
                        "selectedCandidates": [_public_hit(item, include_excerpt=debug) for item in selected],
                        "evidenceGate": gate,
                        "agent": _public_agent_meta(agent_meta, include_raw=debug),
                        "invalidCitationIds": invalid_citation_ids,
                    },
                )
                response.meta.latencyMs = int((time.perf_counter_ns() - started_ns) / 1_000_000)
                trace.finish(response.model_dump())
                return response
        except RagError:
            raise
        except Exception as exc:
            if isinstance(exc, RagExecutionError):
                raise
            raise RagExecutionError(str(exc)) from exc

    def _no_answer(
        self,
        request_id: str,
        trace_id: str,
        snapshot: Snapshot,
        knowledge_base_id: str,
        prepared: PreparedScope,
        reason: str,
        latency_ms: int,
        selected: list[dict[str, Any]],
        diagnostics: dict[str, Any],
        agent_meta: dict[str, Any],
        debug: bool = False,
    ) -> QueryResponse:
        citations = [self._citation(item) for item in selected[:3]] if debug else []
        confidence, confidence_band = self._confidence(selected)
        meta = QueryMeta(
            knowledgeBaseId=knowledge_base_id,
            releaseId=prepared.release.get("release_id") if prepared.release else None,
            knowledgeVersion=prepared.release.get("knowledge_version") if prepared.release else None,
            serviceVersion=self.settings.service_version,
            latencyMs=latency_ms,
            traceId=trace_id,
            embeddingProvider=self.embedding.status.provider,
            rerankerProvider=self.reranker.status.provider,
            generationProvider=agent_meta.get("provider"),
            agentFramework=agent_meta.get("framework"),
        )
        return QueryResponse(
            contractVersion=self.settings.contract_version,
            requestId=request_id,
            traceId=trace_id,
            snapshot=snapshot,
            result="NO_ANSWER",
            answer=None,
            reasonCode=reason,
            grounding=Grounding(
                confidence=confidence,
                confidenceBand=confidence_band,
                confidenceType="DIAGNOSTIC_NOT_CALIBRATED_PROBABILITY",
                sourceReferences=citations,
                evidenceCoverage=0,
                scores=[
                    {
                        "rerankerScore": item.get("reranker_score"),
                        "rerankerScoreNormalized": item.get("reranker_score_normalized"),
                        "retrievalScore": item.get("retrieval_score"),
                        "vectorScore": item.get("vector_score"),
                        "lexicalScore": item.get("lexical_score"),
                        "scoreType": self.reranker.status.provider,
                        "higherIsBetter": True,
                        "model": self.reranker.status.model,
                    }
                    for item in selected[:8]
                ],
            ),
            meta=meta,
            diagnostics=diagnostics,
        )

    async def formal_query(self, request: QueryRequest) -> QueryResponse:
        if request.contractVersion != self.settings.contract_version:
            raise RagError("INVALID_REQUEST", "不支持的合同版本", 400)
        if request.knowledgeScope.tenantId != "local-default":
            raise RagError("UNAUTHORIZED", "当前本地实例不包含该tenant", 401)
        if not request.constraints.requireGrounding:
            raise RagError("INVALID_QUERY", "正式查询必须启用requireGrounding", 422)
        return await self.run_query(
            question=request.query.text,
            snapshot=request.snapshot,
            source=request.source,
            knowledge_base_id=request.knowledgeScope.knowledgeBaseId,
            trace_id=request.traceId,
            request_id=request.requestId,
            context=request.context,
            max_answer_chars=request.constraints.maxAnswerChars,
            require_grounding=request.constraints.requireGrounding,
            debug=False,
        )

    async def debug_query(self, request: DebugQueryRequest) -> QueryResponse:
        trace_id = "debug-" + uuid.uuid4().hex
        snapshot = Snapshot(
            conversationKey="DEBUG:local",
            conversationVersion=0,
            inputFingerprint="sha256:" + _json_hash(request.model_dump()),
        )
        return await self.run_query(
            question=request.question,
            snapshot=snapshot,
            source=SourceInfo(platform=request.platform, accountId="debug", channelType=request.channelType),
            knowledge_base_id=request.knowledgeBaseId,
            trace_id=trace_id,
            request_id="debug-request-" + uuid.uuid4().hex[:12],
            context=request.context,
            release_id=request.releaseId,
            test_session_id=request.testSessionId,
            max_answer_chars=request.maxAnswerChars,
            use_rewrite=request.useRewrite,
            use_agent=request.useAgent,
            top_k=request.topK,
            debug=True,
        )

    async def debug_retrieve(self, request: DebugQueryRequest) -> dict[str, Any]:
        prepared = self._prepare_scope(
            knowledge_base_id=request.knowledgeBaseId,
            release_id=request.releaseId,
            test_session_id=request.testSessionId,
        )
        trace_id = "retrieve-" + uuid.uuid4().hex
        trace = self.tracer.start(
            trace_id=trace_id,
            request_id="retrieve-request-" + uuid.uuid4().hex[:12],
            name="rag.retrieve_debug",
            input_value={"question": request.question, "scope": prepared.scope.mode},
            attributes={"knowledgeBaseId": request.knowledgeBaseId},
        )
        with trace:
            with trace.span("query.rewrite", "workflow", {"question": request.question}) as span:
                variants = await self._rewrite(request.question, request.context, trace) if request.useRewrite else [_normalize_query(request.question)]
                span.set_output({"variants": variants})
            with trace.span("retrieval.hybrid", "retriever", {"variants": variants, "scope": prepared.scope.mode}) as retrieval_span:
                retrieval_result = await asyncio.to_thread(
                    self.retriever.search_many, variants, prepared.scope, top_k=request.topK
                )
                retrieval_span.add_candidates(retrieval_result.get("candidatePool", [])[:40])
                retrieval_span.set_output({"candidateCount": len(retrieval_result.get("candidatePool", [])), "selectedCount": len(retrieval_result.get("selected", []))})
            selected = retrieval_result.get("selected", [])
            runs = retrieval_result.get("variantPools", [])
            allowed, reason, gate = self._evidence_gate(request.question, selected, require_grounding=True)
            result = {
                "traceId": trace_id,
                "scope": prepared.scope.mode,
                "releaseId": prepared.release.get("release_id") if prepared.release else None,
                "knowledgeVersion": prepared.release.get("knowledge_version") if prepared.release else None,
                "variants": variants,
                "retrievalRuns": runs,
                "selected": [_public_hit(item, include_excerpt=True) for item in selected],
                "candidates": [_public_hit(item, include_excerpt=True) for item in retrieval_result.get("candidatePool", selected)],
                "evidenceGate": {**gate, "allowed": allowed, "reasonCode": reason},
            }
            trace.finish(result)
            return result

    def create_test_session(self, request: TestSessionCreate) -> TestSessionResponse:
        if request.mode == "SINGLE_DOCUMENT" and request.baseReleaseId:
            raise RagError("INVALID_REQUEST", "SINGLE_DOCUMENT不能绑定正式基础版本", 422)
        base_release_id = None if request.mode == "SINGLE_DOCUMENT" else request.baseReleaseId or (self.active_release(request.knowledgeBaseId) or {}).get("release_id")
        candidates = [item.model_dump() for item in request.candidateDocuments]
        for candidate in candidates:
            row = self.db.get_document_version(candidate["documentId"], int(candidate["documentVersion"]))
            if not row or row.get("status") != "READY_FOR_TEST" or row.get("knowledge_base_id") != request.knowledgeBaseId:
                raise RagError("DOCUMENT_VERSION_NOT_READY", "候选文档尚未准备好测试", 422, details=candidate)
        if request.mode == "SINGLE_DOCUMENT" and len(candidates) != 1:
            raise RagError("INVALID_REQUEST", "SINGLE_DOCUMENT必须且只能选择一个文档", 422)
        if request.mode == "PRE_RELEASE" and not candidates and not base_release_id:
            raise RagError("RAG_NOT_READY", "PRE_RELEASE需要基础版本或候选文档", 422)
        if base_release_id:
            base = self.db.get_release(base_release_id)
            if not base or base.get("knowledge_base_id") != request.knowledgeBaseId or base.get("status") not in {"PUBLISHED", "ROLLED_BACK"}:
                raise RagError("INVALID_REQUEST", "基础知识版本不可用于测试", 422)
        session_id = "test-session-" + uuid.uuid4().hex[:16]
        row = self.db.create_test_session(
            {
                "test_session_id": session_id,
                "knowledge_base_id": request.knowledgeBaseId,
                "mode": request.mode,
                "base_release_id": base_release_id,
                "candidate_documents": candidates,
                "operator_id": request.operatorId,
            }
        )
        return TestSessionResponse(
            requestId=request.requestId,
            testSessionId=session_id,
            status=row["status"],
            mode=row["mode"],
            baseReleaseId=row.get("base_release_id"),
            candidateDocuments=[TestCandidate(**item) for item in row.get("candidate_documents", [])],
            createdAt=row.get("created_at"),
        )

    async def test_query(self, session_id: str, request: TestQueryRequest) -> TestAnswerResponse:
        session = self.db.get_test_session(session_id)
        if not session:
            raise RagError("RESOURCE_NOT_FOUND", "测试会话不存在", 404)
        existing = self.db.get_test_answer_by_request(session_id, request.requestId)
        if existing:
            return TestAnswerResponse(
                requestId=request.requestId,
                testSessionId=session_id,
                answerId=existing["answer_id"],
                result=existing["result"],
                answer=AnswerBody.model_validate(existing["answer"]) if existing.get("answer") else None,
                decision=existing.get("decision", "ENABLED"),
                sourceReferences=[Citation.model_validate(item) for item in existing.get("source_references", [])],
                grounding=Grounding.model_validate(existing["grounding"]) if existing.get("grounding") else None,
                reasonCode=existing.get("reason_code"),
                traceId=existing["trace_id"],
                meta={"knowledgeVersion": (existing.get("grounding") or {}).get("knowledgeVersion")},
            )
        trace_id = "test-" + uuid.uuid4().hex
        snapshot = Snapshot(
            conversationKey=f"TEST:{session_id}",
            conversationVersion=0,
            inputFingerprint="sha256:" + _json_hash({"question": request.question, "context": [item.model_dump() for item in request.context]}),
        )
        response = await self.run_query(
            question=request.question,
            snapshot=snapshot,
            source=SourceInfo(platform="DOUYIN", accountId="test", channelType="DIRECT_MESSAGE"),
            knowledge_base_id=session["knowledge_base_id"],
            trace_id=trace_id,
            request_id=request.requestId,
            context=request.context,
            test_session_id=session_id,
            debug=True,
        )
        answer_id = "test-answer-" + hashlib.sha256(
            f"{session_id}:{request.requestId}".encode("utf-8")
        ).hexdigest()[:16]
        saved = self.db.save_test_answer(
            {
                "answer_id": answer_id,
                "test_session_id": session_id,
                "request_id": request.requestId,
                "trace_id": trace_id,
                "question": request.question,
                "context": [item.model_dump() for item in request.context],
                "result": response.result,
                "answer": response.answer.model_dump() if response.answer else None,
                "source_references": [item.model_dump() for item in (response.grounding.sourceReferences if response.grounding else [])],
                "grounding": response.grounding.model_dump() if response.grounding else None,
                "diagnostics": response.diagnostics,
                "reason_code": response.reasonCode,
                "operator_id": session.get("operator_id"),
            }
        )
        return TestAnswerResponse(
            requestId=request.requestId,
            testSessionId=session_id,
            answerId=answer_id,
            result=response.result,
            answer=response.answer,
            decision=saved.get("decision", "ENABLED"),
            sourceReferences=[Citation.model_validate(item) for item in saved.get("source_references", [])],
            grounding=Grounding.model_validate(saved["grounding"]) if saved.get("grounding") else None,
            reasonCode=response.reasonCode,
            traceId=trace_id,
            meta={"latencyMs": response.meta.latencyMs, "knowledgeVersion": response.meta.knowledgeVersion},
        )

    def update_test_decision(self, answer_id: str, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("decision") == "DISABLED" and not request.get("reasonCode"):
            raise RagError("INVALID_REQUEST", "DISABLED必须提供reasonCode", 422)
        row = self.db.update_test_decision(
            answer_id,
            {
                "decision": request["decision"],
                "reason_code": request.get("reasonCode"),
                "note": request.get("note"),
                "operator_id": request.get("operatorId", "local-operator"),
            },
        )
        if not row:
            raise RagError("RESOURCE_NOT_FOUND", "测试答案不存在", 404)
        return {
            "requestId": request.get("requestId"),
            "testSessionId": row["test_session_id"],
            "answerId": row["answer_id"],
            "decision": row["decision"],
            "decisionVersion": row["decision_version"],
            "updatedAt": row["updated_at"],
        }

    def _release_members_for_session(self, session: dict[str, Any], candidates: list[TestCandidate]) -> list[dict[str, Any]]:
        chunk_map: dict[str, dict[str, Any]] = {}
        base_id = session.get("base_release_id") if session.get("mode") != "SINGLE_DOCUMENT" else None
        if base_id:
            release = self.db.get_release(base_id)
            if not release or release.get("knowledge_base_id") != session.get("knowledge_base_id"):
                raise RagError("RESOURCE_NOT_FOUND", "基础知识版本不存在", 404)
            for item in self.db.get_release_members(base_id):
                chunk_map[item["chunk_id"]] = item
        for candidate in candidates:
            row = self.db.get_document_version(candidate.documentId, candidate.documentVersion)
            if not row or row.get("status") != "READY_FOR_TEST" or row.get("knowledge_base_id") != session.get("knowledge_base_id"):
                raise RagError("DOCUMENT_VERSION_NOT_READY", "候选文档版本不可发布", 422)
            for item in self.db.get_chunks_for_document(candidate.documentId, candidate.documentVersion):
                chunk_map[item["chunk_id"]] = item
        if not chunk_map:
            raise RagError("RAG_NOT_READY", "发布范围没有chunk", 422)
        return list(chunk_map.values())

    async def start_release(self, request: ReleaseCreate) -> ReleaseResponse:
        session = self.db.get_test_session(request.testSessionId)
        if not session:
            raise RagError("RESOURCE_NOT_FOUND", "测试会话不存在", 404)
        session_candidates = [TestCandidate(**item) for item in session.get("candidate_documents", [])]
        requested_candidates = request.candidateDocuments
        if requested_candidates and {
            (item.documentId, item.documentVersion) for item in requested_candidates
        } != {(item.documentId, item.documentVersion) for item in session_candidates}:
            raise RagError("DOCUMENT_VERSION_CHANGED", "发布候选文档必须与测试会话快照完全一致", 409)
        candidates = session_candidates
        if not candidates:
            raise RagError("INVALID_REQUEST", "发布必须包含测试会话中的候选文档", 422)
        if request.baseReleaseId and session.get("base_release_id") != request.baseReleaseId:
            raise RagError("DOCUMENT_VERSION_CHANGED", "基础知识版本与测试会话不一致", 409)
        effective_base_release_id = None if session.get("mode") == "SINGLE_DOCUMENT" else session.get("base_release_id")
        test_answers = self.db.list_test_answers(request.testSessionId)
        if not test_answers:
            raise RagError("RELEASE_REQUIRES_TEST", "候选文档至少需要一条测试答案后才能发布", 422)
        session_answers = self.db.list_test_answers(request.testSessionId)
        excluded_chunk_ids = {
            reference.get("chunkId")
            for answer in session_answers
            if answer.get("decision") == "DISABLED"
            for reference in answer.get("source_references", [])
            if reference.get("chunkId")
        }
        members = [
            item for item in self._release_members_for_session(session, candidates)
            if item.get("chunk_id") not in excluded_chunk_ids
        ]
        if not members:
            raise RagError("DISABLED_CASE_VALIDATION_FAILED", "禁用来源移除后发布范围为空", 422)
        release_id = "release-" + uuid.uuid4().hex[:16]
        knowledge_version = f"kb-{session['knowledge_base_id']}-{uuid.uuid4().hex[:10]}"
        enabled_count = sum(1 for answer in self.db.list_test_answers(request.testSessionId) if answer.get("decision") == "ENABLED")
        disabled_count = sum(1 for answer in self.db.list_test_answers(request.testSessionId) if answer.get("decision") == "DISABLED")
        manifest = {
            "knowledgeBaseId": session["knowledge_base_id"],
            "baseReleaseId": effective_base_release_id,
            "testSessionId": request.testSessionId,
            "requestId": request.requestId,
            "publishedBy": request.publishedBy,
            "candidateDocuments": [item.model_dump() for item in candidates],
            "chunkCount": len(members),
            "excludedChunkIds": sorted(excluded_chunk_ids),
            "createdAt": now_iso(),
        }
        row = self.db.create_release(
            release_id=release_id,
            knowledge_base_id=session["knowledge_base_id"],
            knowledge_version=knowledge_version,
            base_release_id=effective_base_release_id,
            manifest=manifest,
            members=members,
            published_by=request.publishedBy,
            publish_note=request.publishNote,
            enabled_count=enabled_count,
            disabled_count=disabled_count,
        )
        task = asyncio.create_task(self._build_and_publish_release(release_id, request.testSessionId))
        self._track(task)
        return self._release_response(row)

    def _release_response(self, row: dict[str, Any]) -> ReleaseResponse:
        return ReleaseResponse(
            releaseId=row["release_id"],
            knowledgeBaseId=row.get("knowledge_base_id", "main-business-kb"),
            knowledgeVersion=row.get("knowledge_version"),
            status=row.get("status", "UNKNOWN"),
            publishedAt=row.get("published_at"),
            publishedBy=row.get("published_by"),
            enabledTestCaseCount=int(row.get("enabled_test_case_count") or 0),
            disabledTestCaseCount=int(row.get("disabled_test_case_count") or 0),
            error=row.get("error"),
            baseReleaseId=row.get("base_release_id"),
            createdAt=row.get("created_at"),
        )

    async def _build_and_publish_release(self, release_id: str, session_id: str) -> None:
        try:
            with self.tracer.start(
                trace_id="release-" + uuid.uuid4().hex,
                request_id=(self.db.get_release(release_id) or {}).get("manifest", {}).get("requestId"),
                name="rag.release",
                input_value={"releaseId": release_id, "testSessionId": session_id},
                attributes={"operation": "release_build"},
            ) as trace:
                with trace.span("release.index", "index", {"releaseId": release_id}) as span:
                    index_result = await asyncio.to_thread(self.retriever.build_release_index, release_id)
                    span.set_output(index_result)
                release_info = self.db.get_release(release_id) or {}
                negatives = self.db.list_negative_cases(release_info.get("knowledge_base_id", "main-business-kb"))
                blocked_ids = {chunk_id for case in negatives for chunk_id in case.get("blocked_chunk_ids", [])}
                members = self.db.get_release_members(release_id)
                violations = [case for case in negatives if set(case.get("blocked_chunk_ids", [])) & {item["chunk_id"] for item in members}]
                if violations:
                    details = [
                        {"negativeCaseId": item.get("negative_case_id"), "question": item.get("question", "")}
                        for item in violations[:20]
                    ]
                    error = {
                        "code": "DISABLED_CASE_VALIDATION_FAILED",
                        "message": "被禁用答案对应的错误来源仍在候选正式版本中",
                        "details": details,
                    }
                    self.db.update_release(release_id, status="FAILED", error=error)
                    trace.finish(error, "ERROR")
                    return
                release_row = self.db.get_release(release_id) or {}
                published = self.db.publish_release_atomically(
                    release_id,
                    expected_base_release_id=release_row.get("base_release_id"),
                    request_id=release_row.get("manifest", {}).get("requestId"),
                    operator_id=release_row.get("published_by"),
                    details={"index": index_result},
                )
                if not published or published.get("status") != "PUBLISHED":
                    trace.finish(published or {"status": "FAILED"}, "ERROR")
                    return
                try:
                    self.db.update_test_session_status(session_id, "PUBLISHED")
                except Exception as exc:
                    trace.finish(
                        {"status": "PUBLISHED", "sessionUpdateError": redact_text(str(exc), 300)},
                        "ERROR",
                    )
                    return
                trace.finish({"status": "PUBLISHED", "index": index_result})
        except Exception as exc:
            current = self.db.get_release(release_id)
            if current and current.get("status") not in {"PUBLISHED", "ROLLED_BACK"}:
                self.db.update_release(
                    release_id,
                    status="FAILED",
                    error={"code": "RELEASE_BUILD_FAILED", "message": redact_text(str(exc), 500)},
                )
                self.db.clear_active_release_if(release_id)

    def release(self, release_id: str) -> ReleaseResponse:
        row = self.db.get_release(release_id)
        if not row:
            raise RagError("RESOURCE_NOT_FOUND", "发布版本不存在", 404)
        return self._release_response(row)

    def list_releases(self, knowledge_base_id: str = "main-business-kb") -> list[ReleaseResponse]:
        return [self._release_response(row) for row in self.db.list_releases(knowledge_base_id)]

    async def rollback(
        self,
        target_release_id: str,
        *,
        request_id: str | None = None,
        operator_id: str | None = None,
        note: str = "",
    ) -> ReleaseResponse:
        row = self.db.get_release(target_release_id)
        if not row or row.get("status") not in {"PUBLISHED", "ROLLED_BACK"}:
            raise RagError("RESOURCE_NOT_FOUND", "可回滚版本不存在", 404)
        member_ids = {item["chunk_id"] for item in self.db.get_release_members(target_release_id)}
        violations = [
            case
            for case in self.db.list_negative_cases(row.get("knowledge_base_id", "main-business-kb"))
            if member_ids & set(case.get("blocked_chunk_ids", []))
        ]
        if violations:
            raise RagError(
                "DISABLED_CASE_VALIDATION_FAILED",
                "目标回滚版本包含当前仍被禁用的错误来源",
                422,
                details=[{"negativeCaseId": item.get("negative_case_id"), "question": item.get("question")} for item in violations[:20]],
            )
        trace_id = "rollback-" + uuid.uuid4().hex
        with self.tracer.start(
            trace_id=trace_id,
            request_id=request_id,
            name="rag.rollback",
            input_value={"targetReleaseId": target_release_id},
            attributes={"operatorId": operator_id},
        ) as trace:
            active_now = self.active_release(row.get("knowledge_base_id", "main-business-kb"))
            try:
                updated = self.db.rollback_release_atomically(
                    target_release_id,
                    request_id=request_id,
                    operator_id=operator_id,
                    note=note,
                    expected_active_release_id=active_now.get("release_id") if active_now else "",
                )
            except DatabaseConflict as exc:
                trace.finish({"errorCode": "RELEASE_BASE_CHANGED"}, "ERROR")
                raise RagError("RELEASE_BASE_CHANGED", str(exc), 409) from exc
            trace.finish({"releaseId": target_release_id, "status": "ROLLED_BACK"})
        return self._release_response(updated or row)

    def start_evaluation(
        self,
        cases: list[dict[str, Any]],
        *,
        request_id: str,
        concurrency: int = 3,
        use_generation: bool = True,
    ) -> str:
        case_ids = [str(case.get("id") or case.get("case_id") or "") for case in cases]
        if not cases or any(not case_id for case_id in case_ids) or len(set(case_ids)) != len(case_ids):
            raise RagError("INVALID_REQUEST", "评测集必须包含非空且唯一的case ID", 422)
        active = self.active_release("main-business-kb")
        if not active:
            raise RagError("RAG_NOT_READY", "评测需要一个已发布知识版本", 503, True)
        eval_run_id = "eval-" + uuid.uuid4().hex[:16]
        self.db.create_evaluation_run(eval_run_id, request_id, len(cases))
        dataset_hash = _json_hash(cases)
        self.db.configure_evaluation_run(
            eval_run_id,
            dataset_hash=dataset_hash,
            release_id=active["release_id"],
            config={
                "useGeneration": use_generation,
                "concurrency": concurrency,
                "embeddingModel": self.embedding.status.model,
                "rerankerModel": self.reranker.status.model,
                "generationModel": self.settings.llm_model,
            },
        )
        task = asyncio.create_task(
            self._run_evaluation_safe(
                cases,
                request_id=request_id,
                concurrency=concurrency,
                use_generation=use_generation,
                eval_run_id=eval_run_id,
                release_id=active["release_id"],
            )
        )
        self._track(task)
        return eval_run_id

    async def _run_evaluation_safe(self, *args: Any, eval_run_id: str, **kwargs: Any) -> str:
        try:
            return await self.run_evaluation(*args, eval_run_id=eval_run_id, **kwargs)
        except asyncio.CancelledError:
            self.db.mark_evaluation_failed(eval_run_id, "评测任务被取消")
            raise
        except Exception as exc:
            self.db.mark_evaluation_failed(eval_run_id, redact_text(str(exc), 500))
            raise

    async def run_evaluation(
        self,
        cases: list[dict[str, Any]],
        *,
        request_id: str,
        concurrency: int = 3,
        use_generation: bool = True,
        eval_run_id: str | None = None,
        release_id: str | None = None,
    ) -> str:
        if eval_run_id is None:
            eval_run_id = "eval-" + uuid.uuid4().hex[:16]
            self.db.create_evaluation_run(eval_run_id, request_id, len(cases))
            active = self.active_release("main-business-kb")
            release_id = release_id or (active["release_id"] if active else None)
        semaphore = asyncio.Semaphore(concurrency)

        async def one(case: dict[str, Any]) -> None:
            async with semaphore:
                case_id = str(case.get("id") or case.get("case_id") or uuid.uuid4().hex[:8])
                expected = str(case.get("expectedResult") or case.get("expected_result") or "ANSWERED")
                question = str(case.get("question") or "")
                trace_id = "eval-" + eval_run_id + "-" + case_id
                try:
                    snapshot = Snapshot(
                        conversationKey=f"EVAL:{case_id}",
                        conversationVersion=0,
                        inputFingerprint="sha256:" + _json_hash(case),
                    )
                    context = [QueryMessage.model_validate(item) for item in case.get("context", [])]
                    response = await self.run_query(
                        question=question,
                        snapshot=snapshot,
                        source=SourceInfo(platform="DOUYIN", accountId="eval", channelType="DIRECT_MESSAGE"),
                        knowledge_base_id=str(case.get("knowledgeBaseId") or "main-business-kb"),
                        trace_id=trace_id,
                        request_id=f"{request_id}-{case_id}",
                        context=context,
                        use_agent=use_generation,
                        generate=use_generation,
                        release_id=release_id,
                    )
                    assertions: list[dict[str, Any]] = []
                    if response.result != expected:
                        assertions.append({"code": "RESULT_MISMATCH", "expected": expected, "actual": response.result})
                    expected_reason_codes = set(case.get("expectedReasonCodes", []))
                    expected_reason = case.get("expectedReasonCode")
                    if expected_reason:
                        expected_reason_codes.add(str(expected_reason))
                    if expected_reason_codes and response.reasonCode not in expected_reason_codes:
                        assertions.append(
                            {
                                "code": "REASON_MISMATCH",
                                "expected": sorted(expected_reason_codes),
                                "actual": response.reasonCode,
                            }
                        )
                    passed = not assertions
                    expected_chunks = set(case.get("expectedChunkIds", []))
                    citations = response.grounding.sourceReferences if response.grounding else []
                    if expected_chunks:
                        actual_chunks = {item.chunkId for item in citations}
                        if not (expected_chunks & actual_chunks):
                            assertions.append({"code": "CITATION_MISMATCH", "expected": sorted(expected_chunks), "actual": sorted(actual_chunks)})
                    expected_title = str(case.get("expectedTitleContains") or "")
                    if expected_title:
                        if not citations or not any(expected_title in item.title for item in citations):
                            assertions.append({"code": "CITATION_MISMATCH", "expectedTitle": expected_title})
                    expected_titles = [str(item) for item in case.get("expectedTitleContainsAny", [])]
                    if expected_titles and not (
                        citations
                        and any(any(expected in citation.title for expected in expected_titles) for citation in citations)
                    ):
                        assertions.append({"code": "CITATION_MISMATCH", "expectedTitles": expected_titles})
                    if expected == "ANSWERED":
                        if not response.answer or not response.answer.text.strip():
                            assertions.append({"code": "ANSWER_MISSING"})
                        if not citations:
                            assertions.append({"code": "CITATION_MISSING"})
                        elif any(item.verificationStatus != "VERIFIED" for item in citations):
                            assertions.append({"code": "CITATION_NOT_VERIFIED"})
                        reference = str(case.get("referenceAnswer") or "")
                        required_terms = [str(item) for item in case.get("requiredTerms", [])]
                        if reference and response.answer:
                            overlap = _answer_overlap(response.answer.text, reference)
                            term_hits = sum(1 for term in required_terms if term and term in response.answer.text)
                            if overlap < 0.12 and term_hits == 0:
                                assertions.append({"code": "ANSWER_CONTENT_LOW_OVERLAP", "overlap": round(overlap, 4), "requiredTermHits": term_hits})
                        if use_generation:
                            agent_diag = (response.diagnostics or {}).get("agent") or {}
                            if agent_diag.get("provider") not in {"deepagents", "openai_compatible"}:
                                assertions.append({"code": "MODEL_NOT_EXECUTED", "provider": agent_diag.get("provider")})
                            if agent_diag.get("error"):
                                assertions.append({"code": "MODEL_EXECUTION_DEGRADED", "error": agent_diag.get("error")})
                    passed = not assertions
                    self.db.save_evaluation_case(
                        eval_run_id,
                        {
                            "case_id": case_id,
                            "expected_result": expected,
                            "actual_result": response.result,
                            "passed": passed,
                            "category": case.get("category"),
                            "question": question,
                            "expected": {
                                "result": expected,
                                "reasonCode": case.get("expectedReasonCode"),
                                "reasonCodes": case.get("expectedReasonCodes", []),
                                "title": case.get("expectedTitleContains"),
                                "titles": case.get("expectedTitleContainsAny", []),
                            },
                            "assertions": assertions,
                            "trace_id": response.traceId,
                            "response": response.model_dump(),
                        },
                    )
                except Exception as exc:
                    self.db.save_evaluation_case(
                        eval_run_id,
                        {
                            "case_id": case_id,
                            "expected_result": expected,
                            "actual_result": "ERROR",
                            "passed": False,
                            "category": case.get("category"),
                            "question": question,
                            "expected": {"result": expected},
                            "assertions": [{"code": "CASE_EXECUTION_ERROR", "error": redact_text(str(exc), 500)}],
                            "trace_id": trace_id,
                            "error": redact_text(str(exc), 500),
                        },
                    )

        try:
            await asyncio.gather(*(one(case) for case in cases))
        except asyncio.CancelledError:
            self.db.mark_evaluation_failed(eval_run_id, "评测任务被取消")
            raise
        except Exception as exc:
            self.db.mark_evaluation_failed(eval_run_id, redact_text(str(exc), 500))
            raise
        result = self.db.get_evaluation_run(eval_run_id) or {}
        case_rows = result.get("cases", [])
        if len(case_rows) != len(cases):
            self.db.mark_evaluation_failed(
                eval_run_id,
                f"评测结果数量不一致: expected={len(cases)}, actual={len(case_rows)}",
            )
            return eval_run_id
        total = len(case_rows)
        passed = sum(1 for item in case_rows if item.get("passed"))
        by_type: dict[str, dict[str, int]] = {}
        rows_by_case = {str(row.get("case_id")): row for row in case_rows}
        for case in cases:
            row = rows_by_case.get(str(case.get("id") or case.get("case_id") or ""), {})
            category = str(case.get("category") or "unknown")
            stats = by_type.setdefault(category, {"total": 0, "passed": 0})
            stats["total"] += 1
            stats["passed"] += int(row.get("passed", 0))
        summary = {
            "total": total,
            "passed": passed,
            "accuracy": passed / total if total else 0,
            "byCategory": by_type,
            "releaseId": release_id,
            "datasetHash": _json_hash(cases),
            "useGeneration": use_generation,
        }
        self.db.finish_evaluation_run(eval_run_id, "COMPLETED", summary)
        return eval_run_id

    def trace(self, trace_id: str) -> dict[str, Any]:
        result = self.db.get_trace(trace_id)
        if not result:
            raise RagError("RESOURCE_NOT_FOUND", "Trace不存在", 404)
        return result

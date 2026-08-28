from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .models import (
    DebugQueryRequest,
    DecisionRequest,
    DocumentResponse,
    EvaluationRunRequest,
    QueryRequest,
    QueryResponse,
    ReleaseCreate,
    ReleaseResponse,
    RollbackRequest,
    TestAnswerResponse,
    TestQueryRequest,
    TestSessionCreate,
    TestSessionResponse,
    TraceFeedbackRequest,
    UploadMetadata,
    now_iso,
)
from .pipeline import RagError, RagService


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = RagService(settings)
    app.state.service = service
    await service.recover_tasks()
    yield
    await service.close()


app = FastAPI(
    title="Local RAG Service",
    version=settings.service_version,
    description="Local-first traceable RAG service with document ingestion, release governance, and debug UI.",
    lifespan=lifespan,
)


@app.middleware("http")
async def security_headers(request: Request, call_next: Any):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
    return response

bearer_scheme = HTTPBearer(auto_error=False)


def service_from(request: Request) -> RagService:
    return request.app.state.service


async def require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> None:
    service = service_from(request)
    if not service.settings.auth_enabled:
        return
    supplied = credentials.credentials if credentials else ""
    if not supplied or not hmac.compare_digest(supplied, service.settings.bearer_token):
        raise RagError("UNAUTHORIZED", "Bearer token无效", 401)


def error_response(request_id: str | None, error: RagError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={
            "contractVersion": settings.contract_version,
            "requestId": request_id or "unknown",
            "error": {
                "code": error.code,
                "message": str(error),
                "retryable": error.retryable,
                "retryAfterMs": 1000 if error.retryable else None,
                "details": error.details,
            },
        },
    )


@app.exception_handler(RagError)
async def rag_error_handler(request: Request, exc: RagError):
    return error_response(request.headers.get("x-request-id"), exc)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    error = RagError("INVALID_REQUEST", "请求字段不合法", 422, details=jsonable_encoder(exc.errors()))
    return error_response(request.headers.get("x-request-id"), error)


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    error = RagError("INTERNAL_ERROR", "RAG服务内部异常", 500, True)
    return error_response(request.headers.get("x-request-id"), error)


def request_id_from(body: Any, header: str | None) -> str:
    body_id = getattr(body, "requestId", None) if body is not None else None
    if header and body_id and header != body_id:
        raise RagError("INVALID_REQUEST", "Header与请求体requestId不一致", 400)
    return str(body_id or header or "request-" + uuid.uuid4().hex)


def replay_or_conflict(service: RagService, request_id: str, operation: str, payload: Any) -> JSONResponse | None:
    payload_hash = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    existing = service.db.get_idempotency(request_id)
    if not existing:
        return None
    if existing["operation"] != operation or existing["payload_hash"] != payload_hash:
        raise RagError("IDEMPOTENCY_CONFLICT", "同一requestId对应了不同请求内容", 409)
    return JSONResponse(status_code=existing["status_code"], content=existing["response"])


def save_idempotency(service: RagService, request_id: str, operation: str, payload: Any, status: int, response: Any) -> None:
    payload_hash = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    service.db.save_idempotency(request_id, operation, payload_hash, status, response)


def serialized(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


async def execute_idempotent(
    service: RagService,
    *,
    request_id: str,
    operation: str,
    payload: Any,
    status_code: int,
    handler: Any,
) -> JSONResponse:
    payload_hash = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    lock = await service.idempotency_lock(request_id)
    async with lock:
        existing = service.db.get_idempotency(request_id)
        if existing:
            if existing["operation"] != operation or existing["payload_hash"] != payload_hash:
                raise RagError("IDEMPOTENCY_CONFLICT", "同一requestId对应了不同请求内容", 409)
            if existing.get("state") == "PENDING":
                return JSONResponse(
                    status_code=409,
                    content={
                        "contractVersion": settings.contract_version,
                        "requestId": request_id,
                        "error": {
                            "code": "IDEMPOTENCY_IN_PROGRESS",
                            "message": "该requestId的副作用正在恢复或执行中，请稍后重试",
                            "retryable": True,
                            "retryAfterMs": 1000,
                            "details": None,
                        },
                    },
                )
            return JSONResponse(status_code=existing["status_code"], content=existing["response"])
        reserved = service.db.reserve_idempotency(request_id, operation, payload_hash, status_code)
        if not reserved:
            existing = service.db.get_idempotency(request_id)
            if existing and existing.get("state") == "PENDING":
                raise RagError("IDEMPOTENCY_IN_PROGRESS", "该requestId的副作用正在执行中，请稍后重试", 409, True)
            raise RagError("IDEMPOTENCY_CONFLICT", "无法保留requestId", 409)
        try:
            value = handler()
            if inspect.isawaitable(value):
                value = await value
            content = serialized(value)
            service.db.save_idempotency(request_id, operation, payload_hash, status_code, content)
            return JSONResponse(status_code=status_code, content=content)
        except Exception:
            service.db.delete_idempotency(request_id)
            raise


@app.get("/rag-api/v1/health", dependencies=[Depends(require_auth)])
async def health(request: Request):
    service = service_from(request)
    active = service.active_release()
    providers = service.provider_status()
    if not active:
        status = "NOT_READY"
    elif any(item.get("error") and not item.get("fallback", False) for item in providers.values() if isinstance(item, dict)):
        status = "DEGRADED"
    else:
        status = "READY"
    return {
        "status": status,
        "contractVersion": settings.contract_version,
        "serviceVersion": settings.service_version,
        "knowledgeVersion": active.get("knowledge_version") if active else None,
        "releaseId": active.get("release_id") if active else None,
        "timestamp": now_iso(),
        "providers": providers,
    }


@app.get("/rag-admin-api/v1/system/providers", dependencies=[Depends(require_auth)])
async def providers(request: Request):
    return service_from(request).provider_status()


@app.get("/rag-admin-api/v1/system/chunk-policies", dependencies=[Depends(require_auth)])
async def chunk_policies():
    from .chunking import POLICY_PROFILES

    return {
        "policies": [
            {"name": name, "profile": profile, "selection": "manual_or_deterministic"}
            for name, profile in POLICY_PROFILES.items()
        ]
    }


@app.get("/rag-admin-api/v1/system/status", dependencies=[Depends(require_auth)])
async def system_status(request: Request):
    service = service_from(request)
    active = service.active_release()
    counts = service.db.system_counts()
    latest = counts.get("latestEvaluation") or {}
    phase = "READY" if active else ("INDEXING" if counts.get("readyDocuments") else "WAITING_FOR_DOCUMENT")
    if latest.get("status") == "RUNNING":
        phase = "EVALUATING"
    return {
        "phase": phase,
        "progress": {
            "completed": latest.get("completed_cases", 0),
            "total": latest.get("total_cases", 0),
            "accuracy": (latest.get("summary") or {}).get("accuracy"),
        },
        "counts": counts,
        "activeRelease": active,
        "updatedAt": now_iso(),
    }


@app.get("/rag-admin-api/v1/negative-cases", dependencies=[Depends(require_auth)])
async def negative_cases(request: Request, knowledge_base_id: str = "main-business-kb"):
    return {"negativeCases": service_from(request).db.list_negative_cases(knowledge_base_id)}


@app.post("/rag-api/v1/query", response_model=QueryResponse, dependencies=[Depends(require_auth)])
async def formal_query(request: Request, body: QueryRequest, x_request_id: str | None = Header(default=None)):
    service = service_from(request)
    request_id = request_id_from(body, x_request_id)
    payload = body.model_dump(mode="json")
    return await execute_idempotent(
        service,
        request_id=request_id,
        operation="formal_query",
        payload=payload,
        status_code=200,
        handler=lambda: service.formal_query(body),
    )


@app.post("/rag-admin-api/v1/documents", response_model=DocumentResponse, status_code=202, dependencies=[Depends(require_auth)])
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    metadata: str = Form("{}"),
    x_request_id: str | None = Header(default=None),
):
    service = service_from(request)
    try:
        metadata_payload = json.loads(metadata or "{}")
    except json.JSONDecodeError as exc:
        raise RagError("INVALID_REQUEST", "metadata必须是JSON", 400) from exc
    if not isinstance(metadata_payload, dict):
        raise RagError("INVALID_REQUEST", "metadata必须是对象", 400)
    try:
        metadata_payload = UploadMetadata.model_validate(metadata_payload).model_dump(mode="json", exclude_none=True)
    except Exception as exc:
        raise RagError("INVALID_REQUEST", "metadata字段不合法", 422, details=str(exc)) from exc
    request_id = str(metadata_payload.get("requestId") or x_request_id or "upload-" + uuid.uuid4().hex)
    if x_request_id and metadata_payload.get("requestId") and x_request_id != metadata_payload["requestId"]:
        raise RagError("INVALID_REQUEST", "Header与metadata.requestId不一致", 400)
    content = await file.read(service.settings.max_file_bytes + 1)
    payload = {"filename": file.filename, "size": len(content), "sha256": hashlib.sha256(content).hexdigest(), "metadata": metadata_payload}
    return await execute_idempotent(
        service,
        request_id=request_id,
        operation="upload_document",
        payload=payload,
        status_code=202,
        handler=lambda: service.accept_upload(
            content=content,
            filename=file.filename or "uploaded.bin",
            mime_type=file.content_type,
            metadata=metadata_payload,
            request_id=request_id,
        ),
    )


@app.get("/rag-admin-api/v1/documents", dependencies=[Depends(require_auth)])
async def list_documents(request: Request, knowledge_base_id: str = "main-business-kb"):
    return {"documents": [item.model_dump(mode="json") for item in service_from(request).list_documents(knowledge_base_id)]}


@app.get("/rag-admin-api/v1/documents/{document_id}", dependencies=[Depends(require_auth)])
async def document_status(request: Request, document_id: str, version: int | None = None):
    return service_from(request).get_document(document_id, version).model_dump(mode="json")


@app.get("/rag-admin-api/v1/documents/{document_id}/chunks", dependencies=[Depends(require_auth)])
async def document_chunks(request: Request, document_id: str, version: int | None = None):
    return {"documentId": document_id, "documentVersion": version, "chunks": service_from(request).document_chunks(document_id, version)}


@app.post("/rag-admin-api/v1/test-sessions", response_model=TestSessionResponse, dependencies=[Depends(require_auth)])
async def create_test_session(request: Request, body: TestSessionCreate, x_request_id: str | None = Header(default=None)):
    service = service_from(request)
    request_id = request_id_from(body, x_request_id)
    payload = body.model_dump(mode="json")
    return await execute_idempotent(
        service,
        request_id=request_id,
        operation="create_test_session",
        payload=payload,
        status_code=200,
        handler=lambda: service.create_test_session(body),
    )


@app.get("/rag-admin-api/v1/test-sessions/{session_id}", dependencies=[Depends(require_auth)])
async def get_test_session(request: Request, session_id: str):
    service = service_from(request)
    row = service.db.get_test_session(session_id)
    if not row:
        raise RagError("RESOURCE_NOT_FOUND", "测试会话不存在", 404)
    return {
        "testSessionId": row["test_session_id"],
        "knowledgeBaseId": row["knowledge_base_id"],
        "mode": row["mode"],
        "baseReleaseId": row.get("base_release_id"),
        "candidateDocuments": row.get("candidate_documents", []),
        "status": row["status"],
        "operatorId": row.get("operator_id"),
        "createdAt": row.get("created_at"),
        "updatedAt": row.get("updated_at"),
        "answers": service.db.list_test_answers(session_id),
    }


@app.get("/rag-admin-api/v1/test-sessions", dependencies=[Depends(require_auth)])
async def list_test_sessions(request: Request, knowledge_base_id: str = "main-business-kb"):
    service = service_from(request)
    sessions = []
    for row in service.db.list_test_sessions(knowledge_base_id):
        sessions.append(
            {
                "testSessionId": row["test_session_id"],
                "knowledgeBaseId": row["knowledge_base_id"],
                "mode": row["mode"],
                "baseReleaseId": row.get("base_release_id"),
                "candidateDocuments": row.get("candidate_documents", []),
                "status": row["status"],
                "operatorId": row.get("operator_id"),
                "createdAt": row.get("created_at"),
            }
        )
    return {"sessions": sessions}


@app.post("/rag-admin-api/v1/test-sessions/{session_id}/query", response_model=TestAnswerResponse, dependencies=[Depends(require_auth)])
async def test_query(request: Request, session_id: str, body: TestQueryRequest, x_request_id: str | None = Header(default=None)):
    service = service_from(request)
    request_id = request_id_from(body, x_request_id)
    payload = {"sessionId": session_id, **body.model_dump(mode="json")}
    return await execute_idempotent(
        service,
        request_id=request_id,
        operation="test_query",
        payload=payload,
        status_code=200,
        handler=lambda: service.test_query(session_id, body),
    )


@app.put("/rag-admin-api/v1/test-sessions/{session_id}/answers/{answer_id}/decision", dependencies=[Depends(require_auth)])
async def answer_decision(request: Request, session_id: str, answer_id: str, body: DecisionRequest):
    service = service_from(request)
    decision = body
    answer = service.db.get_test_answer(answer_id)
    if answer is None:
        raise RagError("RESOURCE_NOT_FOUND", "测试答案不存在", 404)
    if answer.get("test_session_id") != session_id:
        raise RagError("RESOURCE_NOT_FOUND", "测试答案不属于该会话", 404)
    payload = {"sessionId": session_id, "answerId": answer_id, **decision.model_dump(mode="json")}
    return await execute_idempotent(
        service,
        request_id=decision.requestId,
        operation="answer_decision",
        payload=payload,
        status_code=200,
        handler=lambda: _update_decision(service, answer_id, decision),
    )


async def _update_decision(service: RagService, answer_id: str, decision: DecisionRequest) -> dict[str, Any]:
    return service.update_test_decision(answer_id, decision.model_dump(mode="json"))


@app.get("/rag-admin-api/v1/test-sessions/{session_id}/answers", dependencies=[Depends(require_auth)])
async def test_answers(request: Request, session_id: str):
    if not service_from(request).db.get_test_session(session_id):
        raise RagError("RESOURCE_NOT_FOUND", "测试会话不存在", 404)
    return {"answers": service_from(request).db.list_test_answers(session_id)}


@app.post("/rag-admin-api/v1/knowledge-bases/{knowledge_base_id}/releases", response_model=ReleaseResponse, status_code=202, dependencies=[Depends(require_auth)])
async def create_release(request: Request, knowledge_base_id: str, body: ReleaseCreate, x_request_id: str | None = Header(default=None)):
    service = service_from(request)
    session = service.db.get_test_session(body.testSessionId)
    if not session:
        raise RagError("RESOURCE_NOT_FOUND", "测试会话不存在", 404)
    if session.get("knowledge_base_id") != knowledge_base_id:
        raise RagError("INVALID_REQUEST", "知识库与测试会话不一致", 422)
    request_id = request_id_from(body, x_request_id)
    payload = {"knowledgeBaseId": knowledge_base_id, **body.model_dump(mode="json")}
    return await execute_idempotent(
        service,
        request_id=request_id,
        operation="create_release",
        payload=payload,
        status_code=202,
        handler=lambda: service.start_release(body),
    )


@app.get("/rag-admin-api/v1/releases", dependencies=[Depends(require_auth)])
async def list_releases(request: Request, knowledge_base_id: str = "main-business-kb"):
    service = service_from(request)
    return {"active": service.active_release(knowledge_base_id), "releases": [item.model_dump(mode="json") for item in service.list_releases(knowledge_base_id)]}


@app.get("/rag-admin-api/v1/releases/{release_id}", dependencies=[Depends(require_auth)])
async def release_status(request: Request, release_id: str):
    service = service_from(request)
    result = service.release(release_id).model_dump(mode="json")
    result["manifest"] = (service.db.get_release(release_id) or {}).get("manifest", {})
    result["events"] = service.db.list_release_events(release_id)
    return result


@app.get("/rag-admin-api/v1/releases/{release_id}/members", dependencies=[Depends(require_auth)])
async def release_members(request: Request, release_id: str):
    if not service_from(request).db.get_release(release_id):
        raise RagError("RESOURCE_NOT_FOUND", "发布版本不存在", 404)
    members = service_from(request).db.get_release_members(release_id)
    return {
        "releaseId": release_id,
        "members": [
            {
                key: value
                for key, value in item.items()
                if key not in {"embedding_blob", "embedding_dim"}
            }
            for item in members
        ],
    }


@app.post("/rag-admin-api/v1/releases/{release_id}/rollback", response_model=ReleaseResponse, dependencies=[Depends(require_auth)])
async def rollback_release(request: Request, release_id: str, body: RollbackRequest):
    if body.targetReleaseId != release_id:
        raise RagError("INVALID_REQUEST", "路径版本与targetReleaseId不一致", 400)
    service = service_from(request)
    payload = {"releaseId": release_id, **body.model_dump(mode="json")}
    return await execute_idempotent(
        service,
        request_id=body.requestId,
        operation="rollback_release",
        payload=payload,
        status_code=200,
        handler=lambda: service.rollback(
            release_id,
            request_id=body.requestId,
            operator_id=body.operatorId,
            note=body.note,
        ),
    )


@app.post("/rag-admin-api/v1/debug/query", response_model=QueryResponse, dependencies=[Depends(require_auth)])
async def debug_query(request: Request, body: DebugQueryRequest):
    return (await service_from(request).debug_query(body)).model_dump(mode="json")


@app.post("/rag-admin-api/v1/debug/retrieve", dependencies=[Depends(require_auth)])
async def debug_retrieve(request: Request, body: DebugQueryRequest):
    return await service_from(request).debug_retrieve(body)


@app.get("/rag-admin-api/v1/traces", dependencies=[Depends(require_auth)])
async def list_traces(request: Request, limit: int = 50, offset: int = 0):
    return {"traces": service_from(request).db.list_traces(min(200, max(1, limit)), max(0, offset))}


@app.get("/rag-admin-api/v1/traces/{trace_id}", dependencies=[Depends(require_auth)])
async def trace_detail(request: Request, trace_id: str):
    return service_from(request).trace(trace_id)


@app.post("/rag-admin-api/v1/traces/{trace_id}/feedback", dependencies=[Depends(require_auth)])
async def trace_feedback(request: Request, trace_id: str, body: TraceFeedbackRequest):
    service = service_from(request)
    if not service.db.get_trace(trace_id):
        raise RagError("RESOURCE_NOT_FOUND", "Trace不存在", 404)
    data = {
        "feedback_id": "feedback-" + uuid.uuid4().hex[:16],
        "trace_id": trace_id,
        "request_id": body.requestId,
        "rating": body.rating,
        "note": str(body.note or "")[:4000],
        "reviewer_id": body.reviewerId,
        "tags": body.tags,
    }
    payload = {"traceId": trace_id, **body.model_dump(mode="json")}
    return await execute_idempotent(
        service,
        request_id=body.requestId,
        operation="trace_feedback",
        payload=payload,
        status_code=200,
        handler=lambda: service.db.save_feedback(data),
    )


@app.get("/rag-admin-api/v1/evaluations", dependencies=[Depends(require_auth)])
async def list_evaluations(request: Request):
    return {"evaluations": service_from(request).db.list_evaluation_runs()}


@app.get("/rag-admin-api/v1/evaluations/{eval_run_id}", dependencies=[Depends(require_auth)])
async def evaluation_detail(request: Request, eval_run_id: str):
    result = service_from(request).db.get_evaluation_run(eval_run_id)
    if not result:
        raise RagError("RESOURCE_NOT_FOUND", "评测运行不存在", 404)
    return result


@app.post("/rag-admin-api/v1/evaluations/run", status_code=202, dependencies=[Depends(require_auth)])
async def run_evaluation(request: Request, body: EvaluationRunRequest):
    service = service_from(request)
    cases = body.cases
    if cases is None:
        cases_path = Path(__file__).resolve().parents[1] / "eval" / "questions-200.json"
        if not cases_path.exists():
            raise RagError("RESOURCE_NOT_FOUND", "默认200题评测集不存在", 404)
        cases = json.loads(cases_path.read_text(encoding="utf-8"))
    payload = {"cases": cases, **body.model_dump(mode="json")}

    async def start() -> dict[str, Any]:
        eval_run_id = service.start_evaluation(
            cases,
            request_id=body.requestId,
            concurrency=body.concurrency,
            use_generation=body.useGeneration,
        )
        return {"evalRunId": eval_run_id, "status": "RUNNING", "totalCases": len(cases)}

    return await execute_idempotent(
        service,
        request_id=body.requestId,
        operation="start_evaluation",
        payload=payload,
        status_code=202,
        handler=start,
    )


static_dir = Path(__file__).resolve().parent.parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

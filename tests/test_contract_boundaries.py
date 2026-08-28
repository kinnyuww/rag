from __future__ import annotations

import asyncio
import csv
import io
from pathlib import Path

import pytest

from app.config import Settings
from app.models import (
    DecisionRequest,
    ReleaseCreate,
    Snapshot,
    SourceInfo,
    TestCandidate as CandidatePayload,
    TestQueryRequest as SessionQueryPayload,
    TestSessionCreate as SessionPayload,
)
from app.pipeline import RagError, RagService
from app.tracing import safe_summary


def settings_for(tmp_path: Path) -> Settings:
    return Settings(
        auth_enabled=False,
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "data" / "uploads",
        model_cache_dir=tmp_path / "models",
        sqlite_path=tmp_path / "data" / "rag.db",
        qdrant_path=tmp_path / "data" / "qdrant",
        embedding_provider="hash",
        reranker_provider="lexical",
        generation_provider="deterministic",
        deep_agent_enabled=False,
        query_rewrite_enabled=False,
    )


def csv_bytes(question: str = "如何注册？") -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["标准问题", "标准答案", "关键词", "大类", "小类"])
    writer.writerow([question, "请携带证件到窗口办理。", "证件,窗口", "sample-kb", "测试"])
    return output.getvalue().encode()


def test_trace_redacts_short_secrets():
    summary = safe_summary("sk-abcdefghijklmnop bearer very-secret-value")
    assert "abcdefghijklmnop" not in str(summary)
    assert "very-secret-value" not in str(summary)


async def wait_ready(service: RagService, document_id: str) -> None:
    for _ in range(100):
        if service.get_document(document_id).status != "PROCESSING":
            break
        await asyncio.sleep(0.02)
    assert service.get_document(document_id).status == "READY_FOR_TEST"


@pytest.mark.asyncio
async def test_scope_isolation_and_release_uses_session_snapshot(tmp_path: Path):
    settings = settings_for(tmp_path)
    settings.prepare_directories()
    service = RagService(settings)
    try:
        a = await service.accept_upload(
            content=csv_bytes("A知识如何办理？"),
            filename="a.csv",
            mime_type="text/csv",
            metadata={"knowledgeBaseId": "kb-a", "title": "A"},
            request_id="upload-a",
        )
        b = await service.accept_upload(
            content=csv_bytes("B知识如何办理？"),
            filename="b.csv",
            mime_type="text/csv",
            metadata={"knowledgeBaseId": "kb-b", "title": "B"},
            request_id="upload-b",
        )
        await wait_ready(service, a.documentId)
        await wait_ready(service, b.documentId)
        with pytest.raises(RagError) as cross:
            service.create_test_session(
                SessionPayload(
                    requestId="cross-session",
                    knowledgeBaseId="kb-a",
                    candidateDocuments=[CandidatePayload(documentId=b.documentId, documentVersion=1)],
                )
            )
        assert cross.value.code in {"DOCUMENT_VERSION_NOT_READY", "INVALID_REQUEST"}

        session = service.create_test_session(
            SessionPayload(
                requestId="valid-session",
                knowledgeBaseId="kb-a",
                candidateDocuments=[CandidatePayload(documentId=a.documentId, documentVersion=1)],
            )
        )
        with pytest.raises(RagError) as mismatch:
            await service.start_release(
                ReleaseCreate(
                    requestId="mismatch-release",
                    testSessionId=session.testSessionId,
                    candidateDocuments=[CandidatePayload(documentId=b.documentId, documentVersion=1)],
                )
            )
        assert mismatch.value.code == "DOCUMENT_VERSION_CHANGED"
        with pytest.raises(RagError) as untested:
            await service.start_release(
                ReleaseCreate(requestId="untested-release", testSessionId=session.testSessionId)
            )
        assert untested.value.code == "RELEASE_REQUIRES_TEST"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_disabled_case_can_be_reenabled_and_trace_candidates_are_bounded(tmp_path: Path):
    settings = settings_for(tmp_path)
    settings.prepare_directories()
    service = RagService(settings)
    try:
        upload = await service.accept_upload(
            content=csv_bytes("测试问题？"),
            filename="test.csv",
            mime_type="text/csv",
            metadata={"knowledgeBaseId": "main-business-kb", "title": "Test"},
            request_id="upload-boundary",
        )
        await wait_ready(service, upload.documentId)
        session = service.create_test_session(
            SessionPayload(
                requestId="session-boundary",
                candidateDocuments=[CandidatePayload(documentId=upload.documentId, documentVersion=1)],
            )
        )
        answer = await service.test_query(
            session.testSessionId,
            SessionQueryPayload(requestId="answer-boundary", question="测试问题？"),
        )
        disabled = service.update_test_decision(
            answer.answerId,
            DecisionRequest(
                requestId="disable-boundary",
                decision="DISABLED",
                reasonCode="ANSWER_INACCURATE",
                note="需要修正",
            ).model_dump(mode="json"),
        )
        assert disabled["decision"] == "DISABLED"
        assert service.db.list_negative_cases()
        enabled = service.update_test_decision(
            answer.answerId,
            DecisionRequest(requestId="enable-boundary", decision="ENABLED").model_dump(mode="json"),
        )
        assert enabled["decision"] == "ENABLED"
        assert service.db.list_negative_cases() == []
        trace = service.trace(answer.traceId)
        assert trace["spans"][0]["status"] == "OK"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_single_document_scope_excludes_active_release_and_duplicate_eval_ids_fail(tmp_path: Path):
    settings = settings_for(tmp_path)
    settings.prepare_directories()
    service = RagService(settings)
    try:
        upload = await service.accept_upload(
            content=csv_bytes("候选问题？"),
            filename="candidate.csv",
            mime_type="text/csv",
            metadata={"knowledgeBaseId": "main-business-kb", "title": "Candidate"},
            request_id="upload-single",
        )
        await wait_ready(service, upload.documentId)
        session = service.create_test_session(
            SessionPayload(
                requestId="single-session",
                mode="SINGLE_DOCUMENT",
                candidateDocuments=[CandidatePayload(documentId=upload.documentId, documentVersion=1)],
            )
        )
        prepared = service._prepare_scope(
            knowledge_base_id="main-business-kb", test_session_id=session.testSessionId
        )
        assert prepared.scope.mode == "TEST"
        assert prepared.scope.chunk_ids
        assert prepared.release["release_id"] is None
        with pytest.raises(RagError) as duplicate:
            service.start_evaluation(
                [{"id": "same", "question": "a"}, {"id": "same", "question": "b"}],
                request_id="duplicate-eval",
            )
        assert duplicate.value.code == "INVALID_REQUEST"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_disabled_source_blocks_paraphrase_and_rollback(tmp_path: Path):
    settings = settings_for(tmp_path)
    settings.prepare_directories()
    service = RagService(settings)
    try:
        upload = await service.accept_upload(
            content=csv_bytes("测试问题如何办理？"),
            filename="negative.csv",
            mime_type="text/csv",
            metadata={"knowledgeBaseId": "main-business-kb", "title": "Negative"},
            request_id="upload-negative",
        )
        await wait_ready(service, upload.documentId)
        session = service.create_test_session(
            SessionPayload(
                requestId="session-negative",
                candidateDocuments=[CandidatePayload(documentId=upload.documentId, documentVersion=1)],
            )
        )
        answer = await service.test_query(
            session.testSessionId,
            SessionQueryPayload(requestId="answer-negative", question="测试问题如何办理？"),
        )
        release = await service.start_release(
            ReleaseCreate(requestId="release-negative", testSessionId=session.testSessionId)
        )
        for _ in range(100):
            await asyncio.sleep(0.02)
            if service.release(release.releaseId).status != "BUILDING":
                break
        assert service.release(release.releaseId).status == "PUBLISHED"
        service.update_test_decision(
            answer.answerId,
            DecisionRequest(
                requestId="disable-negative",
                decision="DISABLED",
                reasonCode="ANSWER_INACCURATE",
                note="错误答案",
            ).model_dump(mode="json"),
        )
        blocked = await service.run_query(
            question="这个测试问题怎么办理？",
            snapshot=Snapshot(conversationKey="negative", conversationVersion=1, inputFingerprint="sha256:n"),
            source=SourceInfo(),
            knowledge_base_id="main-business-kb",
            trace_id="trace-negative-paraphrase",
            request_id="query-negative-paraphrase",
            generate=False,
        )
        assert blocked.result == "NO_ANSWER"
        assert blocked.reasonCode == "DISABLED_TEST_CASE"
        with pytest.raises(RagError) as rollback:
            await service.rollback(release.releaseId, request_id="rollback-negative")
        assert rollback.value.code == "DISABLED_CASE_VALIDATION_FAILED"
    finally:
        await service.close()

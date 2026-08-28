from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .models import now_iso


class DatabaseConflict(RuntimeError):
    pass


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


class Database:
    """Small SQLite repository; large source files remain on disk."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock, self.connection() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def initialize(self) -> None:
        with self._lock, self.connection() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode = WAL;
                PRAGMA synchronous = NORMAL;

                CREATE TABLE IF NOT EXISTS service_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    knowledge_base_id TEXT NOT NULL,
                    current_version INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    mime_type TEXT,
                    category TEXT,
                    source_department TEXT,
                    source_owner TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS document_versions (
                    document_id TEXT NOT NULL,
                    document_version INTEGER NOT NULL,
                    knowledge_base_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    mime_type TEXT,
                    byte_size INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    parser TEXT,
                    chunk_policy TEXT,
                    processing_result_json TEXT NOT NULL DEFAULT '{}',
                    error_json TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (document_id, document_version),
                    FOREIGN KEY (document_id) REFERENCES documents(document_id)
                );
                CREATE INDEX IF NOT EXISTS idx_document_versions_status
                    ON document_versions(knowledge_base_id, status);
                CREATE INDEX IF NOT EXISTS idx_document_versions_hash
                    ON document_versions(content_hash);

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    document_version INTEGER NOT NULL,
                    ordinal INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    title TEXT NOT NULL,
                    section_path_json TEXT NOT NULL DEFAULT '[]',
                    location_json TEXT NOT NULL DEFAULT '{}',
                    content_hash TEXT NOT NULL,
                    lexical_text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    embedding_blob BLOB,
                    embedding_dim INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (document_id, document_version)
                        REFERENCES document_versions(document_id, document_version)
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_document
                    ON chunks(document_id, document_version, ordinal);

                CREATE TABLE IF NOT EXISTS releases (
                    release_id TEXT PRIMARY KEY,
                    knowledge_base_id TEXT NOT NULL,
                    knowledge_version TEXT,
                    status TEXT NOT NULL,
                    base_release_id TEXT,
                    manifest_json TEXT NOT NULL DEFAULT '{}',
                    published_by TEXT,
                    publish_note TEXT,
                    enabled_test_case_count INTEGER NOT NULL DEFAULT 0,
                    disabled_test_case_count INTEGER NOT NULL DEFAULT 0,
                    error_json TEXT,
                    created_at TEXT NOT NULL,
                    published_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_releases_kb_created
                    ON releases(knowledge_base_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS release_events (
                    event_id TEXT PRIMARY KEY,
                    release_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    request_id TEXT,
                    operator_id TEXT,
                    note TEXT,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (release_id) REFERENCES releases(release_id)
                );

                CREATE TABLE IF NOT EXISTS release_members (
                    release_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    document_version INTEGER NOT NULL,
                    PRIMARY KEY (release_id, chunk_id),
                    FOREIGN KEY (release_id) REFERENCES releases(release_id),
                    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
                );
                CREATE INDEX IF NOT EXISTS idx_release_members_chunk
                    ON release_members(release_id, document_id, document_version);

                CREATE TABLE IF NOT EXISTS active_releases (
                    knowledge_base_id TEXT PRIMARY KEY,
                    release_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (release_id) REFERENCES releases(release_id)
                );

                CREATE TABLE IF NOT EXISTS test_sessions (
                    test_session_id TEXT PRIMARY KEY,
                    knowledge_base_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    base_release_id TEXT,
                    candidate_documents_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL,
                    operator_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS test_answers (
                    answer_id TEXT PRIMARY KEY,
                    test_session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    context_json TEXT NOT NULL DEFAULT '[]',
                    result TEXT NOT NULL,
                    answer_json TEXT,
                    source_references_json TEXT NOT NULL DEFAULT '[]',
                    grounding_json TEXT,
                    diagnostics_json TEXT NOT NULL DEFAULT '{}',
                    reason_code TEXT,
                    decision TEXT NOT NULL DEFAULT 'ENABLED',
                    decision_reason_code TEXT,
                    decision_note TEXT,
                    decision_version INTEGER NOT NULL DEFAULT 1,
                    operator_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (test_session_id) REFERENCES test_sessions(test_session_id)
                );
                CREATE INDEX IF NOT EXISTS idx_test_answers_session
                    ON test_answers(test_session_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS negative_cases (
                    negative_case_id TEXT PRIMARY KEY,
                    answer_id TEXT NOT NULL,
                    knowledge_base_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    rejected_answer TEXT,
                    expected_correction TEXT,
                    scope_json TEXT NOT NULL DEFAULT '{}',
                    blocked_chunk_ids_json TEXT NOT NULL DEFAULT '[]',
                    reason_code TEXT,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (answer_id) REFERENCES test_answers(answer_id)
                );

                CREATE TABLE IF NOT EXISTS idempotency_keys (
                    request_id TEXT PRIMARY KEY,
                    operation TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    response_json TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'COMPLETED',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trace_runs (
                    trace_id TEXT PRIMARY KEY,
                    request_id TEXT,
                    business_trace_id TEXT,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_summary TEXT NOT NULL DEFAULT '{}',
                    output_summary TEXT NOT NULL DEFAULT '{}',
                    attributes_json TEXT NOT NULL DEFAULT '{}',
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    duration_ms INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_trace_runs_started
                    ON trace_runs(started_at DESC);

                CREATE TABLE IF NOT EXISTS trace_spans (
                    span_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    parent_span_id TEXT,
                    name TEXT NOT NULL,
                    stage_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_summary TEXT NOT NULL DEFAULT '{}',
                    output_summary TEXT NOT NULL DEFAULT '{}',
                    attributes_json TEXT NOT NULL DEFAULT '{}',
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    duration_ms INTEGER,
                    error_code TEXT,
                    error_message TEXT,
                    FOREIGN KEY (trace_id) REFERENCES trace_runs(trace_id)
                );
                CREATE INDEX IF NOT EXISTS idx_trace_spans_trace
                    ON trace_spans(trace_id, started_at);

                CREATE TABLE IF NOT EXISTS trace_candidates (
                    trace_id TEXT NOT NULL,
                    span_id TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    chunk_id TEXT NOT NULL,
                    document_id TEXT,
                    document_version INTEGER,
                    retrieval_score REAL,
                    vector_score REAL,
                    lexical_score REAL,
                    reranker_score REAL,
                    reranker_score_normalized REAL,
                    selected INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (span_id, rank),
                    FOREIGN KEY (trace_id) REFERENCES trace_runs(trace_id),
                    FOREIGN KEY (span_id) REFERENCES trace_spans(span_id)
                );

                CREATE TABLE IF NOT EXISTS trace_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    note TEXT,
                    reviewer_id TEXT NOT NULL,
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (trace_id) REFERENCES trace_runs(trace_id)
                );
                CREATE INDEX IF NOT EXISTS idx_trace_feedback_trace
                    ON trace_feedback(trace_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    eval_run_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_cases INTEGER NOT NULL DEFAULT 0,
                    completed_cases INTEGER NOT NULL DEFAULT 0,
                    dataset_hash TEXT,
                    release_id TEXT,
                    config_json TEXT NOT NULL DEFAULT '{}',
                    summary_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evaluation_cases (
                    eval_run_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    category TEXT,
                    question TEXT,
                    expected_result TEXT,
                    expected_json TEXT NOT NULL DEFAULT '{}',
                    actual_result TEXT,
                    passed INTEGER NOT NULL DEFAULT 0,
                    assertions_json TEXT NOT NULL DEFAULT '[]',
                    trace_id TEXT,
                    response_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (eval_run_id, case_id),
                    FOREIGN KEY (eval_run_id) REFERENCES evaluation_runs(eval_run_id)
                );
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
            if "metadata_json" not in columns:
                conn.execute("ALTER TABLE chunks ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'" )
            evaluation_columns = {row[1] for row in conn.execute("PRAGMA table_info(evaluation_runs)").fetchall()}
            if "dataset_hash" not in evaluation_columns:
                conn.execute("ALTER TABLE evaluation_runs ADD COLUMN dataset_hash TEXT")
            if "release_id" not in evaluation_columns:
                conn.execute("ALTER TABLE evaluation_runs ADD COLUMN release_id TEXT")
            if "config_json" not in evaluation_columns:
                conn.execute("ALTER TABLE evaluation_runs ADD COLUMN config_json TEXT NOT NULL DEFAULT '{}'")
            case_columns = {row[1] for row in conn.execute("PRAGMA table_info(evaluation_cases)").fetchall()}
            if "category" not in case_columns:
                conn.execute("ALTER TABLE evaluation_cases ADD COLUMN category TEXT")
            if "question" not in case_columns:
                conn.execute("ALTER TABLE evaluation_cases ADD COLUMN question TEXT")
            if "expected_json" not in case_columns:
                conn.execute("ALTER TABLE evaluation_cases ADD COLUMN expected_json TEXT NOT NULL DEFAULT '{}'")
            if "assertions_json" not in case_columns:
                conn.execute("ALTER TABLE evaluation_cases ADD COLUMN assertions_json TEXT NOT NULL DEFAULT '[]'")
            idempotency_columns = {row[1] for row in conn.execute("PRAGMA table_info(idempotency_keys)").fetchall()}
            if "state" not in idempotency_columns:
                conn.execute("ALTER TABLE idempotency_keys ADD COLUMN state TEXT NOT NULL DEFAULT 'COMPLETED'")
            conn.commit()

    @staticmethod
    def row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row else None

    def get_meta(self, key: str) -> str | None:
        with self.connection() as conn:
            row = conn.execute("SELECT value FROM service_meta WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO service_meta(key,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",
                (key, value, now_iso()),
            )

    def find_document_by_hash(self, knowledge_base_id: str, content_hash: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM document_versions WHERE knowledge_base_id=? AND content_hash=? "
                "ORDER BY created_at DESC LIMIT 1",
                (knowledge_base_id, content_hash),
            ).fetchone()
            return self.row(row)

    def create_document_version(
        self,
        *,
        document_id: str,
        knowledge_base_id: str,
        title: str,
        filename: str,
        mime_type: str | None,
        byte_size: int,
        content_hash: str,
        source_path: str,
        metadata: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        now = now_iso()
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT * FROM document_versions WHERE knowledge_base_id=? AND content_hash=?",
                (knowledge_base_id, content_hash),
            ).fetchone()
            if existing:
                return dict(existing), True
            parent = conn.execute(
                "SELECT current_version FROM documents WHERE document_id=?",
                (document_id,),
            ).fetchone()
            version = int(parent["current_version"] + 1) if parent else 1
            if parent:
                conn.execute(
                    "UPDATE documents SET current_version=?,title=?,filename=?,mime_type=?,"
                    "updated_at=? WHERE document_id=?",
                    (version, title, filename, mime_type, now, document_id),
                )
            else:
                conn.execute(
                    "INSERT INTO documents(document_id,knowledge_base_id,current_version,title,filename,"
                    "mime_type,category,source_department,source_owner,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        document_id,
                        knowledge_base_id,
                        version,
                        title,
                        filename,
                        mime_type,
                        metadata.get("category"),
                        metadata.get("sourceDepartment"),
                        metadata.get("sourceOwner"),
                        now,
                        now,
                    ),
                )
            conn.execute(
                "INSERT INTO document_versions(document_id,document_version,knowledge_base_id,title,filename,"
                "mime_type,byte_size,content_hash,source_path,status,progress,metadata_json,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    document_id,
                    version,
                    knowledge_base_id,
                    title,
                    filename,
                    mime_type,
                    byte_size,
                    content_hash,
                    source_path,
                    "PROCESSING",
                    0,
                    dumps(metadata),
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM document_versions WHERE document_id=? AND document_version=?",
                (document_id, version),
            ).fetchone()
            return dict(row), False

    def get_document_version(self, document_id: str, version: int | None = None) -> dict[str, Any] | None:
        with self.connection() as conn:
            if version is None:
                row = conn.execute(
                    "SELECT d.*,v.* FROM documents d JOIN document_versions v "
                    "ON d.document_id=v.document_id AND d.current_version=v.document_version "
                    "WHERE d.document_id=?",
                    (document_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM document_versions WHERE document_id=? AND document_version=?",
                    (document_id, version),
                ).fetchone()
            return self.row(row)

    def list_documents(self, knowledge_base_id: str = "main-business-kb") -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT d.*,v.status,v.progress,v.processing_result_json,v.error_json,v.chunk_policy,"
                "v.content_hash,v.byte_size,v.parser,v.document_version,v.created_at AS version_created_at "
                "FROM documents d JOIN document_versions v ON d.document_id=v.document_id "
                "AND d.current_version=v.document_version WHERE d.knowledge_base_id=? "
                "ORDER BY d.updated_at DESC",
                (knowledge_base_id,),
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["processing_result"] = loads(item.pop("processing_result_json"), {})
                item["error"] = loads(item.pop("error_json"), None)
                result.append(item)
            return result

    def update_document_version(
        self,
        document_id: str,
        version: int,
        *,
        status: str | None = None,
        progress: int | None = None,
        parser: str | None = None,
        chunk_policy: str | None = None,
        processing_result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        fields: list[str] = []
        values: list[Any] = []
        if status is not None:
            fields.append("status=?")
            values.append(status)
        if progress is not None:
            fields.append("progress=?")
            values.append(max(0, min(100, progress)))
        if parser is not None:
            fields.append("parser=?")
            values.append(parser)
        if chunk_policy is not None:
            fields.append("chunk_policy=?")
            values.append(chunk_policy)
        if processing_result is not None:
            fields.append("processing_result_json=?")
            values.append(dumps(processing_result))
        if error is not None:
            fields.append("error_json=?")
            values.append(dumps(error))
        elif status in {"PROCESSING", "READY_FOR_TEST"}:
            fields.append("error_json=NULL")
        fields.append("updated_at=?")
        values.append(now_iso())
        values.extend([document_id, version])
        with self.transaction() as conn:
            conn.execute(
                f"UPDATE document_versions SET {','.join(fields)} "
                "WHERE document_id=? AND document_version=?",
                values,
            )
            row = conn.execute(
                "SELECT * FROM document_versions WHERE document_id=? AND document_version=?",
                (document_id, version),
            ).fetchone()
            return self.row(row)

    def insert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        with self.transaction() as conn:
            first = chunks[0]
            conn.execute(
                "DELETE FROM chunks WHERE document_id=? AND document_version=?",
                (first["document_id"], first["document_version"]),
            )
            for chunk in chunks:
                conn.execute(
                    "INSERT INTO chunks(chunk_id,document_id,document_version,ordinal,text,title,"
                    "section_path_json,location_json,content_hash,lexical_text,metadata_json,embedding_blob,embedding_dim,created_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        chunk["chunk_id"],
                        chunk["document_id"],
                        chunk["document_version"],
                        chunk["ordinal"],
                        chunk["text"],
                        chunk.get("title", ""),
                        dumps(chunk.get("section_path", [])),
                        dumps(chunk.get("location", {})),
                        chunk["content_hash"],
                        chunk.get("lexical_text", chunk["text"]),
                        dumps(chunk.get("metadata", {})),
                        chunk.get("embedding_blob"),
                        chunk.get("embedding_dim"),
                        now_iso(),
                    ),
                )

    def get_chunks_by_ids(self, chunk_ids: list[str], knowledge_base_id: str | None = None) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        with self.connection() as conn:
            if knowledge_base_id:
                rows = conn.execute(
                    f"SELECT c.* FROM chunks c JOIN document_versions v ON "
                    "v.document_id=c.document_id AND v.document_version=c.document_version "
                    f"WHERE c.chunk_id IN ({placeholders}) AND v.knowledge_base_id=?",
                    [*chunk_ids, knowledge_base_id],
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
                    chunk_ids,
                ).fetchall()
            by_id = {row["chunk_id"]: dict(row) for row in rows}
            result = []
            for chunk_id in chunk_ids:
                if chunk_id in by_id:
                    item = by_id[chunk_id]
                    item["section_path"] = loads(item.pop("section_path_json"), [])
                    item["location"] = loads(item.pop("location_json"), {})
                    item["metadata"] = loads(item.pop("metadata_json"), {})
                    result.append(item)
            return result

    def get_chunks_for_document(self, document_id: str, version: int | None = None) -> list[dict[str, Any]]:
        with self.connection() as conn:
            if version is None:
                row = conn.execute(
                    "SELECT current_version FROM documents WHERE document_id=?", (document_id,)
                ).fetchone()
                version = int(row["current_version"]) if row else None
            if version is None:
                return []
            rows = conn.execute(
                "SELECT * FROM chunks WHERE document_id=? AND document_version=? ORDER BY ordinal",
                (document_id, version),
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["section_path"] = loads(item.pop("section_path_json"), [])
                item["location"] = loads(item.pop("location_json"), {})
                item["metadata"] = loads(item.pop("metadata_json"), {})
                result.append(item)
            return result

    def create_release(
        self,
        *,
        release_id: str,
        knowledge_base_id: str,
        knowledge_version: str,
        base_release_id: str | None,
        manifest: dict[str, Any],
        members: list[dict[str, Any]],
        published_by: str,
        publish_note: str,
        enabled_count: int,
        disabled_count: int,
    ) -> dict[str, Any]:
        with self.transaction() as conn:
            created = now_iso()
            conn.execute(
                "INSERT INTO releases(release_id,knowledge_base_id,knowledge_version,status,base_release_id,"
                "manifest_json,published_by,publish_note,enabled_test_case_count,disabled_test_case_count,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    release_id,
                    knowledge_base_id,
                    knowledge_version,
                    "BUILDING",
                    base_release_id,
                    dumps(manifest),
                    published_by,
                    publish_note,
                    enabled_count,
                    disabled_count,
                    created,
                ),
            )
            conn.executemany(
                "INSERT INTO release_members(release_id,chunk_id,document_id,document_version) VALUES(?,?,?,?)",
                [
                    (release_id, m["chunk_id"], m["document_id"], m["document_version"])
                    for m in members
                ],
            )
            row = conn.execute("SELECT * FROM releases WHERE release_id=?", (release_id,)).fetchone()
            return dict(row)

    def update_release(
        self,
        release_id: str,
        *,
        status: str,
        error: dict[str, Any] | None = None,
        activate: bool = False,
    ) -> dict[str, Any] | None:
        with self.transaction() as conn:
            published_at = now_iso() if status == "PUBLISHED" else None
            conn.execute(
                "UPDATE releases SET status=?,error_json=?,published_at=COALESCE(?,published_at) WHERE release_id=?",
                (status, dumps(error) if error else None, published_at, release_id),
            )
            if activate:
                if status not in {"PUBLISHED", "ROLLED_BACK"}:
                    raise ValueError("only a publishable release can be active")
                row = conn.execute(
                    "SELECT knowledge_base_id FROM releases WHERE release_id=?", (release_id,)
                ).fetchone()
                if not row:
                    raise ValueError("release not found")
                conn.execute(
                    "INSERT INTO active_releases(knowledge_base_id,release_id,updated_at) VALUES(?,?,?) "
                    "ON CONFLICT(knowledge_base_id) DO UPDATE SET release_id=excluded.release_id,"
                    "updated_at=excluded.updated_at",
                    (row["knowledge_base_id"], release_id, now_iso()),
                )
            row = conn.execute("SELECT * FROM releases WHERE release_id=?", (release_id,)).fetchone()
            return self._release_row(row)

    def publish_release_atomically(
        self,
        release_id: str,
        expected_base_release_id: str | None = None,
        *,
        request_id: str | None = None,
        operator_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Publish only if the active pointer still matches the snapshot's base."""
        with self.transaction() as conn:
            release = conn.execute("SELECT * FROM releases WHERE release_id=?", (release_id,)).fetchone()
            if not release:
                return None
            if release["status"] != "BUILDING":
                return self._release_row(release)
            active = conn.execute(
                "SELECT release_id FROM active_releases WHERE knowledge_base_id=?",
                (release["knowledge_base_id"],),
            ).fetchone()
            active_id = active["release_id"] if active else None
            if active_id != expected_base_release_id:
                error = {
                    "code": "RELEASE_BASE_CHANGED",
                    "message": "基础知识版本已变化，发布快照已失效",
                    "expectedBaseReleaseId": expected_base_release_id,
                    "actualBaseReleaseId": active_id,
                }
                conn.execute(
                    "UPDATE releases SET status='FAILED',error_json=? WHERE release_id=?",
                    (dumps(error), release_id),
                )
                return self._release_row(
                    conn.execute("SELECT * FROM releases WHERE release_id=?", (release_id,)).fetchone()
                )
            published_at = now_iso()
            conn.execute(
                "UPDATE releases SET status='PUBLISHED',published_at=?,error_json=NULL WHERE release_id=?",
                (published_at, release_id),
            )
            conn.execute(
                "INSERT INTO active_releases(knowledge_base_id,release_id,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(knowledge_base_id) DO UPDATE SET release_id=excluded.release_id,"
                "updated_at=excluded.updated_at",
                (release["knowledge_base_id"], release_id, published_at),
            )
            conn.execute(
                "INSERT INTO release_events(event_id,release_id,event_type,request_id,operator_id,note,details_json,created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    "release-event-" + uuid.uuid4().hex[:16],
                    release_id,
                    "PUBLISHED",
                    request_id,
                    operator_id or release["published_by"],
                    release["publish_note"],
                    dumps(details or {}),
                    published_at,
                ),
            )
            return self._release_row(
                conn.execute("SELECT * FROM releases WHERE release_id=?", (release_id,)).fetchone()
            )

    def clear_active_release_if(self, release_id: str) -> None:
        with self.transaction() as conn:
            conn.execute("DELETE FROM active_releases WHERE release_id=?", (release_id,))

    def rollback_release_atomically(
        self,
        release_id: str,
        *,
        request_id: str | None,
        operator_id: str | None,
        note: str,
        expected_active_release_id: str | None = None,
    ) -> dict[str, Any] | None:
        with self.transaction() as conn:
            release = conn.execute("SELECT * FROM releases WHERE release_id=?", (release_id,)).fetchone()
            if not release or release["status"] not in {"PUBLISHED", "ROLLED_BACK"}:
                return None
            active = conn.execute(
                "SELECT release_id FROM active_releases WHERE knowledge_base_id=?",
                (release["knowledge_base_id"],),
            ).fetchone()
            active_id = active["release_id"] if active else None
            if expected_active_release_id is not None and active_id != expected_active_release_id:
                raise DatabaseConflict("active release changed during rollback")
            timestamp = now_iso()
            conn.execute(
                "UPDATE releases SET status='ROLLED_BACK',published_at=COALESCE(published_at,?) WHERE release_id=?",
                (timestamp, release_id),
            )
            conn.execute(
                "INSERT INTO active_releases(knowledge_base_id,release_id,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(knowledge_base_id) DO UPDATE SET release_id=excluded.release_id,"
                "updated_at=excluded.updated_at",
                (release["knowledge_base_id"], release_id, timestamp),
            )
            conn.execute(
                "INSERT INTO release_events(event_id,release_id,event_type,request_id,operator_id,note,details_json,created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    "release-event-" + uuid.uuid4().hex[:16],
                    release_id,
                    "ROLLED_BACK",
                    request_id,
                    operator_id,
                    note,
                    "{}",
                    timestamp,
                ),
            )
            return self._release_row(
                conn.execute("SELECT * FROM releases WHERE release_id=?", (release_id,)).fetchone()
            )

    def _release_row(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if not row:
            return None
        item = dict(row)
        item["manifest"] = loads(item.pop("manifest_json"), {})
        item["error"] = loads(item.pop("error_json"), None)
        return item

    def get_release(self, release_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM releases WHERE release_id=?", (release_id,)).fetchone()
            return self._release_row(row)

    def list_releases(self, knowledge_base_id: str = "main-business-kb") -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM releases WHERE knowledge_base_id=? ORDER BY created_at DESC",
                (knowledge_base_id,),
            ).fetchall()
            return [self._release_row(row) for row in rows if row]

    def get_active_release(self, knowledge_base_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT r.* FROM active_releases a JOIN releases r ON r.release_id=a.release_id "
                "WHERE a.knowledge_base_id=? AND r.status IN ('PUBLISHED','ROLLED_BACK')",
                (knowledge_base_id,),
            ).fetchone()
            return self._release_row(row)

    def get_release_members(self, release_id: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT rm.chunk_id,rm.document_id,rm.document_version,c.text,c.title,c.section_path_json,"
                "c.location_json,c.content_hash,c.lexical_text,c.metadata_json,c.embedding_blob,c.embedding_dim "
                "FROM release_members rm JOIN chunks c ON c.chunk_id=rm.chunk_id "
                "WHERE rm.release_id=? ORDER BY rm.document_id,rm.document_version,c.ordinal",
                (release_id,),
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["section_path"] = loads(item.pop("section_path_json"), [])
                item["location"] = loads(item.pop("location_json"), {})
                item["metadata"] = loads(item.pop("metadata_json"), {})
                result.append(item)
            return result

    def update_test_session_status(self, session_id: str, status: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE test_sessions SET status=?,updated_at=? WHERE test_session_id=?",
                (status, now_iso(), session_id),
            )

    def create_test_session(self, data: dict[str, Any]) -> dict[str, Any]:
        now = now_iso()
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO test_sessions(test_session_id,knowledge_base_id,mode,base_release_id,"
                "candidate_documents_json,status,operator_id,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    data["test_session_id"],
                    data["knowledge_base_id"],
                    data["mode"],
                    data.get("base_release_id"),
                    dumps(data.get("candidate_documents", [])),
                    "TESTING",
                    data.get("operator_id", "local-operator"),
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM test_sessions WHERE test_session_id=?", (data["test_session_id"],)
            ).fetchone()
            return self._test_session_row(row)

    def _test_session_row(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if not row:
            return None
        item = dict(row)
        item["candidate_documents"] = loads(item.pop("candidate_documents_json"), [])
        return item

    def get_test_session(self, session_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM test_sessions WHERE test_session_id=?", (session_id,)
            ).fetchone()
            return self._test_session_row(row)

    def list_test_answers(self, session_id: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM test_answers WHERE test_session_id=? ORDER BY created_at DESC", (session_id,)
            ).fetchall()
            return [self._test_answer_row(row) for row in rows if row]

    def save_test_answer(self, data: dict[str, Any]) -> dict[str, Any]:
        now = now_iso()
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO test_answers(answer_id,test_session_id,request_id,trace_id,question,context_json,"
                "result,answer_json,source_references_json,grounding_json,diagnostics_json,reason_code,"
                "operator_id,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    data["answer_id"],
                    data["test_session_id"],
                    data["request_id"],
                    data["trace_id"],
                    data["question"],
                    dumps(data.get("context", [])),
                    data["result"],
                    dumps(data.get("answer")) if data.get("answer") is not None else None,
                    dumps(data.get("source_references", [])),
                    dumps(data.get("grounding")) if data.get("grounding") is not None else None,
                    dumps(data.get("diagnostics", {})),
                    data.get("reason_code"),
                    data.get("operator_id"),
                    now,
                    now,
                ),
            )
            row = conn.execute("SELECT * FROM test_answers WHERE answer_id=?", (data["answer_id"],)).fetchone()
            return self._test_answer_row(row)

    def _test_answer_row(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if not row:
            return None
        item = dict(row)
        item["context"] = loads(item.pop("context_json"), [])
        item["answer"] = loads(item.pop("answer_json"), None)
        item["source_references"] = loads(item.pop("source_references_json"), [])
        item["grounding"] = loads(item.pop("grounding_json"), None)
        item["diagnostics"] = loads(item.pop("diagnostics_json"), {})
        return item

    def get_test_answer(self, answer_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM test_answers WHERE answer_id=?", (answer_id,)).fetchone()
            return self._test_answer_row(row)

    def get_test_answer_by_request(self, session_id: str, request_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM test_answers WHERE test_session_id=? AND request_id=? "
                "ORDER BY created_at DESC LIMIT 1",
                (session_id, request_id),
            ).fetchone()
            return self._test_answer_row(row)

    def update_test_decision(self, answer_id: str, data: dict[str, Any]) -> dict[str, Any] | None:
        with self.transaction() as conn:
            row = conn.execute("SELECT * FROM test_answers WHERE answer_id=?", (answer_id,)).fetchone()
            if not row:
                return None
            session_row = conn.execute(
                "SELECT knowledge_base_id FROM test_sessions WHERE test_session_id=?",
                (row["test_session_id"],),
            ).fetchone()
            knowledge_base_id = session_row["knowledge_base_id"] if session_row else "main-business-kb"
            next_version = int(row["decision_version"]) + 1
            conn.execute(
                "UPDATE test_answers SET decision=?,decision_reason_code=?,decision_note=?,"
                "decision_version=?,operator_id=?,updated_at=? WHERE answer_id=?",
                (
                    data["decision"],
                    data.get("reason_code"),
                    data.get("note"),
                    next_version,
                    data.get("operator_id", "local-operator"),
                    now_iso(),
                    answer_id,
                ),
            )
            if data["decision"] == "ENABLED":
                conn.execute(
                    "UPDATE negative_cases SET active=0 WHERE answer_id=?",
                    (answer_id,),
                )
            else:
                answer = self._test_answer_row(row) or {}
                source_ids = [ref.get("chunkId") for ref in answer.get("source_references", []) if ref.get("chunkId")]
                rejected = (answer.get("answer") or {}).get("text", "")
                if answer.get("result") == "ANSWERED" and source_ids:
                    conn.execute(
                        "INSERT INTO negative_cases(negative_case_id,answer_id,knowledge_base_id,question,"
                        "rejected_answer,expected_correction,scope_json,blocked_chunk_ids_json,reason_code,created_at) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (
                            f"negative-{answer_id}-{next_version}",
                            answer_id,
                            knowledge_base_id,
                            answer.get("question", ""),
                            rejected,
                            data.get("note"),
                            dumps({"testSessionId": answer.get("test_session_id")}),
                            dumps(source_ids),
                            data.get("reason_code"),
                            now_iso(),
                        ),
                    )
            updated = conn.execute("SELECT * FROM test_answers WHERE answer_id=?", (answer_id,)).fetchone()
            return self._test_answer_row(updated)

    def list_negative_cases(self, knowledge_base_id: str = "main-business-kb") -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM negative_cases WHERE knowledge_base_id=? AND active=1 ORDER BY created_at",
                (knowledge_base_id,),
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["scope"] = loads(item.pop("scope_json"), {})
                item["blocked_chunk_ids"] = loads(item.pop("blocked_chunk_ids_json"), [])
                result.append(item)
            return result

    def list_test_sessions(self, knowledge_base_id: str = "main-business-kb", limit: int = 50) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM test_sessions WHERE knowledge_base_id=? ORDER BY created_at DESC LIMIT ?",
                (knowledge_base_id, limit),
            ).fetchall()
            return [self._test_session_row(row) for row in rows if row]

    def list_processing_documents(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM document_versions WHERE status IN ('PROCESSING','UPLOADED') ORDER BY created_at"
            ).fetchall()
            return [dict(row) for row in rows]

    def list_building_releases(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM releases WHERE status='BUILDING' ORDER BY created_at").fetchall()
            return [self._release_row(row) for row in rows if row]

    def list_running_evaluations(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM evaluation_runs WHERE status='RUNNING' ORDER BY created_at").fetchall()
            return [dict(row) for row in rows]

    def mark_evaluation_failed(self, eval_run_id: str, message: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE evaluation_runs SET status='FAILED',summary_json=?,updated_at=? WHERE eval_run_id=?",
                (dumps({"error": message}), now_iso(), eval_run_id),
            )

    def save_release_event(self, data: dict[str, Any]) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO release_events(event_id,release_id,event_type,request_id,operator_id,note,details_json,created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    data["event_id"],
                    data["release_id"],
                    data["event_type"],
                    data.get("request_id"),
                    data.get("operator_id"),
                    data.get("note"),
                    dumps(data.get("details", {})),
                    now_iso(),
                ),
            )

    def list_release_events(self, release_id: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM release_events WHERE release_id=? ORDER BY created_at DESC",
                (release_id,),
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["details"] = loads(item.pop("details_json"), {})
                result.append(item)
            return result

    def get_idempotency(self, request_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM idempotency_keys WHERE request_id=?", (request_id,)).fetchone()
            if not row:
                return None
            item = dict(row)
            item["response"] = loads(item.pop("response_json"), {})
            return item

    def save_idempotency(
        self, request_id: str, operation: str, payload_hash: str, status_code: int, response: Any
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO idempotency_keys(request_id,operation,payload_hash,status_code,response_json,created_at) "
                "VALUES(?,?,?,?,?,?) ON CONFLICT(request_id) DO UPDATE SET operation=excluded.operation,"
                "payload_hash=excluded.payload_hash,status_code=excluded.status_code,response_json=excluded.response_json,state='COMPLETED'",
                (request_id, operation, payload_hash, status_code, dumps(response), now_iso()),
            )

    def reserve_idempotency(
        self, request_id: str, operation: str, payload_hash: str, status_code: int = 202
    ) -> bool:
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO idempotency_keys(request_id,operation,payload_hash,status_code,response_json,state,created_at) "
                "VALUES(?,?,?,?,?,'PENDING',?)",
                (request_id, operation, payload_hash, status_code, dumps({"pending": True}), now_iso()),
            )
            return cursor.rowcount == 1

    def delete_idempotency(self, request_id: str) -> None:
        with self.transaction() as conn:
            conn.execute("DELETE FROM idempotency_keys WHERE request_id=? AND state='PENDING'", (request_id,))

    def purge_idempotency(self, max_age_seconds: int) -> None:
        with self.transaction() as conn:
            conn.execute(
                "DELETE FROM idempotency_keys WHERE state='COMPLETED' AND "
                "julianday(created_at) < julianday('now', ?)",
                (f"-{max(60, int(max_age_seconds))} seconds",),
            )

    def create_trace(self, data: dict[str, Any]) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO trace_runs(trace_id,request_id,business_trace_id,name,status,input_summary,"
                "attributes_json,started_at) VALUES(?,?,?,?,?,?,?,?)",
                (
                    data["trace_id"],
                    data.get("request_id"),
                    data.get("business_trace_id"),
                    data.get("name", "rag.run"),
                    "RUNNING",
                    dumps(data.get("input_summary", {})),
                    dumps(data.get("attributes", {})),
                    data["started_at"],
                ),
            )

    def finish_trace(self, trace_id: str, data: dict[str, Any]) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE trace_runs SET status=?,output_summary=?,ended_at=?,duration_ms=? WHERE trace_id=?",
                (
                    data.get("status", "OK"),
                    dumps(data.get("output_summary", {})),
                    data.get("ended_at", now_iso()),
                    data.get("duration_ms"),
                    trace_id,
                ),
            )

    def create_span(self, data: dict[str, Any]) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO trace_spans(span_id,trace_id,parent_span_id,name,stage_type,status,input_summary,"
                "attributes_json,started_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    data["span_id"],
                    data["trace_id"],
                    data.get("parent_span_id"),
                    data["name"],
                    data.get("stage_type", "internal"),
                    "RUNNING",
                    dumps(data.get("input_summary", {})),
                    dumps(data.get("attributes", {})),
                    data["started_at"],
                ),
            )

    def finish_span(self, span_id: str, data: dict[str, Any]) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE trace_spans SET status=?,output_summary=?,attributes_json=?,ended_at=?,duration_ms=?,"
                "error_code=?,error_message=? WHERE span_id=?",
                (
                    data.get("status", "OK"),
                    dumps(data.get("output_summary", {})),
                    dumps(data.get("attributes", {})),
                    data.get("ended_at", now_iso()),
                    data.get("duration_ms"),
                    data.get("error_code"),
                    data.get("error_message"),
                    span_id,
                ),
            )

    def save_trace_candidates(self, trace_id: str, span_id: str, candidates: list[dict[str, Any]]) -> None:
        def trace_payload(item: dict[str, Any]) -> dict[str, Any]:
            text = str(item.get("text") or "")[:800]
            text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[email]", text)
            text = re.sub(r"(?<!\d)(?:\+?\d[\d -]{7,}\d)(?!\d)", "[phone]", text)
            text = re.sub(r"(?i)(?:bearer\s+|sk-)[A-Za-z0-9._~-]{8,}", "[secret]", text)
            return {
                "chunk_id": item.get("chunk_id"),
                "document_id": item.get("document_id"),
                "document_version": item.get("document_version"),
                "title": str(item.get("title") or "")[:300],
                "excerpt": text,
                "section_path": item.get("section_path", [])[:12] if isinstance(item.get("section_path", []), list) else [],
                "location": item.get("location", {}),
                "content_hash": item.get("content_hash"),
                "retrieval_score": item.get("retrieval_score"),
                "vector_score": item.get("vector_score"),
                "lexical_score": item.get("lexical_score"),
                "reranker_score": item.get("reranker_score"),
                "reranker_score_normalized": item.get("reranker_score_normalized"),
                "question_match": item.get("question_match"),
            }

        with self.transaction() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO trace_candidates(trace_id,span_id,rank,chunk_id,document_id,"
                "document_version,retrieval_score,vector_score,lexical_score,reranker_score,"
                "reranker_score_normalized,selected,payload_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        trace_id,
                        span_id,
                        i + 1,
                        item.get("chunk_id", ""),
                        item.get("document_id"),
                        item.get("document_version"),
                        item.get("retrieval_score"),
                        item.get("vector_score"),
                        item.get("lexical_score"),
                        item.get("reranker_score"),
                        item.get("reranker_score_normalized"),
                        1 if item.get("selected") else 0,
                        dumps(trace_payload(item)),
                    )
                    for i, item in enumerate(candidates)
                ],
            )

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            run = conn.execute("SELECT * FROM trace_runs WHERE trace_id=?", (trace_id,)).fetchone()
            if not run:
                return None
            spans = conn.execute(
                "SELECT * FROM trace_spans WHERE trace_id=? ORDER BY started_at", (trace_id,)
            ).fetchall()
            candidates = conn.execute(
                "SELECT * FROM trace_candidates WHERE trace_id=? ORDER BY span_id,rank", (trace_id,)
            ).fetchall()
            feedback = conn.execute(
                "SELECT * FROM trace_feedback WHERE trace_id=? ORDER BY created_at DESC", (trace_id,)
            ).fetchall()
            result = dict(run)
            result["input_summary"] = loads(result.pop("input_summary"), {})
            result["output_summary"] = loads(result.pop("output_summary"), {})
            result["attributes"] = loads(result.pop("attributes_json"), {})
            result["spans"] = []
            for span in spans:
                item = dict(span)
                item["input_summary"] = loads(item.pop("input_summary"), {})
                item["output_summary"] = loads(item.pop("output_summary"), {})
                item["attributes"] = loads(item.pop("attributes_json"), {})
                result["spans"].append(item)
            result["candidates"] = [
                {**dict(row), "payload": loads(row["payload_json"], {})} for row in candidates
            ]
            result["feedback"] = [
                {**dict(row), "tags": loads(row["tags_json"], [])} for row in feedback
            ]
            return result

    def list_traces(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT trace_id,request_id,business_trace_id,name,status,started_at,ended_at,duration_ms,"
                "output_summary FROM trace_runs ORDER BY started_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["output_summary"] = loads(item.pop("output_summary"), {})
                result.append(item)
            return result

    def save_feedback(self, data: dict[str, Any]) -> dict[str, Any]:
        with self.transaction() as conn:
            note = str(data.get("note") or "")[:4000]
            note = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[email]", note)
            note = re.sub(r"(?<!\d)(?:\+?\d[\d -]{7,}\d)(?!\d)", "[phone]", note)
            conn.execute(
                "INSERT INTO trace_feedback(feedback_id,trace_id,request_id,rating,note,reviewer_id,"
                "tags_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (
                    data["feedback_id"],
                    data["trace_id"],
                    data["request_id"],
                    data["rating"],
                    note,
                    data.get("reviewer_id", "local-reviewer"),
                    dumps(data.get("tags", [])),
                    now_iso(),
                ),
            )
            row = conn.execute(
                "SELECT * FROM trace_feedback WHERE feedback_id=?", (data["feedback_id"],)
            ).fetchone()
            item = dict(row)
            item["tags"] = loads(item.pop("tags_json"), [])
            return item

    def create_evaluation_run(self, eval_run_id: str, request_id: str, total: int) -> dict[str, Any]:
        now = now_iso()
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO evaluation_runs(eval_run_id,request_id,status,total_cases,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?)",
                (eval_run_id, request_id, "RUNNING", total, now, now),
            )
            row = conn.execute("SELECT * FROM evaluation_runs WHERE eval_run_id=?", (eval_run_id,)).fetchone()
            return dict(row)

    def configure_evaluation_run(
        self, eval_run_id: str, *, dataset_hash: str, release_id: str | None, config: dict[str, Any]
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE evaluation_runs SET dataset_hash=?,release_id=?,config_json=?,updated_at=? WHERE eval_run_id=?",
                (dataset_hash, release_id, dumps(config), now_iso(), eval_run_id),
            )

    def save_evaluation_case(self, eval_run_id: str, data: dict[str, Any]) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO evaluation_cases(eval_run_id,case_id,category,question,expected_result,"
                "expected_json,actual_result,passed,assertions_json,trace_id,response_json,error,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    eval_run_id,
                    data["case_id"],
                    data.get("category"),
                    data.get("question"),
                    data.get("expected_result"),
                    dumps(data.get("expected", {})),
                    data.get("actual_result"),
                    1 if data.get("passed") else 0,
                    dumps(data.get("assertions", [])),
                    data.get("trace_id"),
                    dumps(data.get("response", {})),
                    data.get("error"),
                    now_iso(),
                ),
            )
            conn.execute(
                "UPDATE evaluation_runs SET completed_cases=(SELECT COUNT(*) FROM evaluation_cases WHERE eval_run_id=?),"
                "updated_at=? WHERE eval_run_id=?",
                (eval_run_id, now_iso(), eval_run_id),
            )

    def finish_evaluation_run(self, eval_run_id: str, status: str, summary: dict[str, Any]) -> None:
        with self.transaction() as conn:
            completed = conn.execute(
                "SELECT COUNT(*) AS count FROM evaluation_cases WHERE eval_run_id=?", (eval_run_id,)
            ).fetchone()["count"]
            conn.execute(
                "UPDATE evaluation_runs SET status=?,completed_cases=?,summary_json=?,updated_at=? "
                "WHERE eval_run_id=?",
                (status, completed, dumps(summary), now_iso(), eval_run_id),
            )

    def get_evaluation_run(self, eval_run_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM evaluation_runs WHERE eval_run_id=?", (eval_run_id,)).fetchone()
            if not row:
                return None
            result = dict(row)
            result["summary"] = loads(result.pop("summary_json"), {})
            result["config"] = loads(result.pop("config_json", "{}"), {})
            cases = conn.execute(
                "SELECT * FROM evaluation_cases WHERE eval_run_id=? ORDER BY case_id", (eval_run_id,)
            ).fetchall()
            result["cases"] = []
            for case in cases:
                item = dict(case)
                item["expected"] = loads(item.pop("expected_json"), {})
                item["assertions"] = loads(item.pop("assertions_json"), [])
                item["response"] = loads(item.pop("response_json"), {})
                result["cases"].append(item)
            return result

    def list_evaluation_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM evaluation_runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["summary"] = loads(item.pop("summary_json"), {})
                item["config"] = loads(item.pop("config_json", "{}"), {})
                result.append(item)
            return result

    def system_counts(self) -> dict[str, Any]:
        with self.connection() as conn:
            documents = conn.execute("SELECT COUNT(*) AS count FROM document_versions").fetchone()["count"]
            ready_documents = conn.execute(
                "SELECT COUNT(*) AS count FROM document_versions WHERE status='READY_FOR_TEST'"
            ).fetchone()["count"]
            chunks = conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()["count"]
            traces = conn.execute("SELECT COUNT(*) AS count FROM trace_runs").fetchone()["count"]
            feedback = conn.execute("SELECT COUNT(*) AS count FROM trace_feedback").fetchone()["count"]
            latest_eval = conn.execute(
                "SELECT eval_run_id,status,total_cases,completed_cases,summary_json FROM evaluation_runs "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            evaluation = None
            if latest_eval:
                evaluation = dict(latest_eval)
                evaluation["summary"] = loads(evaluation.pop("summary_json"), {})
            return {
                "documents": documents,
                "readyDocuments": ready_documents,
                "chunks": chunks,
                "traces": traces,
                "feedback": feedback,
                "latestEvaluation": evaluation,
            }

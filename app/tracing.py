from __future__ import annotations

import contextvars
import hashlib
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator
from contextlib import contextmanager

from .db import Database
from .models import now_iso


_current_trace: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "rag_current_trace", default=None
)
_current_span: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "rag_current_span", default=None
)

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d -]{7,}\d)(?!\d)")
_TOKEN_RE = re.compile(r"(?i)(bearer\s+|sk-[A-Za-z0-9_-]{8,})[A-Za-z0-9._~-]*")


def redact_text(value: str, limit: int = 2000) -> str:
    text = str(value or "")
    text = _EMAIL_RE.sub("[email]", text)
    text = _PHONE_RE.sub("[phone]", text)
    text = _TOKEN_RE.sub("[secret]", text)
    return text[:limit]


def stable_hash(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8", "ignore")).hexdigest()[:24]


def safe_summary(value: Any, *, text_limit: int = 320) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        preview_limit = min(text_limit, 48)
        preview = redact_text(value, preview_limit)
        if len(value) > preview_limit:
            preview += "..."
        return {"preview": preview, "sha256": stable_hash(value), "length": len(value)}
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in list(value.items())[:80]:
            key_text = str(key)
            if key_text.lower() in {"api_key", "authorization", "token", "password", "secret"}:
                output[key_text] = "[redacted]"
            else:
                output[key_text] = safe_summary(item, text_limit=text_limit)
        return output
    if isinstance(value, (list, tuple)):
        return [safe_summary(item, text_limit=text_limit) for item in list(value)[:80]]
    return redact_text(repr(value), text_limit)


@dataclass
class SpanHandle:
    recorder: "TraceHandle"
    span_id: str
    name: str
    stage_type: str
    parent_span_id: str | None
    started_at: str
    started_ns: int
    input_value: Any = None
    attributes: dict[str, Any] = field(default_factory=dict)
    output_value: Any = None
    error_code: str | None = None
    error_message: str | None = None
    _token: contextvars.Token | None = None

    def set_output(self, value: Any) -> None:
        self.output_value = value

    def set_attributes(self, **values: Any) -> None:
        self.attributes.update(values)

    def add_candidates(self, candidates: list[dict[str, Any]]) -> None:
        self.recorder.db.save_trace_candidates(self.recorder.trace_id, self.span_id, candidates)

    def __enter__(self) -> "SpanHandle":
        self._token = _current_span.set(self.span_id)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if exc is not None:
            self.error_code = getattr(exc, "code", None) or exc.__class__.__name__.upper()
            self.error_message = redact_text(str(exc), 500)
            status = "ERROR"
        else:
            status = "OK"
        duration_ms = max(0, int((time.perf_counter_ns() - self.started_ns) / 1_000_000))
        self.recorder.db.finish_span(
            self.span_id,
            {
                "status": status,
                "input_summary": safe_summary(self.input_value),
                "output_summary": safe_summary(self.output_value),
                "attributes": safe_summary(self.attributes),
                "ended_at": now_iso(),
                "duration_ms": duration_ms,
                "error_code": self.error_code,
                "error_message": self.error_message,
            },
        )
        if self._token is not None:
            _current_span.reset(self._token)
        return False


class TraceHandle:
    def __init__(
        self,
        db: Database,
        *,
        trace_id: str,
        request_id: str | None,
        business_trace_id: str | None,
        name: str,
        input_value: Any,
        attributes: dict[str, Any] | None = None,
    ):
        self.db = db
        self.trace_id = trace_id
        self.request_id = request_id
        self.business_trace_id = business_trace_id
        self.name = name
        self.started_ns = time.perf_counter_ns()
        self.started_at = now_iso()
        self.output_value: Any = None
        self.attributes = attributes or {}
        self._trace_token: contextvars.Token | None = None
        self._span_token: contextvars.Token | None = None
        self.db.create_trace(
            {
                "trace_id": trace_id,
                "request_id": request_id,
                "business_trace_id": business_trace_id,
                "name": name,
                "input_summary": safe_summary(input_value),
                "attributes": safe_summary(self.attributes),
                "started_at": self.started_at,
            }
        )
        self.root_span_id = self._create_span(
            "rag.run", "workflow", input_value, self.attributes
        ).span_id

    def _create_span(
        self,
        name: str,
        stage_type: str,
        input_value: Any = None,
        attributes: dict[str, Any] | None = None,
    ) -> SpanHandle:
        span = SpanHandle(
            recorder=self,
            span_id=uuid.uuid4().hex,
            name=name,
            stage_type=stage_type,
            parent_span_id=_current_span.get() or self.root_span_id if hasattr(self, "root_span_id") else None,
            started_at=now_iso(),
            started_ns=time.perf_counter_ns(),
            input_value=input_value,
            attributes=attributes or {},
        )
        self.db.create_span(
            {
                "span_id": span.span_id,
                "trace_id": self.trace_id,
                "parent_span_id": span.parent_span_id,
                "name": name,
                "stage_type": stage_type,
                "input_summary": safe_summary(input_value),
                "attributes": safe_summary(attributes or {}),
                "started_at": span.started_at,
            }
        )
        return span

    @contextmanager
    def span(
        self,
        name: str,
        stage_type: str = "internal",
        input_value: Any = None,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[SpanHandle]:
        span = self._create_span(name, stage_type, input_value, attributes)
        with span:
            yield span

    def finish(self, output: Any, status: str = "OK") -> None:
        self.output_value = output
        duration_ms = max(0, int((time.perf_counter_ns() - self.started_ns) / 1_000_000))
        self.db.finish_span(
            self.root_span_id,
            {
                "status": status,
                "input_summary": {},
                "output_summary": safe_summary(output),
                "attributes": safe_summary(self.attributes),
                "ended_at": now_iso(),
                "duration_ms": duration_ms,
            },
        )
        self.db.finish_trace(
            self.trace_id,
            {
                "status": status,
                "output_summary": safe_summary(output),
                "ended_at": now_iso(),
                "duration_ms": duration_ms,
            },
        )

    def __enter__(self) -> "TraceHandle":
        self._trace_token = _current_trace.set(self.trace_id)
        self._span_token = _current_span.set(self.root_span_id)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if exc is not None:
            self.finish({"error": redact_text(str(exc))}, "ERROR")
        elif self.output_value is None:
            self.finish({}, "OK")
        if self._span_token is not None:
            _current_span.reset(self._span_token)
        if self._trace_token is not None:
            _current_trace.reset(self._trace_token)
        return False


class TraceRecorder:
    def __init__(self, db: Database):
        self.db = db

    def start(
        self,
        *,
        trace_id: str | None = None,
        request_id: str | None = None,
        business_trace_id: str | None = None,
        name: str = "rag.run",
        input_value: Any = None,
        attributes: dict[str, Any] | None = None,
    ) -> TraceHandle:
        return TraceHandle(
            self.db,
            trace_id=trace_id or uuid.uuid4().hex,
            request_id=request_id,
            business_trace_id=business_trace_id,
            name=name,
            input_value=input_value,
            attributes=attributes,
        )

    @staticmethod
    def current_trace_id() -> str | None:
        return _current_trace.get()

    @staticmethod
    def current_span_id() -> str | None:
        return _current_span.get()

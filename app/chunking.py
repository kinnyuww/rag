from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any

from .parsers import ParsedDocument, ParsedElement


POLICY_PROFILES: dict[str, dict[str, int]] = {
    "FAQ_STRUCTURED": {"chunk_chars": 1400, "overlap_chars": 120},
    "TABLE_ROWS": {"chunk_chars": 1600, "overlap_chars": 100},
    "MARKDOWN_HIERARCHICAL": {"chunk_chars": 1300, "overlap_chars": 120},
    "PROSE_RECURSIVE": {"chunk_chars": 1200, "overlap_chars": 120},
}


@dataclass
class ChunkDraft:
    text: str
    title: str
    section_path: list[str]
    location: dict[str, Any]
    metadata: dict[str, Any]


def approximate_tokens(text: str) -> int:
    # Chinese text is roughly two characters per model token; this is only a sizing hint.
    return max(1, math.ceil(len(text) / 2))


def choose_policy(parsed: ParsedDocument, override: str | None = None) -> str:
    if override:
        normalized = override.strip().upper()
        if normalized not in POLICY_PROFILES:
            raise ValueError(f"unknown chunk policy: {override}")
        return normalized
    if any(element.kind == "faq_row" or element.metadata.get("structured") for element in parsed.elements):
        return "FAQ_STRUCTURED"
    if any(element.kind == "table_row" for element in parsed.elements):
        return "TABLE_ROWS"
    if any(element.kind == "heading" for element in parsed.elements) or parsed.parser == "markdown":
        return "MARKDOWN_HIERARCHICAL"
    return "PROSE_RECURSIVE"


def _split_text(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= chunk_chars:
        return [text.strip()]
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_chars,
            chunk_overlap=min(overlap_chars, chunk_chars // 4),
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            strip_whitespace=True,
        )
        result = splitter.split_text(text)
        if result:
            return result
    except Exception:
        pass
    # Conservative fallback for environments where the optional splitter cannot load.
    step = max(1, chunk_chars - overlap_chars)
    return [text[start : start + chunk_chars].strip() for start in range(0, len(text), step) if text[start : start + chunk_chars].strip()]


def _faq_text(element: ParsedElement) -> str:
    fields = element.metadata
    if fields.get("question") or fields.get("answer"):
        return "\n".join(
            f"{label}: {fields.get(key, '')}"
            for label, key in (
                ("问题", "question"),
                ("答案", "answer"),
                ("关键词", "keywords"),
                ("大类", "category"),
                ("小类", "subcategory"),
            )
            if fields.get(key)
        )
    return element.text


def _make_drafts(parsed: ParsedDocument, policy: str) -> list[ChunkDraft]:
    profile = POLICY_PROFILES[policy]
    drafts: list[ChunkDraft] = []
    if policy in {"FAQ_STRUCTURED", "TABLE_ROWS"}:
        for element in parsed.elements:
            text = _faq_text(element)
            for piece_index, piece in enumerate(
                _split_text(text, profile["chunk_chars"], profile["overlap_chars"])
            ):
                location = dict(element.location)
                if piece_index:
                    location["splitPart"] = piece_index + 1
                title = element.metadata.get("question") or (element.section_path[-1] if element.section_path else "")
                drafts.append(
                    ChunkDraft(
                        text=piece,
                        title=str(title or "未命名内容"),
                        section_path=element.section_path.copy(),
                        location=location,
                        metadata=dict(element.metadata),
                    )
                )
        return drafts

    current_path: list[str] = []
    buffer: list[str] = []
    locations: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal buffer, locations
        if not buffer:
            return
        body = "\n".join(buffer).strip()
        if body:
            for piece_index, piece in enumerate(
                _split_text(body, profile["chunk_chars"], profile["overlap_chars"])
            ):
                location: dict[str, Any] = {}
                if locations:
                    location.update(locations[0])
                    if len(locations) > 1:
                        location["sourceLocations"] = locations[:12]
                if piece_index:
                    location["splitPart"] = piece_index + 1
                drafts.append(
                    ChunkDraft(
                        text=piece,
                        title=current_path[-1] if current_path else "未命名内容",
                        section_path=current_path.copy(),
                        location=location,
                        metadata={},
                    )
                )
        buffer = []
        locations = []

    for element in parsed.elements:
        if element.kind == "heading":
            flush()
            current_path = element.section_path.copy() or [element.text]
            continue
        piece = element.text.strip()
        if not piece:
            continue
        contextual = f"章节: {' / '.join(current_path)}\n{piece}" if current_path else piece
        if buffer and len("\n".join(buffer)) + len(contextual) > profile["chunk_chars"]:
            flush()
        buffer.append(contextual)
        locations.append(element.location)
    flush()
    return drafts


def build_chunks(
    parsed: ParsedDocument,
    *,
    document_id: str,
    document_version: int,
    policy_override: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    policy = choose_policy(parsed, policy_override)
    drafts = _make_drafts(parsed, policy)
    chunks: list[dict[str, Any]] = []
    for ordinal, draft in enumerate(drafts):
        text = re.sub(r"\n{3,}", "\n\n", draft.text).strip()
        if not text:
            continue
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        identity = f"{document_id}:{document_version}:{ordinal}:{content_hash}"
        chunk_id = f"chunk-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"
        lexical_text = " ".join(
            str(value)
            for value in (
                draft.metadata.get("question", ""),
                draft.metadata.get("keywords", ""),
                draft.title,
                text,
            )
            if value
        )
        chunks.append(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "document_version": document_version,
                "ordinal": ordinal,
                "text": text,
                "title": draft.title,
                "section_path": draft.section_path,
                "location": draft.location,
                "content_hash": content_hash,
                "lexical_text": lexical_text,
                "metadata": draft.metadata,
                "approx_tokens": approximate_tokens(text),
            }
        )
    if not chunks:
        raise ValueError("chunking produced no chunks")
    return policy, chunks

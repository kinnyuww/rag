from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Snapshot(BaseModel):
    model_config = ConfigDict(extra="ignore")

    conversationKey: str = Field(min_length=1, max_length=300)
    conversationVersion: int = Field(ge=0)
    inputFingerprint: str = Field(min_length=1, max_length=200)


class SourceInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    platform: Literal["DOUYIN", "WEIBO", "BILIBILI", "WECHAT_CHANNELS"] = "DOUYIN"
    accountId: str = "local-default"
    channelType: Literal["COMMENT", "DIRECT_MESSAGE"] = "DIRECT_MESSAGE"


class KnowledgeScope(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tenantId: str = "local-default"
    knowledgeBaseId: str = "main-business-kb"


class QueryMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["USER", "ASSISTANT"]
    messageId: str | None = None
    text: str = Field(min_length=1, max_length=4000)
    sentAt: str | None = None


class QueryPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = Field(min_length=1, max_length=4000)
    messageIds: list[str] = Field(default_factory=list, max_length=50)
    lastMessageAt: str | None = None


class QueryConstraints(BaseModel):
    model_config = ConfigDict(extra="ignore")

    language: str = "zh-CN"
    answerFormat: Literal["PLAIN_TEXT"] = "PLAIN_TEXT"
    maxAnswerChars: int = Field(default=600, ge=20, le=4000)
    requireGrounding: bool = True


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    contractVersion: str = "1.0"
    requestId: str = Field(min_length=1, max_length=200)
    traceId: str = Field(min_length=1, max_length=200)
    snapshot: Snapshot
    source: SourceInfo
    knowledgeScope: KnowledgeScope = Field(default_factory=KnowledgeScope)
    query: QueryPayload
    context: list[QueryMessage] = Field(default_factory=list, max_length=12)
    constraints: QueryConstraints = Field(default_factory=QueryConstraints)

    @field_validator("context")
    @classmethod
    def cap_context_size(cls, value: list[QueryMessage]) -> list[QueryMessage]:
        return value[-12:]


class Citation(BaseModel):
    documentId: str
    documentVersion: int
    chunkId: str
    title: str
    excerpt: str = ""
    page: int | None = None
    sectionPath: list[str] = Field(default_factory=list)
    sheet: str | None = None
    rowStart: int | None = None
    rowEnd: int | None = None
    contentHash: str | None = None
    verificationStatus: Literal["VERIFIED", "REPAIRED", "UNVERIFIED"] = "VERIFIED"


class ScoreBreakdown(BaseModel):
    rerankerScore: float | None = None
    rerankerScoreNormalized: float | None = None
    retrievalScore: float | None = None
    vectorScore: float | None = None
    lexicalScore: float | None = None
    scoreType: str
    higherIsBetter: bool = True
    model: str


class Grounding(BaseModel):
    confidence: float = Field(ge=0, le=1)
    confidenceBand: Literal["HIGH", "MEDIUM", "LOW"]
    confidenceType: str = "DIAGNOSTIC_NOT_CALIBRATED_PROBABILITY"
    sourceReferences: list[Citation] = Field(default_factory=list)
    scores: list[ScoreBreakdown] = Field(default_factory=list)
    evidenceCoverage: float = Field(default=0, ge=0, le=1)


class AnswerBody(BaseModel):
    text: str
    format: Literal["PLAIN_TEXT"] = "PLAIN_TEXT"


class QueryMeta(BaseModel):
    knowledgeBaseId: str
    releaseId: str | None = None
    knowledgeVersion: str | None = None
    serviceVersion: str = "0.1.0"
    latencyMs: int = 0
    traceId: str
    embeddingProvider: str | None = None
    rerankerProvider: str | None = None
    generationProvider: str | None = None
    agentFramework: str | None = None


class QueryResponse(BaseModel):
    contractVersion: str = "1.0"
    requestId: str
    traceId: str
    snapshot: Snapshot
    result: Literal["ANSWERED", "NO_ANSWER"]
    answer: AnswerBody | None = None
    reasonCode: str | None = None
    grounding: Grounding | None = None
    meta: QueryMeta
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class UploadMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    requestId: str | None = None
    knowledgeBaseId: str = "main-business-kb"
    title: str | None = None
    category: str | None = None
    sourceDepartment: str | None = None
    sourceOwner: str | None = None
    effectiveFrom: str | None = None
    effectiveTo: str | None = None
    uploadedBy: str = "local-operator"
    chunkPolicyOverride: str | None = None


class DocumentResponse(BaseModel):
    documentId: str
    documentVersion: int
    title: str
    filename: str
    status: str
    progress: int = 0
    createdAt: str
    updatedAt: str
    processingResult: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None
    chunkPolicy: str | None = None
    contentHash: str | None = None


class TestCandidate(BaseModel):
    documentId: str
    documentVersion: int


class TestSessionCreate(BaseModel):
    requestId: str
    knowledgeBaseId: str = "main-business-kb"
    mode: Literal["SINGLE_DOCUMENT", "PRE_RELEASE"] = "PRE_RELEASE"
    baseReleaseId: str | None = None
    candidateDocuments: list[TestCandidate] = Field(default_factory=list)
    operatorId: str = "local-operator"


class TestSessionResponse(BaseModel):
    requestId: str | None = None
    testSessionId: str
    status: str
    mode: str
    baseReleaseId: str | None = None
    candidateDocuments: list[TestCandidate] = Field(default_factory=list)
    createdAt: str | None = None


class TestQueryRequest(BaseModel):
    requestId: str
    question: str = Field(min_length=1, max_length=4000)
    context: list[QueryMessage] = Field(default_factory=list, max_length=12)


class TestAnswerResponse(BaseModel):
    requestId: str
    testSessionId: str
    answerId: str
    result: Literal["ANSWERED", "NO_ANSWER"]
    answer: AnswerBody | None = None
    decision: Literal["ENABLED", "DISABLED"] = "ENABLED"
    sourceReferences: list[Citation] = Field(default_factory=list)
    grounding: Grounding | None = None
    reasonCode: str | None = None
    traceId: str
    meta: dict[str, Any] = Field(default_factory=dict)


class DecisionRequest(BaseModel):
    requestId: str
    decision: Literal["ENABLED", "DISABLED"]
    reasonCode: Literal["ANSWER_INACCURATE", "SHOULD_HANDOFF", "SOURCE_INCORRECT", "OTHER"] | None = None
    note: str | None = Field(default=None, max_length=2000)
    operatorId: str = "local-operator"


class ReleaseCreate(BaseModel):
    requestId: str
    testSessionId: str
    baseReleaseId: str | None = None
    candidateDocuments: list[TestCandidate] = Field(default_factory=list)
    publishedBy: str = "local-operator"
    publishNote: str = ""


class ReleaseResponse(BaseModel):
    releaseId: str
    knowledgeBaseId: str = "main-business-kb"
    knowledgeVersion: str | None = None
    status: str
    publishedAt: str | None = None
    publishedBy: str | None = None
    enabledTestCaseCount: int = 0
    disabledTestCaseCount: int = 0
    error: dict[str, Any] | None = None
    baseReleaseId: str | None = None
    createdAt: str | None = None


class RollbackRequest(BaseModel):
    requestId: str
    targetReleaseId: str
    operatorId: str = "local-operator"
    note: str = ""


class TraceFeedbackRequest(BaseModel):
    requestId: str
    rating: Literal["GOOD", "BAD"]
    note: str | None = Field(default=None, max_length=4000)
    reviewerId: str = "local-reviewer"
    tags: list[str] = Field(default_factory=list, max_length=20)


class DebugQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    knowledgeBaseId: str = "main-business-kb"
    releaseId: str | None = None
    testSessionId: str | None = None
    context: list[QueryMessage] = Field(default_factory=list, max_length=12)
    platform: Literal["DOUYIN", "WEIBO", "BILIBILI", "WECHAT_CHANNELS"] = "DOUYIN"
    channelType: Literal["COMMENT", "DIRECT_MESSAGE"] = "DIRECT_MESSAGE"
    maxAnswerChars: int = Field(default=600, ge=20, le=4000)
    useRewrite: bool = True
    useAgent: bool = True
    topK: int = Field(default=8, ge=1, le=30)


class EvaluationRunRequest(BaseModel):
    requestId: str
    cases: list[dict[str, Any]] | None = None
    concurrency: int = Field(default=3, ge=1, le=8)
    useGeneration: bool = True

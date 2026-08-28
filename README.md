# Local RAG Service

Standalone local-first RAG service for the AI digital-operations system. It exposes the business contract under `/rag-api/v1/*`, administrative and debug capabilities under `/rag-admin-api/v1/*`, and a developer control room at `/`.

## What It Does

- Accepts PDF, DOCX, XLSX, TXT, Markdown, CSV, TSV, JSON, and JSONL sources.
- Uses format-aware deterministic chunk policies. FAQ rows stay intact; long content uses the open-source LangChain recursive splitter with structure metadata retained.
- Uses FastEmbed `BAAI/bge-small-zh-v1.5` for local Chinese embeddings and `BAAI/bge-reranker-base` for local cross-encoder reranking.
- Uses Qdrant local mode for persistent vector indexing, with a SQLite/NumPy fallback for development.
- Uses hybrid vector + BM25 retrieval, reciprocal-rank fusion, reranking, evidence gating, source citations, and fail-closed `NO_ANSWER` behavior.
- Uses a bounded Deep Agents 0.7.x query synthesizer. The agent receives only a read-only `search_knowledge` tool; filesystem tools, shell execution, and subagents are not exposed.
- Stores immutable document versions, test sessions, test decisions, releases, active-release pointer, rollback history, traces, candidate scores, feedback, and evaluation runs in SQLite.

## Local Start

Python 3.11-3.13 is supported. The repository includes a locked `uv` environment:

```bash
uv sync
cp .env.example .env
uv run uvicorn app.main:app --host 127.0.0.1 --port 18080
```

For local development, the launcher can optionally read an operator-provided token document into process memory without writing the URL or key to `.env`:

```bash
uv run python scripts/start_local.py --port 18280
```

The port is configurable. Use a different port when another local service already owns the default.

Open <http://127.0.0.1:18080/> and use the local token from `RAG_BEARER_TOKEN` (the development default is `dev-local-token`). Do not put a real provider key in Git or in frontend code.

The first ingestion downloads the embedding model. The first query that reaches reranking downloads the cross-encoder model. Keep `RAG_MODEL_CACHE_DIR` on a persistent volume.

## Provider Configuration

Set these variables in `.env` or the process environment:

```dotenv
RAG_LLM_BASE_URL=https://your-openai-compatible-host/v1
RAG_LLM_API_KEY=provided-out-of-band
RAG_LLM_MODEL=gpt-5.6-sol
RAG_GENERATION_PROVIDER=openai_compatible
RAG_DEEP_AGENT_ENABLED=true
```

Embedding and reranking remain local by default. If no generation endpoint is configured, the service uses a clearly labelled deterministic top-evidence fallback for development. Provider state is visible in the UI and health response.

## Import A Knowledge Source

Upload a supported source from Documents, wait for `READY_FOR_TEST`, create a `PRE_RELEASE` or `SINGLE_DOCUMENT` test session, review answers, and publish from Releases. The same flow is available through the documented admin APIs. Source files and generated indexes are runtime data and should stay outside Git.

## API Groups

| Group | Routes |
|---|---|
| Formal | `GET /rag-api/v1/health`, `POST /rag-api/v1/query` |
| Ingestion | `POST/GET /rag-admin-api/v1/documents`, `GET .../chunks` |
| Review | `POST /rag-admin-api/v1/test-sessions`, `POST .../query`, `PUT .../decision` |
| Release | `POST /rag-admin-api/v1/knowledge-bases/{id}/releases`, `GET /rag-admin-api/v1/releases`, `POST .../rollback` |
| Debug | `POST /rag-admin-api/v1/debug/query`, provider and chunk inspection routes |
| Observability | `GET /rag-admin-api/v1/traces`, `GET .../{traceId}`, `POST .../feedback` |
| Evaluation | `POST /rag-admin-api/v1/evaluations/run`, `GET .../{evalRunId}` |

Swagger is available at `/docs`; the frontend API Surface tab lists the same routes.

See [ARCHITECTURE.md](ARCHITECTURE.md) for component boundaries, upstream open-source references, chunk policy selection, score semantics, release concurrency controls, and the evaluation contract.

## Trace And Score Semantics

Each query trace records preparation, rewrite, hybrid retrieval, evidence gate, answer generation, citations, errors, and candidate score progression. The response exposes:

- `rerankerScore`: raw model output;
- `rerankerScoreNormalized`: monotonic diagnostic normalization used for ranking;
- `retrievalScore`: final hybrid ranking score;
- `grounding.confidence`: a diagnostic evidence score combining top reranker score, score gap, and retrieval agreement;
- `grounding.confidenceType=DIAGNOSTIC_NOT_CALIBRATED_PROBABILITY`.

These values are not correctness probabilities. A calibrated probability or interval requires a labelled evaluation set and a separately versioned calibration model.

Reviewers can mark a trace `GOOD` or `BAD` and attach a note. Test answers can be `ENABLED` or `DISABLED`; disabled cases create scoped negative regression records and can block a release when their rejected source remains in the candidate snapshot.

## Docker

Docker is not installed on the current development machine, but the image and Compose configuration are included:

```bash
cp .env.example .env
docker compose up --build
```

The model cache and `data/` directory are mounted so releases and downloaded models survive restarts. The container binds to loopback by default through Compose.

## Evaluation

Generate or refresh the bundled matrix:

```bash
uv run python scripts/generate_eval.py
```

Use the evaluation scripts with a project-owned, versioned dataset that is kept outside the framework repository. Every case should store a Trace ID; strict assertions should cover result, allowed reason code, verified citations, source identity, answer/reference overlap, required terms, and actual model execution without fallback. The repository intentionally does not include business evaluation data or run outputs.

Pass the dataset explicitly when running the CLI:

```bash
uv run python scripts/run_eval.py --dataset /path/to/evaluation.json --base-url http://127.0.0.1:18080 --token "$RAG_BEARER_TOKEN"
```

## Security Boundaries

- Formal queries can only use the active published release. They cannot select drafts or candidate documents.
- User/context/document text is treated as untrusted data and cannot execute tools, commands, or instructions.
- The service never receives platform cookies, platform credentials, or enterprise messaging secrets.
- Trace payloads are redacted and bounded by default. Full raw prompts are not stored by default.
- Publication and rollback are explicit operator actions; a failed build never moves the active pointer.

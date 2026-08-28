# RAG Service Handoff

## Current State

- The implementation is in this repository and is intentionally independent of any business corpus.
- Start the local control room at a configurable loopback address, for example <http://127.0.0.1:18080/>.
- Swagger is available at `/docs`.
- Provider credentials must be supplied out of band and are never committed. `scripts/start_local.py` can load an operator-provided token document into process memory.
- Docker and Compose definitions are included; runtime data and model caches are ignored.

## Models

- Embedding: local FastEmbed/ONNX `BAAI/bge-small-zh-v1.5`, 512 dimensions.
- Reranker: local FastEmbed/ONNX `BAAI/bge-reranker-base`, cross-encoder relevance scores.
- Generation and constrained rewrite: OpenAI-compatible `gpt-5.6-sol`.
- Agent runtime: `deepagents==0.7.9` on LangGraph, with only a read-only `search_knowledge` tool exposed.

## Verification

- Run `uv run pytest -q` for parser, chunking, scope, release, negative-case, and trace-boundary tests.
- Parser fixtures cover PDF, DOCX, XLSX, TXT, Markdown, and explicit scanned-PDF rejection.
- Supply a local, versioned evaluation dataset to `scripts/run_eval.py`; the framework does not ship business questions or run outputs.
- Playwright can verify the control room at desktop and mobile viewports.
- Formal query, debug query, source citations, reranker score output, trace feedback, session restore, release, and rollback are exposed through the API/UI.

## Resume Commands

From the repository root:

```bash
uv sync
uv run python scripts/start_local.py --port 18080
```

To run the matrix against a running service:

```bash
uv run python scripts/run_eval.py --base-url http://127.0.0.1:18080 --token "$RAG_BEARER_TOKEN" --concurrency 3 --min-accuracy 0.99
```

## Next Engineering Task

Use reviewer GOOD/BAD notes and failed traces from a project-owned corpus to tune retrieval or prompts. Keep the dataset hash and release ID fixed while comparing model or chunk-policy changes; do not add a larger agent unless traces demonstrate that the bounded workflow is insufficient.

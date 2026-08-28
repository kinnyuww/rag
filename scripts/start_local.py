from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


def load_provider_from_pdf(environment: dict[str, str], token_pdf: Path) -> None:
    if environment.get("RAG_LLM_BASE_URL") and environment.get("RAG_LLM_API_KEY"):
        return
    if not token_pdf.exists():
        return
    try:
        import pymupdf

        document = pymupdf.open(token_pdf)
        text = "\n".join(page.get_text() for page in document)
        document.close()
    except Exception:
        return
    parts = text.split()
    base_url = next((item for item in parts if item.startswith("https://") or item.startswith("http://")), "")
    api_key = next((item for item in parts if item.startswith("sk-")), "")
    if base_url:
        environment.setdefault("RAG_LLM_BASE_URL", base_url.rstrip("/"))
    if api_key:
        environment.setdefault("RAG_LLM_API_KEY", api_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the local RAG service without persisting provider secrets.")
    parser.add_argument("--host", default=os.environ.get("RAG_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("RAG_PORT", "18080")))
    parser.add_argument("--token-pdf", type=Path, default=WORKSPACE / "token.pdf")
    args = parser.parse_args()

    environment = os.environ.copy()
    environment.setdefault("RAG_HOST", args.host)
    environment.setdefault("RAG_PORT", str(args.port))
    environment.setdefault("RAG_AUTH_ENABLED", "true")
    environment.setdefault("RAG_BEARER_TOKEN", "dev-local-token")
    environment.setdefault("RAG_DATA_DIR", str(ROOT / "data"))
    environment.setdefault("RAG_UPLOAD_DIR", str(ROOT / "data" / "uploads"))
    environment.setdefault("RAG_SQLITE_PATH", str(ROOT / "data" / "rag.db"))
    environment.setdefault("RAG_QDRANT_PATH", str(ROOT / "data" / "qdrant"))
    environment.setdefault("RAG_MODEL_CACHE_DIR", str(ROOT / "models"))
    environment.setdefault("RAG_LLM_MODEL", "gpt-5.6-sol")
    load_provider_from_pdf(environment, args.token_pdf)

    print(f"RAG Control Room: http://{args.host}:{args.port}")
    print("Provider credentials are loaded in process memory and are not written to .env.")
    os.chdir(ROOT)
    os.execvpe(
        sys.executable,
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", args.host, "--port", str(args.port)],
        environment,
    )


if __name__ == "__main__":
    main()

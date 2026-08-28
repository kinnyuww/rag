from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "local-rag-service"
    service_version: str = "0.1.0"
    contract_version: str = "1.0"
    host: str = "127.0.0.1"
    port: int = 18080
    auth_enabled: bool = True
    bearer_token: str = "dev-local-token"

    data_dir: Path = Path("./data")
    upload_dir: Path = Path("./data/uploads")
    model_cache_dir: Path = Path("./models")
    sqlite_path: Path = Path("./data/rag.db")
    qdrant_path: Path = Path("./data/qdrant")

    embedding_provider: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    reranker_provider: str = "fastembed"
    reranker_model: str = "BAAI/bge-reranker-base"
    allow_model_fallback: bool = True
    fastembed_local_files_only: bool = False

    generation_provider: str = "openai_compatible"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "gpt-5.6-sol"
    llm_timeout_seconds: float = 45.0
    llm_max_output_tokens: int = 700
    query_rewrite_enabled: bool = True
    deep_agent_enabled: bool = True

    max_file_bytes: int = 50 * 1024 * 1024
    max_query_chars: int = 4000
    max_context_messages: int = 12
    max_context_chars: int = 12000
    default_top_k: int = 8
    vector_top_k: int = 20
    lexical_top_k: int = 20
    rerank_top_k: int = 16
    rerank_min_score: float = 0.22
    evidence_min_score: float = 0.38
    max_answer_chars: int = 600

    trace_mode: str = "redacted"
    trace_full_ttl_seconds: int = 86400
    idempotency_ttl_seconds: int = 86400

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def prepare_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.qdrant_path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.prepare_directories()
    return settings

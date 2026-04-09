from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    gemini_api_key: str = ""
    gemini_model: str = "gemini-flash-latest"
    gemini_embed_model: str = "gemini-embedding-001"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "multimodal_rag"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "ragdb"
    postgres_user: str = "raguser"
    postgres_password: str = "ragpassword"
    database_url: str = "sqlite+aiosqlite:///./ragdb.sqlite3"

    app_env: str = "development"
    log_level: str = "INFO"
    max_upload_size_mb: int = 50
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    # embedding_dim: int = 768
    embedding_dim: int = 3072
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()

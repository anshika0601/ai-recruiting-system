"""
Centralized config. Import `settings` anywhere you need an env var.
Keeps os.getenv() calls out of business logic and gives you validation for free.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM providers
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Embeddings
    voyage_api_key: str = ""

    # Vector DB
    chroma_persist_dir: str = "./chroma_data"

    # Tracing
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "ai-recruiter"

    # App
    app_env: str = "development"


settings = Settings()

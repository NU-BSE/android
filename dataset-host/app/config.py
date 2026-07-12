import json
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "ARC Human-Played Games Dataset API"
    api_prefix: str = "/v1"
    database_url: str = "postgresql+psycopg://arc:arc@postgres:5432/arc_dataset"

    s3_bucket: str = "arc-human-games"
    s3_region: str = "us-east-1"
    s3_endpoint_url: str | None = None
    s3_public_endpoint_url: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_force_path_style: bool = False
    s3_create_bucket: bool = False

    read_api_key: str | None = None
    allowed_origins: str = "http://localhost:3000"
    signed_url_ttl_seconds: int = Field(default=900, ge=60, le=86400)
    stream_chunk_bytes: int = Field(default=1024 * 1024, ge=64 * 1024)
    max_page_size: int = Field(default=500, ge=1, le=5000)

    @property
    def allowed_origin_list(self) -> list[str]:
        value = self.allowed_origins.strip()
        if not value:
            return []
        if value.startswith("["):
            parsed = json.loads(value)
            if not isinstance(parsed, list):
                raise ValueError("ALLOWED_ORIGINS JSON must be an array")
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [part.strip() for part in value.split(",") if part.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

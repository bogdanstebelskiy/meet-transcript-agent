from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    llm_model: str = "qwen2.5:7b-instruct"
    embed_model: str = "nomic-embed-text"

    whisper_bin: Path = Path("whisper-cli")
    whisper_model: Path = Path("models/ggml-base.bin")
    whisper_language: str = "en"

    chunk_duration: float = 30.0
    chunk_overlap_duration: float = 5.0

    data_dir: Path = PROJECT_DIR / ".data"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

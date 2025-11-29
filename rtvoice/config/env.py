from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentEnv(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    pico_access_key: str
    openai_api_key: str
    rtvoice_log_level: str = "WARNING"

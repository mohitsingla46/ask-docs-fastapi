from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List, Union, Optional


class Settings(BaseSettings):
    PROJECT_NAME: str
    MONGODB_URL: str
    DATABASE_NAME: str
    BACKEND_CORS_ORIGINS: List[str] = []
    API_V1_STR: str = "/api/v1"
    GOOGLE_CLIENT_ID: str
    JWT_SECRET: str
    UPLOAD_DIR: str = "uploads"
    NOMIC_API_KEY: str
    EMBEDDING_MODEL: str
    PINECONE_INDEX_NAME: str
    PINECONE_API_KEY: str
    GROQ_API_KEY: str
    GROQ_CHAT_MODEL: str
    GROQ_TEMPERATURE: float
    GOOGLE_API_KEY: str
    LANGSMITH_TRACING: str
    LANGSMITH_API_KEY: str
    LANGSMITH_ENDPOINT: str
    LANGSMITH_PROJECT: str
    LANGCHAIN_CALLBACKS_BACKGROUND: str

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()

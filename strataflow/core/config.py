"""
Configuration management using Pydantic Settings.

Supports environment variables, .env files, and type-safe configuration.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    # Neo4j Graph Database
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="password")

    # Redis Cache
    redis_url: str = Field(default="redis://localhost:6379")

    # Vector Database (Qdrant)
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_prefix="DB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class LLMConfig(BaseSettings):
    """LLM provider configuration."""

    provider: Literal["gemini", "openai", "anthropic", "local"] = "gemini"

    # Gemini
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash-thinking-exp"
    gemini_endpoint: str = "https://generativelanguage.googleapis.com/v1beta"

    # OpenAI
    openai_api_key: str | None = None
    openai_model: str = "gpt-4-turbo-preview"

    # Anthropic
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-sonnet-20240229"

    # Local LLM
    local_model_path: str | None = None

    # Generation parameters
    default_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    default_max_tokens: int = Field(default=2000, ge=1, le=100000)

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class AgentConfig(BaseSettings):
    """Agent execution configuration."""

    # Parallelization
    max_parallel_agents: int = Field(default=5, ge=1, le=50)
    agent_timeout_seconds: int = Field(default=300, ge=10)

    # Verification thresholds
    conservative_inference_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    speculative_inference_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

    # Proof search
    max_proof_depth: int = Field(default=5, ge=1, le=20)
    max_proof_branches: int = Field(default=100, ge=1)

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class ObservabilityConfig(BaseSettings):
    """Observability and monitoring configuration."""

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"

    # Metrics
    enable_prometheus: bool = True
    prometheus_port: int = Field(default=9090, ge=1024, le=65535)

    # Tracing
    enable_tracing: bool = True
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    trace_sampling_rate: float = Field(default=0.1, ge=0.0, le=1.0)

    model_config = SettingsConfigDict(
        env_prefix="OBS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class PerformanceConfig(BaseSettings):
    """Performance optimization configuration."""

    # Caching
    enable_embedding_cache: bool = True
    enable_inference_cache: bool = True
    cache_ttl_seconds: int = Field(default=3600, ge=60)

    # Model optimization
    enable_quantization: bool = False
    quantization_bits: Literal[8, 16] = 8

    # Batch processing
    batch_size_propositions: int = Field(default=32, ge=1, le=1000)
    batch_size_embeddings: int = Field(default=64, ge=1, le=1000)

    model_config = SettingsConfigDict(
        env_prefix="PERF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class StrataFlowConfig(BaseSettings):
    """Main StrataFlow configuration."""

    # Application
    app_name: str = "StrataFlow"
    version: str = "2.0.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = True

    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1024, le=65535)
    api_workers: int = Field(default=4, ge=1)

    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    # Paths
    data_dir: Path = Field(default=Path("./data"))
    output_dir: Path = Field(default=Path("./outputs"))
    cache_dir: Path = Field(default=Path("./cache"))

    model_config = SettingsConfigDict(
        env_prefix="STRATAFLOW_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("data_dir", "output_dir", "cache_dir")
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        """Ensure directories exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"

    def is_debug(self) -> bool:
        """Check if debug mode is enabled."""
        return self.debug and not self.is_production()


@lru_cache()
def get_config() -> StrataFlowConfig:
    """
    Get cached configuration instance.

    Uses lru_cache to ensure singleton pattern - configuration is loaded once.
    """
    return StrataFlowConfig()


__all__ = [
    "StrataFlowConfig",
    "DatabaseConfig",
    "LLMConfig",
    "AgentConfig",
    "ObservabilityConfig",
    "PerformanceConfig",
    "get_config",
]

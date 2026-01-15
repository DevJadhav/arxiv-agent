"""Application settings with Pydantic BaseSettings."""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    from xdg_base_dirs import xdg_cache_home, xdg_config_home, xdg_data_home
except ImportError:
    # Fallback for systems without xdg-base-dirs
    def xdg_data_home() -> Path:
        return Path.home() / ".local" / "share"
    
    def xdg_config_home() -> Path:
        return Path.home() / ".config"
    
    def xdg_cache_home() -> Path:
        return Path.home() / ".cache"


# Provider type alias
LLMProviderType = Literal["anthropic", "openai", "gemini"]


class AgentModelConfig(BaseSettings):
    """Per-agent model configuration."""
    
    provider: LLMProviderType | None = None  # None = use default
    model: str | None = None  # None = use provider default


class LLMSettings(BaseSettings):
    """LLM provider configuration."""
    
    # Default provider
    default_provider: LLMProviderType = "anthropic"
    
    # Provider-specific default models
    anthropic_model: str = "claude-sonnet-4-5-20250929"
    anthropic_model_advanced: str = "claude-opus-4-5-20251101"
    anthropic_model_fast: str = "claude-haiku-4-5-20251001"
    openai_model: str = "gpt-4o"
    openai_model_advanced: str = "gpt-4o"
    gemini_model: str = "gemini-1.5-flash"
    gemini_model_advanced: str = "gemini-1.5-pro"
    
    # Generation settings
    max_tokens: int = 4096
    temperature: float = 0.7
    
    # Per-agent overrides (advanced settings)
    analyzer_config: AgentModelConfig = Field(default_factory=AgentModelConfig)
    chat_config: AgentModelConfig = Field(default_factory=AgentModelConfig)
    code_config: AgentModelConfig = Field(default_factory=AgentModelConfig)
    digest_config: AgentModelConfig = Field(default_factory=AgentModelConfig)
    
    def get_provider_model(self, provider: LLMProviderType, advanced: bool = False) -> str:
        """Get the default model for a provider."""
        if provider == "anthropic":
            return self.anthropic_model_advanced if advanced else self.anthropic_model
        elif provider == "openai":
            return self.openai_model_advanced if advanced else self.openai_model
        elif provider == "gemini":
            return self.gemini_model_advanced if advanced else self.gemini_model
        return self.anthropic_model
    
    def get_agent_config(self, agent: str) -> tuple[LLMProviderType, str]:
        """Get provider and model for a specific agent.
        
        Args:
            agent: Agent name (analyzer, chat, code, digest)
        
        Returns:
            Tuple of (provider, model)
        """
        configs = {
            "analyzer": self.analyzer_config,
            "chat": self.chat_config,
            "code": self.code_config,
            "digest": self.digest_config,
        }
        
        config = configs.get(agent)
        
        # Use agent-specific or fall back to default
        provider = config.provider if config and config.provider else self.default_provider
        
        # Use agent-specific model or provider default
        if config and config.model:
            model = config.model
        else:
            # For code agent, use advanced model
            advanced = agent == "code"
            model = self.get_provider_model(provider, advanced)
        
        return provider, model


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""
    
    provider: Literal["local", "openai"] = "local"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32


class RetrievalSettings(BaseSettings):
    """RAG retrieval configuration."""
    
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    top_k: int = 20
    rerank_top_k: int = 5
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"


class ChunkingSettings(BaseSettings):
    """Document chunking configuration."""
    
    strategy: Literal["fixed", "recursive", "section_aware"] = "section_aware"
    chunk_size: int = 768
    chunk_overlap: int = 100
    min_chunk_size: int = 100


class DigestSettings(BaseSettings):
    """Daily digest configuration."""
    
    enabled: bool = True
    schedule_time: str = "06:00"
    timezone: str = "UTC"
    max_papers: int = 10
    categories: list[str] = Field(default_factory=lambda: ["cs.AI", "cs.LG", "cs.CL"])
    keywords: list[str] = Field(default_factory=list)
    min_citation_count: int = 0


class ChatSettings(BaseSettings):
    """Chat session configuration."""
    
    session_timeout_hours: int = 24
    max_history_messages: int = 100
    context_window_tokens: int = 8000
    summarize_after_messages: int = 20


class LibrarySettings(BaseSettings):
    """Library configuration."""
    
    auto_download_pdfs: bool = True
    pdf_storage_format: Literal["year/month", "flat", "category"] = "year/month"
    auto_tag: bool = True
    tag_confidence_threshold: float = 0.7


class UISettings(BaseSettings):
    """UI/Theme configuration."""
    
    theme: str = "default"
    show_progress: bool = True
    table_style: str = "rounded"
    max_width: int = 120


class Paper2CodeSettings(BaseSettings):
    """Paper-to-code generation settings."""
    
    default_framework: Literal["pytorch", "jax", "tensorflow"] = "pytorch"
    generate_tests: bool = True
    generate_docs: bool = True
    include_examples: bool = True
    type_hints: bool = True


class GuardrailsSettings(BaseSettings):
    """Safety and cost guardrails."""
    
    max_daily_tokens: int = 100_000
    warn_on_costly_request: bool = True
    enable_json_repair: bool = True


class Settings(BaseSettings):
    """Main application settings."""
    
    model_config = SettingsConfigDict(
        env_prefix="ARXIV_AGENT_",
        env_nested_delimiter="__",
        extra="ignore",
    )
    
    # Paths
    data_dir: Path = Field(default_factory=lambda: xdg_data_home() / "arxiv-agent")
    cache_dir: Path = Field(default_factory=lambda: xdg_cache_home() / "arxiv-agent")
    config_dir: Path = Field(default_factory=lambda: xdg_config_home() / "arxiv-agent")
    
    # Legacy API keys (prefer using KeyStorage)
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    semantic_scholar_api_key: SecretStr | None = None
    
    # Nested settings
    llm: LLMSettings = Field(default_factory=LLMSettings)
    guardrails: GuardrailsSettings = Field(default_factory=GuardrailsSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    digest: DigestSettings = Field(default_factory=DigestSettings)
    chat: ChatSettings = Field(default_factory=ChatSettings)
    library: LibrarySettings = Field(default_factory=LibrarySettings)
    ui: UISettings = Field(default_factory=UISettings)
    paper2code: Paper2CodeSettings = Field(default_factory=Paper2CodeSettings)
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    
    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        for path in [
            self.data_dir,
            self.data_dir / "db",
            self.data_dir / "vectors",
            self.data_dir / "pdfs",
            self.data_dir / "digests",
            self.data_dir / "exports",
            self.cache_dir,
            self.cache_dir / "api_responses",
            self.cache_dir / "embeddings",
            self.config_dir,
            self.config_dir / "themes",
        ]:
            path.mkdir(parents=True, exist_ok=True)
    
    @property
    def db_path(self) -> Path:
        """Get the main database path."""
        return self.data_dir / "db" / "arxiv-agent.db"
    
    @property
    def vector_db_path(self) -> Path:
        """Get the vector database path."""
        return self.data_dir / "vectors" / "chroma"
    
    def get_pdf_dir(self, year: int, month: int) -> Path:
        """Get PDF directory for a specific date."""
        return self.data_dir / "pdfs" / str(year) / f"{month:02d}"


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_directories()
    return _settings


def reset_settings() -> None:
    """Reset settings instance (useful for testing)."""
    global _settings
    _settings = None

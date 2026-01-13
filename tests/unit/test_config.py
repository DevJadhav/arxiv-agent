"""Unit tests for ArXiv Agent configuration."""

import pytest
from pathlib import Path

from arxiv_agent.config.settings import (
    Settings,
    LLMSettings,
    EmbeddingSettings,
    DigestSettings,
    get_settings,
    reset_settings,
)


class TestSettings:
    """Tests for Settings class."""
    
    def setup_method(self):
        """Reset settings before each test."""
        reset_settings()
    
    def test_default_settings(self):
        """Test that default settings are created correctly."""
        settings = Settings()
        
        assert settings.llm.anthropic_model == "claude-sonnet-4-5-20250514"
        assert settings.llm.default_provider == "anthropic"
        assert settings.embedding.model == "allenai/specter2"
        assert settings.embedding.dimension == 768
        assert settings.digest.max_papers == 10
        assert settings.retrieval.dense_weight == 0.7
    
    def test_paths_are_pathlib(self):
        """Test that path settings are Path objects."""
        settings = Settings()
        
        assert isinstance(settings.data_dir, Path)
        assert isinstance(settings.cache_dir, Path)
        assert isinstance(settings.config_dir, Path)
    
    def test_db_path_property(self):
        """Test that db_path property works correctly."""
        settings = Settings()
        
        assert settings.db_path.name == "arxiv-agent.db"
        assert settings.db_path.parent.name == "db"
    
    def test_vector_db_path_property(self):
        """Test that vector_db_path property works correctly."""
        settings = Settings()
        
        assert settings.vector_db_path.name == "chroma"
        assert settings.vector_db_path.parent.name == "vectors"
    
    def test_get_pdf_dir(self):
        """Test PDF directory generation."""
        settings = Settings()
        
        pdf_dir = settings.get_pdf_dir(2025, 1)
        
        assert "2025" in str(pdf_dir)
        assert "01" in str(pdf_dir)
    
    def test_ensure_directories(self, tmp_path):
        """Test directory creation."""
        settings = Settings(
            data_dir=tmp_path / "data",
            cache_dir=tmp_path / "cache",
            config_dir=tmp_path / "config",
        )
        
        settings.ensure_directories()
        
        assert (tmp_path / "data" / "db").exists()
        assert (tmp_path / "data" / "vectors").exists()
        assert (tmp_path / "data" / "pdfs").exists()
        assert (tmp_path / "cache").exists()
        assert (tmp_path / "config").exists()


class TestLLMSettings:
    """Tests for LLM settings."""
    
    def test_default_models(self):
        """Test default LLM models for providers."""
        settings = LLMSettings()
        
        assert settings.anthropic_model == "claude-sonnet-4-5-20250514"
        assert settings.anthropic_model_advanced == "claude-opus-4-5-20250101"
        assert settings.openai_model == "gpt-4o"
        assert settings.gemini_model == "gemini-2.5-pro"
        assert settings.max_tokens == 4096
        assert settings.temperature == 0.7
    
    def test_get_agent_config(self):
        """Test per-agent configuration."""
        settings = LLMSettings()
        
        # Default should use anthropic
        provider, model = settings.get_agent_config("analyzer")
        assert provider == "anthropic"
        assert model == settings.anthropic_model
        
        # Code agent uses advanced model
        provider, model = settings.get_agent_config("code")
        assert provider == "anthropic"
        assert model == settings.anthropic_model_advanced
    
    def test_get_provider_model(self):
        """Test getting model for provider."""
        settings = LLMSettings()
        
        assert settings.get_provider_model("anthropic") == "claude-sonnet-4-5-20250514"
        assert settings.get_provider_model("openai") == "gpt-4o"
        assert settings.get_provider_model("gemini") == "gemini-2.5-pro"


class TestDigestSettings:
    """Tests for digest settings."""
    
    def test_default_categories(self):
        """Test default arXiv categories."""
        settings = DigestSettings()
        
        assert "cs.AI" in settings.categories
        assert "cs.LG" in settings.categories
        assert settings.max_papers == 10
        assert settings.enabled is True


class TestGetSettings:
    """Tests for get_settings singleton."""
    
    def setup_method(self):
        """Reset settings before each test."""
        reset_settings()
    
    def test_returns_same_instance(self):
        """Test that get_settings returns singleton."""
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2
    
    def test_reset_creates_new_instance(self):
        """Test that reset_settings allows new instance."""
        settings1 = get_settings()
        reset_settings()
        settings2 = get_settings()
        
        assert settings1 is not settings2

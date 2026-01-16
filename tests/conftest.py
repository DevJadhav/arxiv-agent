"""Pytest configuration and shared fixtures for ArXiv Agent tests."""

import os
from pathlib import Path
from typing import Generator

import pytest
import vcr

from arxiv_agent.config.settings import Settings, get_settings
from arxiv_agent.data.storage import DatabaseManager


# ==================== VCR Configuration ====================

# Path to VCR cassettes
CASSETTES_DIR = Path(__file__).parent / "cassettes"
CASSETTES_DIR.mkdir(exist_ok=True)


def scrub_api_keys(request):
    """Remove API keys from recorded requests."""
    headers_to_scrub = [
        "x-api-key",
        "authorization",
        "api-key",
        "x-goog-api-key",
    ]
    
    for header in headers_to_scrub:
        if header in request.headers:
            request.headers[header] = "REDACTED"
    
    return request


def scrub_response_headers(response):
    """Remove sensitive data from recorded responses."""
    headers_to_remove = [
        "set-cookie",
        "x-request-id",
        "cf-ray",
    ]
    
    for header in headers_to_remove:
        if header in response.get("headers", {}):
            del response["headers"][header]
    
    return response


# Default VCR configuration
vcr_config = vcr.VCR(
    cassette_library_dir=str(CASSETTES_DIR),
    record_mode="once",  # Record once, then replay
    match_on=["uri", "method", "body"],
    filter_headers=[
        ("x-api-key", "REDACTED"),
        ("authorization", "REDACTED"),
        ("api-key", "REDACTED"),
        ("x-goog-api-key", "REDACTED"),
    ],
    before_record_request=scrub_api_keys,
    before_record_response=scrub_response_headers,
    decode_compressed_response=True,
)


@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    """Return cassette directory based on test module."""
    module_name = request.module.__name__.split(".")[-1]
    cassette_dir = CASSETTES_DIR / module_name
    cassette_dir.mkdir(exist_ok=True)
    return cassette_dir


# ==================== Database Fixtures ====================

@pytest.fixture
def temp_db(tmp_path) -> Generator[DatabaseManager, None, None]:
    """Create a temporary database for testing."""
    db_path = tmp_path / "test.db"
    db = DatabaseManager(db_path=db_path)
    yield db


@pytest.fixture
def temp_db_with_fts(tmp_path) -> Generator[DatabaseManager, None, None]:
    """Create a temporary database with FTS5 enabled."""
    db_path = tmp_path / "test_fts.db"
    db = DatabaseManager(db_path=db_path)
    db.init_fts()
    yield db


# ==================== Settings Fixtures ====================

@pytest.fixture
def mock_settings(tmp_path, monkeypatch) -> Settings:
    """Create mock settings with temp directories."""
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    data_dir.mkdir()
    cache_dir.mkdir()
    
    # Reset singleton
    from arxiv_agent.config import settings as settings_module
    settings_module._settings = None
    
    # Mock XDG paths
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    
    settings = get_settings()
    return settings


# ==================== API Client Fixtures ====================

@pytest.fixture
def mock_arxiv_response():
    """Sample arXiv API response for testing."""
    return {
        "id": "arxiv:2401.12345",
        "title": "Test Paper: A Novel Approach",
        "authors": ["Alice Smith", "Bob Jones"],
        "abstract": "This is a test abstract for unit testing purposes.",
        "categories": ["cs.AI", "cs.LG"],
        "published_date": "2024-01-15T00:00:00Z",
    }


@pytest.fixture
def mock_semantic_scholar_response():
    """Sample Semantic Scholar API response for testing."""
    return {
        "paperId": "abc123",
        "citationCount": 42,
        "tldr": {"text": "This paper introduces a novel approach."},
        "references": [],
        "citations": [],
    }


# ==================== LLM Fixtures ====================

@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    from arxiv_agent.core.llm_service import LLMResponse
    
    return LLMResponse(
        content="This is a test response from the LLM.",
        model="claude-3-5-sonnet-20241022",
        provider="anthropic",
        input_tokens=100,
        output_tokens=50,
        finish_reason="end_turn",
    )


# ==================== Async Fixtures ====================

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ==================== VCR Cassette Fixtures ====================

@pytest.fixture
def arxiv_vcr(vcr_cassette_dir):
    """VCR instance configured for arXiv API."""
    return vcr_config.use_cassette(
        str(vcr_cassette_dir / "arxiv.yaml"),
        record_mode="once",
    )


@pytest.fixture
def semantic_scholar_vcr(vcr_cassette_dir):
    """VCR instance configured for Semantic Scholar API."""
    return vcr_config.use_cassette(
        str(vcr_cassette_dir / "semantic_scholar.yaml"),
        record_mode="once",
    )


@pytest.fixture
def anthropic_vcr(vcr_cassette_dir):
    """VCR instance configured for Anthropic API."""
    return vcr_config.use_cassette(
        str(vcr_cassette_dir / "anthropic.yaml"),
        record_mode="once",
        filter_headers=[("x-api-key", "REDACTED")],
    )


@pytest.fixture
def openai_vcr(vcr_cassette_dir):
    """VCR instance configured for OpenAI API."""
    return vcr_config.use_cassette(
        str(vcr_cassette_dir / "openai.yaml"),
        record_mode="once",
        filter_headers=[("authorization", "REDACTED")],
    )


# ==================== Test Data Fixtures ====================

@pytest.fixture
def sample_paper():
    """Create a sample paper for testing."""
    from datetime import datetime
    from arxiv_agent.data.models import Paper
    
    return Paper(
        id="arxiv:2401.12345",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
        categories=["cs.CL", "cs.LG"],
        published_date=datetime(2024, 1, 15),
        citation_count=50000,
    )


@pytest.fixture
def sample_papers(sample_paper):
    """Create multiple sample papers for testing."""
    from datetime import datetime
    from arxiv_agent.data.models import Paper
    
    return [
        sample_paper,
        Paper(
            id="arxiv:2401.12346",
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            authors=["Jacob Devlin", "Ming-Wei Chang"],
            abstract="We introduce a new language representation model called BERT...",
            categories=["cs.CL"],
            published_date=datetime(2024, 1, 16),
            citation_count=30000,
        ),
        Paper(
            id="arxiv:2401.12347",
            title="GPT-4 Technical Report",
            authors=["OpenAI"],
            abstract="We report the development of GPT-4, a large-scale multimodal model...",
            categories=["cs.AI", "cs.CL", "cs.LG"],
            published_date=datetime(2024, 1, 17),
            citation_count=5000,
        ),
    ]

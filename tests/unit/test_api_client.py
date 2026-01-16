"""Unit tests for API client with VCR recording."""

import pytest
import vcr
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from arxiv_agent.core.api_client import APIClientManager, get_api_client


CASSETTES_DIR = Path(__file__).parent.parent / "cassettes" / "test_api_client"


class TestAPIClientManager:
    """Tests for ArXiv API client manager."""
    
    @pytest.fixture
    def client(self):
        """Create API client."""
        with patch("arxiv_agent.core.api_client.get_settings") as mock_settings:
            mock_settings.return_value.api.arxiv_rate_limit = 3.0
            mock_settings.return_value.api.semantic_scholar_rate_limit = 10.0
            mock_settings.return_value.api.semantic_scholar_api_key = None
            return APIClientManager()
    
    def test_client_initialization(self, client):
        """Test client initializes correctly."""
        assert client is not None
    
    @pytest.mark.asyncio
    async def test_search_papers_mock(self, client):
        """Test paper search with mocked response."""
        from arxiv_agent.data.models import Paper
        from datetime import datetime
        
        # Create mock paper
        mock_paper = Paper(
            id="arxiv:2401.12345",
            title="Test Paper",
            authors=["Author One"],
            abstract="Test abstract",
            categories=["cs.AI"],
            published_date=datetime(2024, 1, 15),
        )
        
        with patch.object(client, "search_arxiv", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [mock_paper]
            
            papers = await client.search_arxiv("test query", max_results=10)
            assert len(papers) == 1
            assert papers[0].title == "Test Paper"
    
    @pytest.mark.asyncio
    async def test_get_paper_mock(self, client):
        """Test fetching single paper with mock."""
        from arxiv_agent.data.models import Paper
        from datetime import datetime
        
        mock_paper = Paper(
            id="arxiv:2401.12345",
            title="Test Paper",
            authors=["Author One"],
            abstract="Test abstract",
            categories=["cs.AI"],
            published_date=datetime(2024, 1, 15),
        )
        
        with patch.object(client, "get_paper_by_id", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_paper
            
            paper = await client.get_paper_by_id("2401.12345")
            assert paper.id == "arxiv:2401.12345"


class TestSemanticScholarEnrichment:
    """Tests for Semantic Scholar enrichment."""
    
    @pytest.fixture
    def client(self):
        """Create API client."""
        with patch("arxiv_agent.core.api_client.get_settings") as mock_settings:
            mock_settings.return_value.api.arxiv_rate_limit = 3.0
            mock_settings.return_value.api.semantic_scholar_rate_limit = 10.0
            mock_settings.return_value.api.semantic_scholar_api_key = "test_key"
            return APIClientManager()
    
    @pytest.mark.asyncio
    async def test_enrich_with_citations_mock(self, client):
        """Test citation enrichment with mocked response."""
        from arxiv_agent.data.models import Paper
        from datetime import datetime
        
        paper = Paper(
            id="arxiv:2401.12345",
            title="Test Paper",
            authors=["Author One"],
            abstract="Test abstract",
            categories=["cs.AI"],
            published_date=datetime(2024, 1, 15),
        )
        
        enriched_paper = Paper(
            id=paper.id,
            title=paper.title,
            authors=paper.authors,
            abstract=paper.abstract,
            categories=paper.categories,
            published_date=paper.published_date,
            citation_count=42,
            tldr="This is a test TLDR summary.",
        )
        
        with patch.object(client, "enrich_with_semantic_scholar", new_callable=AsyncMock) as mock_enrich:
            mock_enrich.return_value = enriched_paper
            
            result = await client.enrich_with_semantic_scholar(paper)
            assert result.citation_count == 42
            assert result.tldr == "This is a test TLDR summary."


class TestGlobalClient:
    """Tests for global client instance management."""
    
    def test_get_api_client_returns_instance(self):
        """Test global client getter returns instance."""
        with patch("arxiv_agent.core.api_client.get_settings") as mock_settings:
            mock_settings.return_value.api.arxiv_rate_limit = 3.0
            mock_settings.return_value.api.semantic_scholar_rate_limit = 10.0
            mock_settings.return_value.api.semantic_scholar_api_key = None
            
            # Reset global instance first
            import arxiv_agent.core.api_client as api_module
            api_module._api_client = None
            
            client = get_api_client()
            assert client is not None
            assert isinstance(client, APIClientManager)


# VCR-based integration tests (run with actual API when recording)
@pytest.mark.vcr
class TestAPIClientVCR:
    """VCR-recorded tests for actual API interactions."""
    
    @pytest.fixture
    def vcr_config(self):
        """VCR configuration for this test class."""
        return {
            "cassette_library_dir": str(CASSETTES_DIR),
            "record_mode": "once",
            "match_on": ["uri", "method"],
            "filter_headers": [("x-api-key", "REDACTED")],
        }
    
    @pytest.mark.skip(reason="Requires VCR cassette recording")
    @pytest.mark.asyncio
    async def test_real_search(self):
        """Test real arXiv search (recorded with VCR)."""
        with patch("arxiv_agent.core.api_client.get_settings") as mock_settings:
            mock_settings.return_value.api.arxiv_rate_limit = 3.0
            mock_settings.return_value.api.semantic_scholar_rate_limit = 10.0
            mock_settings.return_value.api.semantic_scholar_api_key = None
            
            client = APIClientManager()
            
            with vcr.VCR().use_cassette(str(CASSETTES_DIR / "real_search.yaml")):
                papers = await client.search_arxiv("transformer attention", max_results=5)
                
                assert len(papers) > 0
                assert papers[0].title is not None

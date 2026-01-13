"""Unit tests for ArXiv Agent data models."""

import pytest
from datetime import datetime

from arxiv_agent.data.models import (
    Paper,
    Analysis,
    Collection,
    ChatSession,
    ChatMessage,
    Tag,
    ReadingHistory,
)


class TestPaperModel:
    """Tests for Paper model."""
    
    def test_paper_creation(self):
        """Test basic paper creation."""
        paper = Paper(
            id="arxiv:2401.12345",
            title="Test Paper",
            authors=["Author One", "Author Two"],
            abstract="This is a test abstract.",
            categories=["cs.AI", "cs.LG"],
        )
        
        assert paper.id == "arxiv:2401.12345"
        assert paper.title == "Test Paper"
        assert len(paper.authors) == 2
        assert "cs.AI" in paper.categories
    
    def test_paper_defaults(self):
        """Test paper default values."""
        paper = Paper(
            id="arxiv:2401.12345",
            title="Test",
            abstract="Abstract",
        )
        
        assert paper.authors == []
        assert paper.categories == []
        assert paper.citation_count == 0
        assert paper.pdf_path is None
        assert paper.tldr is None
    
    def test_paper_with_dates(self):
        """Test paper with date fields."""
        now = datetime.utcnow()
        paper = Paper(
            id="arxiv:2401.12345",
            title="Test",
            abstract="Abstract",
            published_date=now,
            created_at=now,
        )
        
        assert paper.published_date == now
        assert paper.created_at == now


class TestAnalysisModel:
    """Tests for Analysis model."""
    
    def test_analysis_creation(self):
        """Test analysis creation."""
        analysis = Analysis(
            paper_id="arxiv:2401.12345",
            analysis_type="full",
            content={"text": "Analysis content", "sections": []},
            model_used="claude-sonnet-4-5-20250514",
            token_count=1000,
        )
        
        assert analysis.paper_id == "arxiv:2401.12345"
        assert analysis.analysis_type == "full"
        assert "text" in analysis.content
        assert analysis.token_count == 1000


class TestCollectionModel:
    """Tests for Collection model."""
    
    def test_collection_creation(self):
        """Test collection creation."""
        collection = Collection(
            name="My Papers",
            description="Important papers to read",
            color="#FF5733",
        )
        
        assert collection.name == "My Papers"
        assert collection.description == "Important papers to read"
        assert collection.color == "#FF5733"
    
    def test_collection_default_color(self):
        """Test default collection color."""
        collection = Collection(name="Test")
        
        assert collection.color == "#3B82F6"


class TestChatModels:
    """Tests for chat-related models."""
    
    def test_chat_session_creation(self):
        """Test chat session creation."""
        session = ChatSession(
            id="test-uuid",
            paper_id="arxiv:2401.12345",
        )
        
        assert session.id == "test-uuid"
        assert session.paper_id == "arxiv:2401.12345"
        assert session.message_count == 0
    
    def test_chat_message_creation(self):
        """Test chat message creation."""
        message = ChatMessage(
            session_id="test-uuid",
            role="user",
            content="What is this paper about?",
        )
        
        assert message.role == "user"
        assert message.content == "What is this paper about?"
        assert message.retrieved_chunks is None


class TestTagModel:
    """Tests for Tag model."""
    
    def test_tag_creation(self):
        """Test tag creation."""
        tag = Tag(
            name="machine-learning",
            auto_generated=False,
        )
        
        assert tag.name == "machine-learning"
        assert tag.auto_generated is False


class TestReadingHistoryModel:
    """Tests for ReadingHistory model."""
    
    def test_reading_history_creation(self):
        """Test reading history creation."""
        history = ReadingHistory(
            paper_id="arxiv:2401.12345",
            action="viewed",
            duration_sec=120,
        )
        
        assert history.paper_id == "arxiv:2401.12345"
        assert history.action == "viewed"
        assert history.duration_sec == 120

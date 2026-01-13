"""Unit tests for ArXiv Agent database storage."""

import pytest
from datetime import datetime
from pathlib import Path

from arxiv_agent.data.models import Paper, Analysis, Collection, Tag
from arxiv_agent.data.storage import DatabaseManager


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test.db"
    db = DatabaseManager(db_path=db_path)
    yield db


def make_paper(suffix: str = "") -> Paper:
    """Create a fresh paper for testing."""
    return Paper(
        id=f"arxiv:2401.12345{suffix}",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        abstract="We propose a new simple network architecture...",
        categories=["cs.CL", "cs.LG"],
        published_date=datetime(2024, 1, 15),
        citation_count=50000,
    )


class TestDatabaseManager:
    """Tests for DatabaseManager class."""
    
    def test_database_creation(self, temp_db):
        """Test that database is created correctly."""
        assert temp_db.db_path.exists()
    
    def test_session_context_manager(self, temp_db):
        """Test session context manager."""
        with temp_db.session() as session:
            assert session is not None


class TestPaperOperations:
    """Tests for paper CRUD operations."""
    
    def test_save_and_get_paper(self, temp_db):
        """Test saving and retrieving a paper."""
        paper = make_paper()
        temp_db.save_paper(paper)
        
        retrieved = temp_db.get_paper("arxiv:2401.12345")
        assert retrieved is not None
    
    def test_get_nonexistent_paper(self, temp_db):
        """Test retrieving a paper that doesn't exist."""
        retrieved = temp_db.get_paper("arxiv:nonexistent")
        assert retrieved is None
    
    def test_delete_paper(self, temp_db):
        """Test deleting a paper."""
        paper = make_paper()
        temp_db.save_paper(paper)
        
        result = temp_db.delete_paper("arxiv:2401.12345")
        
        assert result is True
        assert temp_db.get_paper("arxiv:2401.12345") is None
    
    def test_search_papers(self, temp_db):
        """Test searching papers."""
        paper = make_paper()
        temp_db.save_paper(paper)
        
        results = temp_db.search_papers(query="attention")
        
        assert len(results) >= 1
    
    def test_list_papers(self, temp_db):
        """Test listing all papers."""
        paper = make_paper()
        temp_db.save_paper(paper)
        
        papers = temp_db.list_papers()
        
        assert len(papers) == 1
    
    def test_count_papers(self, temp_db):
        """Test counting papers."""
        assert temp_db.count_papers() == 0
        
        paper = make_paper()
        temp_db.save_paper(paper)
        
        assert temp_db.count_papers() == 1


class TestAnalysisOperations:
    """Tests for analysis operations."""
    
    def test_save_analysis(self, temp_db):
        """Test saving an analysis."""
        paper = make_paper()
        temp_db.save_paper(paper)
        
        analysis = Analysis(
            paper_id="arxiv:2401.12345",
            analysis_type="full",
            content={"text": "This is the analysis"},
            model_used="claude-sonnet-4-5-20250514",
        )
        
        temp_db.save_analysis(analysis)
        
        # Verify by getting
        latest = temp_db.get_latest_analysis("arxiv:2401.12345", "full")
        assert latest is not None


class TestCollectionOperations:
    """Tests for collection operations."""
    
    def test_create_collection(self, temp_db):
        """Test creating a collection."""
        temp_db.create_collection(
            name="To Read",
            description="Papers to read later",
        )
        
        # Verify
        collections = temp_db.list_collections()
        assert len(collections) == 1
    
    def test_list_collections(self, temp_db):
        """Test listing collections."""
        temp_db.create_collection(name="A Collection")
        temp_db.create_collection(name="B Collection")
        
        collections = temp_db.list_collections()
        
        assert len(collections) == 2
    
    def test_add_paper_to_collection(self, temp_db):
        """Test adding paper to collection."""
        paper = make_paper()
        temp_db.save_paper(paper)
        temp_db.create_collection(name="My Papers")
        
        # Get collection by name
        collection = temp_db.get_collection_by_name("My Papers")
        assert collection is not None
        
        # Use the ID directly instead of from returned object
        collections = temp_db.list_collections()
        assert len(collections) == 1
        # Get collection via get method which has id
        coll = temp_db.get_collection(1)  # First collection has ID 1
        result = temp_db.add_paper_to_collection("arxiv:2401.12345", 1)
        
        assert result is True


class TestTagOperations:
    """Tests for tag operations."""
    
    def test_get_or_create_tag(self, temp_db):
        """Test creating a tag."""
        temp_db.get_or_create_tag("machine-learning")
        
        tags = temp_db.list_tags()
        assert len(tags) == 1
    
    def test_add_tag_to_paper(self, temp_db):
        """Test adding tag to paper."""
        paper = make_paper()
        temp_db.save_paper(paper)
        
        # Create tag separately and use string-based approach
        result = temp_db.add_tag_to_paper("arxiv:2401.12345", "important")
        
        assert result is True
        
        # Verify tag was added
        paper_tags = temp_db.get_paper_tags("arxiv:2401.12345")
        assert len(paper_tags) == 1


class TestChatOperations:
    """Tests for chat operations."""
    
    def test_get_or_create_chat_session(self, temp_db):
        """Test creating chat session."""
        paper = make_paper()
        temp_db.save_paper(paper)
        
        session = temp_db.get_or_create_chat_session("arxiv:2401.12345")
        
        # Store ID before session might be detached
        session_id = session.id
        
        assert session_id is not None
    
    def test_add_chat_message(self, temp_db):
        """Test adding chat message."""
        paper = make_paper()
        temp_db.save_paper(paper)
        session = temp_db.get_or_create_chat_session("arxiv:2401.12345")
        
        # Store session_id immediately
        session_id = session.id
        
        temp_db.add_chat_message(
            session_id,
            "user",
            "What is this paper about?",
        )
        
        history = temp_db.get_chat_history(session_id)
        assert len(history) == 1


class TestReadingHistory:
    """Tests for reading history operations."""
    
    def test_record_interaction(self, temp_db):
        """Test recording an interaction."""
        paper = make_paper()
        temp_db.save_paper(paper)
        
        temp_db.record_interaction("arxiv:2401.12345", "viewed", duration_sec=60)
        
        history = temp_db.get_reading_history("arxiv:2401.12345")
        
        assert len(history) == 1


class TestUserPreferences:
    """Tests for user preferences."""
    
    def test_set_preference(self, temp_db):
        """Test setting a preference."""
        temp_db.set_preference("theme", {"name": "dark"})
        
        value = temp_db.get_preference("theme")
        
        assert value == {"name": "dark"}
    
    def test_update_preference(self, temp_db):
        """Test updating a preference."""
        temp_db.set_preference("theme", {"name": "light"})
        temp_db.set_preference("theme", {"name": "dark"})
        
        value = temp_db.get_preference("theme")
        
        assert value == {"name": "dark"}

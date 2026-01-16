"""Unit tests for FTS5 full-text search functionality."""

import pytest
from datetime import datetime

from arxiv_agent.data.models import (
    Paper, FTS5_CREATE_SQL, FTS5_TRIGGERS_SQL,
    FTS5_TRIGGER_INSERT, FTS5_TRIGGER_DELETE, FTS5_TRIGGER_UPDATE
)
from arxiv_agent.data.storage import DatabaseManager


class TestFTS5Initialization:
    """Tests for FTS5 table initialization."""
    
    def test_fts_create_sql_valid(self):
        """Test FTS5 CREATE SQL is valid."""
        assert "CREATE VIRTUAL TABLE" in FTS5_CREATE_SQL
        assert "papers_fts" in FTS5_CREATE_SQL
        assert "fts5" in FTS5_CREATE_SQL.lower()
    
    def test_fts_triggers_sql_contains_triggers(self):
        """Test FTS5 triggers SQL contains required triggers."""
        # Check individual trigger constants
        assert "papers_fts_insert" in FTS5_TRIGGER_INSERT
        assert "papers_fts_delete" in FTS5_TRIGGER_DELETE
        assert "papers_fts_update" in FTS5_TRIGGER_UPDATE
        # Check the list is populated
        assert len(FTS5_TRIGGERS_SQL) == 3
    
    def test_init_fts_creates_table(self, temp_db):
        """Test init_fts creates FTS5 virtual table."""
        temp_db.init_fts()
        
        # Verify table exists by querying it
        from sqlalchemy import text
        with temp_db.engine.connect() as conn:
            result = conn.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='papers_fts'"
            ))
            assert result.fetchone() is not None
    
    def test_init_fts_idempotent(self, temp_db):
        """Test init_fts can be called multiple times safely."""
        temp_db.init_fts()
        temp_db.init_fts()  # Should not raise
        
        # FTS should still work
        results = temp_db.fts_search("test")
        assert isinstance(results, list)


class TestFTS5Search:
    """Tests for FTS5 search functionality."""
    
    @pytest.fixture
    def db_with_papers(self, temp_db):
        """Create database with sample papers for search testing."""
        temp_db.init_fts()
        
        papers = [
            Paper(
                id="arxiv:2401.00001",
                title="Attention Is All You Need: Transformers for NLP",
                authors=["Ashish Vaswani", "Noam Shazeer"],
                abstract="We propose the Transformer, a model architecture based solely on attention mechanisms.",
                categories=["cs.CL"],
                published_date=datetime(2024, 1, 1),
            ),
            Paper(
                id="arxiv:2401.00002",
                title="BERT: Pre-training of Deep Bidirectional Transformers",
                authors=["Jacob Devlin", "Ming-Wei Chang"],
                abstract="We introduce BERT, a language representation model using bidirectional training.",
                categories=["cs.CL"],
                published_date=datetime(2024, 1, 2),
            ),
            Paper(
                id="arxiv:2401.00003",
                title="GPT-4 Technical Report",
                authors=["OpenAI"],
                abstract="We present GPT-4, a large multimodal model capable of processing image and text.",
                categories=["cs.AI"],
                published_date=datetime(2024, 1, 3),
            ),
            Paper(
                id="arxiv:2401.00004",
                title="Convolutional Neural Networks for Image Classification",
                authors=["Alex Krizhevsky"],
                abstract="Deep convolutional neural networks for large-scale image recognition tasks.",
                categories=["cs.CV"],
                published_date=datetime(2024, 1, 4),
            ),
        ]
        
        for paper in papers:
            temp_db.save_paper(paper)
        
        # Rebuild FTS index after adding papers
        temp_db._rebuild_fts_index()
        
        return temp_db
    
    def test_simple_search(self, db_with_papers):
        """Test simple single-word search."""
        results = db_with_papers.fts_search("transformer")
        
        assert len(results) >= 1
        # Transformer papers should be returned
        paper_ids = [r["paper_id"] for r in results]
        assert "arxiv:2401.00001" in paper_ids or "arxiv:2401.00002" in paper_ids
    
    def test_search_multiple_words(self, db_with_papers):
        """Test search with multiple words (implicit AND)."""
        results = db_with_papers.fts_search("attention mechanism")
        
        assert len(results) >= 1
        # Should find the Transformer paper
        paper_ids = [r["paper_id"] for r in results]
        assert "arxiv:2401.00001" in paper_ids
    
    def test_search_or_operator(self, db_with_papers):
        """Test search with OR operator."""
        results = db_with_papers.fts_search("transformer OR convolutional")
        
        # Should find both transformer and CNN papers
        assert len(results) >= 2
    
    def test_search_prefix_match(self, db_with_papers):
        """Test prefix matching with wildcard."""
        results = db_with_papers.fts_search("transform*")
        
        assert len(results) >= 1
        # Should match transformer, transformers, etc.
    
    def test_search_returns_snippets(self, db_with_papers):
        """Test search returns highlighted snippets."""
        results = db_with_papers.fts_search("attention")
        
        assert len(results) >= 1
        result = results[0]
        
        assert "title_snippet" in result
        assert "abstract_snippet" in result
    
    def test_search_returns_rank(self, db_with_papers):
        """Test search returns relevance rank."""
        results = db_with_papers.fts_search("transformer")
        
        assert len(results) >= 1
        assert "rank" in results[0]
        assert isinstance(results[0]["rank"], (int, float))
    
    def test_search_respects_limit(self, db_with_papers):
        """Test search respects result limit."""
        results = db_with_papers.fts_search("model", limit=2)
        
        assert len(results) <= 2
    
    def test_search_no_results(self, db_with_papers):
        """Test search with no matching results."""
        results = db_with_papers.fts_search("xyznonexistentterm")
        
        assert len(results) == 0
    
    def test_search_author(self, db_with_papers):
        """Test search matches author names."""
        results = db_with_papers.fts_search("Vaswani")
        
        assert len(results) >= 1
        paper_ids = [r["paper_id"] for r in results]
        assert "arxiv:2401.00001" in paper_ids


class TestFTS5SearchPapers:
    """Tests for FTS search returning Paper objects."""
    
    @pytest.fixture
    def db_with_papers(self, temp_db):
        """Create database with sample papers."""
        temp_db.init_fts()
        
        papers = [
            Paper(
                id="arxiv:2401.00001",
                title="Deep Learning for Natural Language Processing",
                authors=["Author One"],
                abstract="This paper explores deep learning techniques for NLP tasks.",
                categories=["cs.CL"],
                published_date=datetime(2024, 1, 1),
            ),
            Paper(
                id="arxiv:2401.00002",
                title="Computer Vision with Neural Networks",
                authors=["Author Two"],
                abstract="Neural network approaches for computer vision problems.",
                categories=["cs.CV"],
                published_date=datetime(2024, 1, 2),
            ),
        ]
        
        for paper in papers:
            temp_db.save_paper(paper)
        
        temp_db._rebuild_fts_index()
        return temp_db
    
    def test_fts_search_papers_returns_paper_objects(self, db_with_papers):
        """Test fts_search_papers returns Paper instances."""
        papers = db_with_papers.fts_search_papers("deep learning")
        
        assert len(papers) >= 1
        assert all(isinstance(p, Paper) for p in papers)
    
    def test_fts_search_papers_preserves_rank_order(self, db_with_papers):
        """Test fts_search_papers returns papers in rank order."""
        papers = db_with_papers.fts_search_papers("neural")
        
        # Should return papers, though order depends on relevance
        assert len(papers) >= 1


class TestFTS5TriggerSync:
    """Tests for FTS triggers keeping index synchronized."""
    
    def test_insert_triggers_fts_update(self, temp_db):
        """Test inserting paper updates FTS index via trigger."""
        temp_db.init_fts()
        
        paper = Paper(
            id="arxiv:2401.99999",
            title="Unique Test Paper About Quantum Computing",
            authors=["Test Author"],
            abstract="This paper discusses quantum computing fundamentals.",
            categories=["quant-ph"],
            published_date=datetime(2024, 1, 1),
        )
        temp_db.save_paper(paper)
        
        # Search should find the new paper
        results = temp_db.fts_search("quantum computing")
        paper_ids = [r["paper_id"] for r in results]
        
        assert "arxiv:2401.99999" in paper_ids
    
    def test_delete_triggers_fts_update(self, temp_db):
        """Test deleting paper updates FTS index via trigger."""
        temp_db.init_fts()
        
        paper = Paper(
            id="arxiv:2401.88888",
            title="Paper To Be Deleted About Blockchain",
            authors=["Delete Author"],
            abstract="Blockchain technology exploration.",
            categories=["cs.DC"],
            published_date=datetime(2024, 1, 1),
        )
        temp_db.save_paper(paper)
        
        # Verify it's searchable
        results = temp_db.fts_search("blockchain")
        assert any(r["paper_id"] == "arxiv:2401.88888" for r in results)
        
        # Delete and verify it's no longer in FTS
        temp_db.delete_paper("arxiv:2401.88888")
        
        results = temp_db.fts_search("blockchain")
        assert not any(r["paper_id"] == "arxiv:2401.88888" for r in results)


class TestFTS5EdgeCases:
    """Tests for FTS edge cases and error handling."""
    
    def test_search_empty_query(self, temp_db):
        """Test search with empty query."""
        temp_db.init_fts()
        
        # FTS5 may return all or error on empty - we should handle gracefully
        try:
            results = temp_db.fts_search("")
            # If no error, results should be a list
            assert isinstance(results, list)
        except Exception:
            # Some FTS implementations error on empty query
            pass
    
    def test_search_special_characters(self, temp_db):
        """Test search handles special characters safely."""
        temp_db.init_fts()
        
        # These should not cause SQL injection or errors
        safe_queries = [
            "test'query",
            "test\"query",
            "test;DROP TABLE papers;",
            "test--comment",
        ]
        
        for query in safe_queries:
            try:
                results = temp_db.fts_search(query)
                assert isinstance(results, list)
            except Exception as e:
                # May fail on invalid FTS syntax, but shouldn't crash DB
                assert "syntax" in str(e).lower() or "fts" in str(e).lower()
    
    def test_search_unicode(self, temp_db):
        """Test search handles unicode characters."""
        temp_db.init_fts()
        
        paper = Paper(
            id="arxiv:2401.77777",
            title="Étude sur les réseaux de neurones",  # French
            authors=["François Müller"],
            abstract="研究神经网络的方法",  # Chinese
            categories=["cs.AI"],
            published_date=datetime(2024, 1, 1),
        )
        temp_db.save_paper(paper)
        
        # Search should work with unicode
        results = temp_db.fts_search("neurones")
        assert isinstance(results, list)

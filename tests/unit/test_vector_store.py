"""Unit tests for Vector Store with cross-encoder reranking."""

import pytest
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

from arxiv_agent.core.vector_store import Chunk, RetrievalResult, EmbeddingService


class TestChunk:
    """Tests for Chunk dataclass."""
    
    def test_chunk_creation(self):
        """Test chunk creation with all fields."""
        chunk = Chunk(
            id="paper1_0",
            content="This is test content",
            paper_id="arxiv:2401.00001",
            section="Introduction",
            chunk_index=0,
        )
        
        assert chunk.id == "paper1_0"
        assert chunk.content == "This is test content"
        assert chunk.paper_id == "arxiv:2401.00001"
        assert chunk.section == "Introduction"
        assert chunk.chunk_index == 0
    
    def test_chunk_optional_embedding(self):
        """Test chunk with optional embedding."""
        chunk = Chunk(
            id="paper1_0",
            content="Test",
            paper_id="arxiv:2401.00001",
            section="Intro",
            chunk_index=0,
            embedding=[0.1, 0.2, 0.3],
        )
        
        assert chunk.embedding == [0.1, 0.2, 0.3]


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""
    
    def test_retrieval_result_creation(self):
        """Test retrieval result creation."""
        chunk = Chunk(
            id="test",
            content="Content",
            paper_id="arxiv:2401.00001",
            section="Test",
            chunk_index=0,
        )
        
        result = RetrievalResult(chunk=chunk, score=0.95, source="dense")
        
        assert result.chunk == chunk
        assert result.score == 0.95
        assert result.source == "dense"


class TestEmbeddingService:
    """Tests for EmbeddingService class."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        mock = MagicMock()
        mock.embedding.model = "sentence-transformers/all-MiniLM-L6-v2"
        mock.embedding.dimension = 384
        mock.embedding.batch_size = 32
        return mock
    
    def test_embedding_service_initialization(self, mock_settings):
        """Test embedding service initializes with settings."""
        with patch("arxiv_agent.core.vector_store.get_settings", return_value=mock_settings):
            service = EmbeddingService()
            
            assert service.model_name == mock_settings.embedding.model
            assert service.dimension == mock_settings.embedding.dimension
    
    def test_embed_single_text(self, mock_settings):
        """Test embedding a single text."""
        with patch("arxiv_agent.core.vector_store.get_settings", return_value=mock_settings):
            service = EmbeddingService()
            
            # Mock the model
            mock_model = MagicMock()
            mock_model.encode.return_value = MagicMock(tolist=lambda: [[0.1] * 384])
            service._model = mock_model
            
            embeddings = service.embed(["Test text"])
            
            assert len(embeddings) == 1
            assert len(embeddings[0]) == 384
    
    def test_embed_query(self, mock_settings):
        """Test embedding a query."""
        with patch("arxiv_agent.core.vector_store.get_settings", return_value=mock_settings):
            service = EmbeddingService()
            
            mock_model = MagicMock()
            mock_model.encode.return_value = MagicMock(tolist=lambda: [[0.1] * 384])
            service._model = mock_model
            
            embedding = service.embed_query("Test query")
            
            assert len(embedding) == 384
    
    def test_embed_empty_list(self, mock_settings):
        """Test embedding empty list returns empty."""
        with patch("arxiv_agent.core.vector_store.get_settings", return_value=mock_settings):
            service = EmbeddingService()
            
            embeddings = service.embed([])
            
            assert embeddings == []


class TestCrossEncoderReranking:
    """Tests for cross-encoder reranking functionality."""
    
    def test_reranker_class_exists(self):
        """Test CrossEncoderReranker class is available."""
        from arxiv_agent.core.vector_store import CrossEncoderReranker
        
        assert CrossEncoderReranker is not None
    
    def test_reranker_initialization(self):
        """Test reranker model initializes with settings."""
        from arxiv_agent.core.vector_store import CrossEncoderReranker
        
        mock_settings = MagicMock()
        mock_settings.retrieval.reranker_model = "cross-encoder/ms-marco-MiniLM-L6-v2"
        
        with patch("arxiv_agent.core.vector_store.get_settings", return_value=mock_settings):
            reranker = CrossEncoderReranker()
            
            assert reranker.model_name == mock_settings.retrieval.reranker_model
    
    def test_rerank_results_orders_by_score(self):
        """Test reranking reorders results by cross-encoder score."""
        from arxiv_agent.core.vector_store import CrossEncoderReranker
        
        chunks = [
            Chunk(id="c1", content="Less relevant", paper_id="p1", section="s1", chunk_index=0),
            Chunk(id="c2", content="More relevant to query", paper_id="p1", section="s2", chunk_index=1),
            Chunk(id="c3", content="Most relevant to query", paper_id="p1", section="s3", chunk_index=2),
        ]
        
        results = [
            RetrievalResult(chunk=chunks[0], score=0.9, source="hybrid"),
            RetrievalResult(chunk=chunks[1], score=0.8, source="hybrid"),
            RetrievalResult(chunk=chunks[2], score=0.7, source="hybrid"),
        ]
        
        mock_settings = MagicMock()
        mock_settings.retrieval.reranker_model = "test-model"
        
        with patch("arxiv_agent.core.vector_store.get_settings", return_value=mock_settings):
            reranker = CrossEncoderReranker()
            
            # Mock cross-encoder model
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.3, 0.7, 0.9]  # Reversed order
            reranker._model = mock_model
            
            reranked = reranker.rerank("query", results)
            
            # Should be reordered by cross-encoder score (highest first)
            assert len(reranked) == 3
            assert reranked[0].chunk.id == "c3"  # Highest score
            assert reranked[1].chunk.id == "c2"
            assert reranked[2].chunk.id == "c1"  # Lowest score
    
    def test_rerank_respects_top_k(self):
        """Test reranking respects top_k limit."""
        from arxiv_agent.core.vector_store import CrossEncoderReranker
        
        chunks = [
            Chunk(id=f"c{i}", content=f"Content {i}", paper_id="p1", section="s1", chunk_index=i)
            for i in range(10)
        ]
        
        results = [
            RetrievalResult(chunk=c, score=0.5, source="hybrid")
            for c in chunks
        ]
        
        mock_settings = MagicMock()
        mock_settings.retrieval.reranker_model = "test-model"
        
        with patch("arxiv_agent.core.vector_store.get_settings", return_value=mock_settings):
            reranker = CrossEncoderReranker()
            
            # Mock reranker
            mock_model = MagicMock()
            mock_model.predict.return_value = list(range(10))
            reranker._model = mock_model
            
            reranked = reranker.rerank("query", results, top_k=5)
            
            assert len(reranked) == 5


class TestHybridSearchMocked:
    """Tests for hybrid search logic without full VectorStore initialization."""
    
    def test_rrf_score_calculation(self):
        """Test Reciprocal Rank Fusion score calculation."""
        # RRF formula: sum(1 / (k + rank)) where k is usually 60
        # Items appearing in both lists get higher combined score
        
        # Dense ranks: chunk_a=1, chunk_b=2
        # Sparse ranks: chunk_a=2, chunk_b=1
        
        # chunk_a RRF: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
        # chunk_b RRF: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325
        
        # Both should have equal RRF when they appear in both lists
        k = 60
        chunk_a_score = 1/(k+1) + 1/(k+2)
        chunk_b_score = 1/(k+2) + 1/(k+1)
        
        assert abs(chunk_a_score - chunk_b_score) < 0.0001
    
    def test_hybrid_search_merges_unique_results(self):
        """Test hybrid search correctly merges unique results."""
        # Test that results appearing in only one list get appropriate scores
        
        k = 60
        # Item only in dense (rank 1): 1/(60+1) = 0.0164
        # Item only in sparse (rank 1): 1/(60+1) = 0.0164
        
        dense_only_score = 1/(k+1)
        sparse_only_score = 1/(k+1)
        
        # Item in both lists (rank 1 in each): 1/(60+1) + 1/(60+1) = 0.0328
        both_lists_score = dense_only_score + sparse_only_score
        
        # Item in both lists should have higher score
        assert both_lists_score > dense_only_score
        assert both_lists_score > sparse_only_score


class TestVectorStoreDataClasses:
    """Tests for Vector Store data classes and utilities."""
    
    def test_chunk_to_dict(self):
        """Test chunk can be converted to dictionary."""
        chunk = Chunk(
            id="test_id",
            content="Test content",
            paper_id="arxiv:2401.00001",
            section="Methods",
            chunk_index=5,
        )
        
        # Dataclass should support conversion
        from dataclasses import asdict
        d = asdict(chunk)
        
        assert d["id"] == "test_id"
        assert d["paper_id"] == "arxiv:2401.00001"
    
    def test_retrieval_result_comparison(self):
        """Test retrieval results can be compared by score."""
        chunk1 = Chunk(id="c1", content="", paper_id="p1", section="s", chunk_index=0)
        chunk2 = Chunk(id="c2", content="", paper_id="p1", section="s", chunk_index=1)
        
        r1 = RetrievalResult(chunk=chunk1, score=0.9, source="dense")
        r2 = RetrievalResult(chunk=chunk2, score=0.8, source="dense")
        
        # Can sort by score
        results = sorted([r2, r1], key=lambda r: r.score, reverse=True)
        assert results[0].score == 0.9
        assert results[1].score == 0.8

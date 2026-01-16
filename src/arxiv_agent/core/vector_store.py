"""Vector store with hybrid search (dense + sparse + reranking)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from arxiv_agent.config.settings import get_settings


@dataclass
class Chunk:
    """Document chunk with metadata."""
    
    id: str
    content: str
    paper_id: str
    section: str
    chunk_index: int
    embedding: list[float] | None = None


@dataclass
class RetrievalResult:
    """Search result with score."""
    
    chunk: Chunk
    score: float
    source: str  # 'dense', 'sparse', 'hybrid'


class EmbeddingService:
    """Manages text embeddings using sentence-transformers."""
    
    def __init__(self):
        """Initialize embedding service."""
        settings = get_settings()
        self.model_name = settings.embedding.model
        self.dimension = settings.embedding.dimension
        self.batch_size = settings.embedding.batch_size
        self._model = None
    
    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query.
        
        Args:
            query: Query string
        
        Returns:
            Embedding vector
        """
        return self.embed([query])[0]


class CrossEncoderReranker:
    """Cross-encoder model for reranking search results."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
        """Initialize cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self._model = CrossEncoder(self.model_name, max_length=512)
        return self._model
    
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank search results using cross-encoder.
        
        Args:
            query: Original search query
            results: List of RetrievalResult to rerank
            top_k: Number of results to return (None = all)
        
        Returns:
            Reranked list of RetrievalResult
        """
        if not results:
            return []
        
        # Create query-document pairs for cross-encoder
        pairs = [(query, result.chunk.content) for result in results]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs)
        
        # Create reranked results with new scores
        reranked = []
        for i, result in enumerate(results):
            reranked.append(RetrievalResult(
                chunk=result.chunk,
                score=float(scores[i]),
                source=f"{result.source}+rerank",
            ))
        
        # Sort by new scores (descending)
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        if top_k:
            reranked = reranked[:top_k]
        
        return reranked


class VectorStore:
    """ChromaDB-based vector store with hybrid search."""
    
    def __init__(self):
        """Initialize vector store."""
        settings = get_settings()
        
        persist_dir = str(settings.vector_db_path)
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        
        self.embedding_service = EmbeddingService()
        
        # Collections for different purposes
        self.chunks_collection = self.client.get_or_create_collection(
            name="paper_chunks",
            metadata={"hnsw:space": "cosine"},
        )
        
        self.papers_collection = self.client.get_or_create_collection(
            name="paper_summaries",
            metadata={"hnsw:space": "cosine"},
        )
        
        # BM25 index for sparse search (rebuilt on demand)
        self._bm25_index = None
        self._bm25_docs: list[str] = []
        self._bm25_ids: list[str] = []
        self._bm25_metadata: list[dict] = []
        
        # Cross-encoder for reranking (lazy loaded)
        self._reranker: CrossEncoderReranker | None = None
        
        logger.debug(f"Vector store initialized at {persist_dir}")
    
    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add document chunks to the vector store.
        
        Args:
            chunks: List of Chunk objects to add
        """
        if not chunks:
            return
        
        # Generate embeddings
        texts = [c.content for c in chunks]
        embeddings = self.embedding_service.embed(texts)
        
        # Prepare for ChromaDB
        ids = [c.id for c in chunks]
        metadatas = [
            {
                "paper_id": c.paper_id,
                "section": c.section,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]
        
        # Upsert to handle duplicates
        self.chunks_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        
        # Invalidate BM25 index
        self._bm25_index = None
        
        logger.info(f"Added {len(chunks)} chunks to vector store")
    
    def add_paper_summary(self, paper_id: str, title: str, abstract: str, metadata: dict | None = None) -> None:
        """Add paper summary for similarity search.
        
        Args:
            paper_id: Paper ID
            title: Paper title
            abstract: Paper abstract
            metadata: Additional metadata
        """
        text = f"{title}\n\n{abstract}"
        embedding = self.embedding_service.embed_query(text)
        
        meta = metadata or {}
        meta["paper_id"] = paper_id
        
        self.papers_collection.upsert(
            ids=[paper_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[meta],
        )
    
    def _build_bm25_index(self, paper_id: str | None = None) -> None:
        """Build BM25 index from documents.
        
        Args:
            paper_id: Optional paper ID to filter documents
        """
        from rank_bm25 import BM25Okapi
        
        # Get all documents
        where_filter = {"paper_id": paper_id} if paper_id else None
        results = self.chunks_collection.get(
            where=where_filter,
            include=["documents", "metadatas"],
        )
        
        if not results["documents"]:
            return
        
        self._bm25_docs = results["documents"]
        self._bm25_ids = results["ids"]
        self._bm25_metadata = results["metadatas"]
        
        # Tokenize for BM25
        tokenized = [doc.lower().split() for doc in self._bm25_docs]
        self._bm25_index = BM25Okapi(tokenized)
        
        logger.debug(f"Built BM25 index with {len(self._bm25_docs)} documents")
    
    def dense_search(
        self,
        query: str,
        paper_id: str | None = None,
        k: int = 20,
    ) -> list[RetrievalResult]:
        """Semantic search using embeddings.
        
        Args:
            query: Search query
            paper_id: Optional paper ID to filter results
            k: Number of results to return
        
        Returns:
            List of RetrievalResult objects
        """
        query_embedding = self.embedding_service.embed_query(query)
        
        where_filter = {"paper_id": paper_id} if paper_id else None
        
        results = self.chunks_collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        
        if not results["ids"] or not results["ids"][0]:
            return []
        
        retrieval_results = []
        for i, doc_id in enumerate(results["ids"][0]):
            chunk = Chunk(
                id=doc_id,
                content=results["documents"][0][i],
                paper_id=results["metadatas"][0][i]["paper_id"],
                section=results["metadatas"][0][i]["section"],
                chunk_index=results["metadatas"][0][i]["chunk_index"],
            )
            # Convert distance to similarity score (1 - distance for cosine)
            score = 1 - results["distances"][0][i]
            retrieval_results.append(RetrievalResult(
                chunk=chunk,
                score=score,
                source="dense",
            ))
        
        return retrieval_results
    
    def sparse_search(
        self,
        query: str,
        paper_id: str | None = None,
        k: int = 20,
    ) -> list[RetrievalResult]:
        """BM25 keyword search.
        
        Args:
            query: Search query
            paper_id: Optional paper ID to filter results
            k: Number of results to return
        
        Returns:
            List of RetrievalResult objects
        """
        # Rebuild index if needed
        if self._bm25_index is None:
            self._build_bm25_index(paper_id)
        
        if not self._bm25_index:
            return []
        
        tokenized_query = query.lower().split()
        scores = self._bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k * 2]  # Get more for filtering
        
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            
            doc_id = self._bm25_ids[idx]
            doc = self._bm25_docs[idx]
            meta = self._bm25_metadata[idx]
            
            # Filter by paper_id if specified
            if paper_id and meta["paper_id"] != paper_id:
                continue
            
            chunk = Chunk(
                id=doc_id,
                content=doc,
                paper_id=meta["paper_id"],
                section=meta["section"],
                chunk_index=meta["chunk_index"],
            )
            results.append(RetrievalResult(
                chunk=chunk,
                score=scores[idx],
                source="sparse",
            ))
            
            if len(results) >= k:
                break
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        paper_id: str | None = None,
        k: int = 20,
        dense_weight: float | None = None,
    ) -> list[RetrievalResult]:
        """Hybrid search combining dense and sparse retrieval with RRF.
        
        Args:
            query: Search query
            paper_id: Optional paper ID to filter results
            k: Number of results to return
            dense_weight: Weight for dense results (default from settings)
        
        Returns:
            List of RetrievalResult objects sorted by RRF score
        """
        settings = get_settings()
        dense_weight = dense_weight or settings.retrieval.dense_weight
        sparse_weight = 1 - dense_weight
        
        # Get results from both methods
        dense_results = self.dense_search(query, paper_id, k * 2)
        sparse_results = self.sparse_search(query, paper_id, k * 2)
        
        # Reciprocal Rank Fusion
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}
        
        k_constant = 60  # RRF constant
        
        for rank, result in enumerate(dense_results):
            chunk_id = result.chunk.id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + dense_weight / (k_constant + rank + 1)
            chunk_map[chunk_id] = result.chunk
        
        for rank, result in enumerate(sparse_results):
            chunk_id = result.chunk.id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + sparse_weight / (k_constant + rank + 1)
            chunk_map[chunk_id] = result.chunk
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        return [
            RetrievalResult(
                chunk=chunk_map[chunk_id],
                score=rrf_scores[chunk_id],
                source="hybrid",
            )
            for chunk_id in sorted_ids[:k]
        ]
    
    @property
    def reranker(self) -> CrossEncoderReranker:
        """Get or create cross-encoder reranker."""
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker
    
    def rerank_results(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank search results using cross-encoder.
        
        Args:
            query: Original search query
            results: List of RetrievalResult to rerank
            top_k: Number of results to return (None = all)
        
        Returns:
            Reranked list of RetrievalResult
        """
        return self.reranker.rerank(query, results, top_k)
    
    def hybrid_search_with_rerank(
        self,
        query: str,
        paper_id: str | None = None,
        k: int = 10,
        rerank_top_k: int = 50,
        dense_weight: float | None = None,
    ) -> list[RetrievalResult]:
        """Hybrid search with cross-encoder reranking for best quality.
        
        Two-stage retrieval:
        1. Retrieve rerank_top_k candidates using hybrid search
        2. Rerank candidates using cross-encoder and return top k
        
        Args:
            query: Search query
            paper_id: Optional paper ID to filter results
            k: Number of final results to return
            rerank_top_k: Number of candidates to rerank
            dense_weight: Weight for dense results in initial retrieval
        
        Returns:
            List of RetrievalResult sorted by cross-encoder score
        """
        # Stage 1: Get candidates via hybrid search
        candidates = self.hybrid_search(query, paper_id, rerank_top_k, dense_weight)
        
        if not candidates:
            return []
        
        # Stage 2: Rerank using cross-encoder
        reranked = self.rerank_results(query, candidates, top_k=k)
        
        logger.debug(f"Reranked {len(candidates)} candidates to top {k}")
        return reranked
    
    def search_similar_papers(
        self,
        query: str,
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search for similar papers by abstract/title similarity.
        
        Args:
            query: Search query
            k: Number of results
        
        Returns:
            List of (paper_id, score) tuples
        """
        query_embedding = self.embedding_service.embed_query(query)
        
        results = self.papers_collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["metadatas", "distances"],
        )
        
        if not results["ids"] or not results["ids"][0]:
            return []
        
        return [
            (results["metadatas"][0][i]["paper_id"], 1 - results["distances"][0][i])
            for i in range(len(results["ids"][0]))
        ]
    
    def delete_paper(self, paper_id: str) -> None:
        """Delete all chunks for a paper.
        
        Args:
            paper_id: Paper ID to delete
        """
        self.chunks_collection.delete(where={"paper_id": paper_id})
        
        try:
            self.papers_collection.delete(ids=[paper_id])
        except Exception:
            pass  # May not exist
        
        self._bm25_index = None  # Invalidate
        logger.info(f"Deleted chunks for paper {paper_id}")
    
    def get_chunk_count(self, paper_id: str | None = None) -> int:
        """Get the number of chunks.
        
        Args:
            paper_id: Optional paper ID to filter
        
        Returns:
            Number of chunks
        """
        where_filter = {"paper_id": paper_id} if paper_id else None
        return self.chunks_collection.count(where=where_filter)


# Global instance
_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Get or create vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def reset_vector_store() -> None:
    """Reset vector store instance."""
    global _vector_store
    _vector_store = None

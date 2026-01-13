"""API client manager for arXiv and Semantic Scholar."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import arxiv
import httpx
from diskcache import Cache
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from arxiv_agent.config.settings import get_settings
from arxiv_agent.data.models import Paper


class RateLimiter:
    """Token bucket rate limiter for API calls."""
    
    def __init__(self, rate: float, per: float = 1.0):
        """Initialize rate limiter.
        
        Args:
            rate: Number of requests allowed
            per: Time period in seconds
        """
        self.rate = rate
        self.per = per
        self.tokens = rate
        self.last_update = datetime.now()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = datetime.now()
            elapsed = (now - self.last_update).total_seconds()
            self.tokens = min(self.rate, self.tokens + elapsed * (self.rate / self.per))
            self.last_update = now
            
            if self.tokens < 1:
                wait_time = (1 - self.tokens) * (self.per / self.rate)
                logger.debug(f"Rate limited, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class APIClientManager:
    """Unified API client with caching and rate limiting."""
    
    def __init__(self):
        """Initialize API client manager."""
        settings = get_settings()
        
        # HTTP client for async requests
        self.http = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "arxiv-agent/0.1.0 (research tool)"},
        )
        
        # arXiv client
        self.arxiv_client = arxiv.Client(
            page_size=100,
            delay_seconds=3,  # Required by arXiv
            num_retries=3,
        )
        
        # Rate limiters
        self.arxiv_limiter = RateLimiter(rate=1, per=3)  # 1 request per 3 seconds
        self.semantic_scholar_limiter = RateLimiter(rate=1, per=1)  # 1 request per second
        
        # Disk cache for API responses
        cache_dir = settings.cache_dir / "api_responses"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = Cache(str(cache_dir))
        self.cache_ttl = 86400  # 24 hours
        
        # Semantic Scholar API key
        self.ss_api_key = (
            settings.semantic_scholar_api_key.get_secret_value()
            if settings.semantic_scholar_api_key
            else None
        )
    
    def _cache_key(self, prefix: str, *args: Any) -> str:
        """Generate a cache key."""
        return f"{prefix}:{':'.join(str(a) for a in args)}"
    
    # ==================== arXiv API ====================
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=3, max=30),
        retry=retry_if_exception_type((httpx.TimeoutException,)),
    )
    async def search_arxiv(
        self,
        query: str,
        max_results: int = 10,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate,
        categories: list[str] | None = None,
        date_from: datetime | None = None,
    ) -> list[Paper]:
        """Search arXiv for papers.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            sort_by: Sort criterion
            categories: Optional list of arXiv categories to filter
            date_from: Optional date to filter papers from
        
        Returns:
            List of Paper objects
        """
        await self.arxiv_limiter.acquire()
        
        # Build query
        parts = [query] if query and query != "*" else []
        if categories:
            cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
            parts.append(f"({cat_query})")
        
        full_query = " AND ".join(parts) if parts else "all:*"
        
        if date_from:
            date_str = date_from.strftime("%Y%m%d")
            full_query = f"{full_query} AND submittedDate:[{date_str}0000 TO 99991231235959]"
        
        logger.debug(f"arXiv query: {full_query}")
        
        search = arxiv.Search(
            query=full_query,
            max_results=max_results,
            sort_by=sort_by,
        )
        
        papers = []
        try:
            for result in self.arxiv_client.results(search):
                # Extract arxiv ID from entry_id
                arxiv_id = result.entry_id.split("/")[-1]
                # Remove version suffix if present
                if "v" in arxiv_id:
                    arxiv_id = arxiv_id.split("v")[0]
                
                paper = Paper(
                    id=f"arxiv:{arxiv_id}",
                    title=result.title.replace("\n", " "),
                    authors=[a.name for a in result.authors],
                    abstract=result.summary.replace("\n", " "),
                    categories=list(result.categories),
                    published_date=result.published,
                    pdf_path=None,
                )
                papers.append(paper)
        except Exception as e:
            logger.error(f"arXiv search failed: {e}")
            raise
        
        logger.info(f"Found {len(papers)} papers for query: {query[:50]}...")
        return papers
    
    async def get_paper_by_id(self, arxiv_id: str) -> Paper | None:
        """Get a specific paper by arXiv ID.
        
        Args:
            arxiv_id: arXiv paper ID (with or without 'arxiv:' prefix)
        
        Returns:
            Paper object or None if not found
        """
        # Normalize ID
        if arxiv_id.startswith("arxiv:"):
            arxiv_id = arxiv_id[6:]
        
        await self.arxiv_limiter.acquire()
        
        search = arxiv.Search(id_list=[arxiv_id])
        
        try:
            result = next(self.arxiv_client.results(search), None)
            if not result:
                return None
            
            return Paper(
                id=f"arxiv:{arxiv_id}",
                title=result.title.replace("\n", " "),
                authors=[a.name for a in result.authors],
                abstract=result.summary.replace("\n", " "),
                categories=list(result.categories),
                published_date=result.published,
                pdf_path=None,
            )
        except Exception as e:
            logger.error(f"Failed to fetch paper {arxiv_id}: {e}")
            return None
    
    async def download_pdf(self, paper: Paper, output_dir: Path | str) -> str:
        """Download PDF for a paper.
        
        Args:
            paper: Paper object
            output_dir: Directory to save PDF
        
        Returns:
            Path to downloaded PDF
        """
        arxiv_id = paper.id.replace("arxiv:", "")
        
        # Check cache for existing path
        cache_key = self._cache_key("pdf_path", paper.id)
        if cached := self.cache.get(cache_key):
            if Path(cached).exists():
                logger.debug(f"Using cached PDF: {cached}")
                return cached
        
        await self.arxiv_limiter.acquire()
        
        search = arxiv.Search(id_list=[arxiv_id])
        result = next(self.arxiv_client.results(search), None)
        
        if not result:
            raise ValueError(f"Paper not found: {arxiv_id}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"arxiv_{arxiv_id.replace('.', '_').replace('/', '_')}.pdf"
        filepath = output_path / filename
        
        logger.info(f"Downloading PDF for {paper.id}...")
        result.download_pdf(dirpath=str(output_path), filename=filename)
        
        # Cache the path
        self.cache.set(cache_key, str(filepath), expire=self.cache_ttl * 30)  # 30 days
        
        return str(filepath)
    
    # ==================== Semantic Scholar API ====================
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
    )
    async def enrich_with_semantic_scholar(self, paper: Paper) -> Paper:
        """Enrich paper with Semantic Scholar data (citations, TLDR).
        
        Args:
            paper: Paper object to enrich
        
        Returns:
            Enriched Paper object
        """
        cache_key = self._cache_key("ss", paper.id)
        if cached := self.cache.get(cache_key):
            paper.citation_count = cached.get("citationCount", 0)
            if tldr := cached.get("tldr"):
                paper.tldr = tldr.get("text")
            return paper
        
        await self.semantic_scholar_limiter.acquire()
        
        arxiv_id = paper.id.replace("arxiv:", "")
        url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
        params = {"fields": "citationCount,tldr,influentialCitationCount"}
        
        headers = {}
        if self.ss_api_key:
            headers["x-api-key"] = self.ss_api_key
        
        try:
            response = await self.http.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            paper.citation_count = data.get("citationCount", 0)
            if tldr := data.get("tldr"):
                paper.tldr = tldr.get("text")
            
            # Cache the response
            self.cache.set(cache_key, data, expire=self.cache_ttl)
            logger.debug(f"Enriched {paper.id}: {paper.citation_count} citations")
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug(f"Paper {paper.id} not found in Semantic Scholar")
            else:
                logger.warning(f"Semantic Scholar API error: {e}")
        except Exception as e:
            logger.warning(f"Failed to enrich {paper.id}: {e}")
        
        return paper
    
    async def get_trending_papers(
        self,
        categories: list[str],
        days: int = 7,
        limit: int = 20,
    ) -> list[Paper]:
        """Get trending papers based on citation velocity.
        
        Args:
            categories: arXiv categories to search
            days: Number of days to look back
            limit: Maximum number of papers to return
        
        Returns:
            List of papers sorted by citations
        """
        date_from = datetime.now() - timedelta(days=days)
        
        # Search recent papers
        papers = await self.search_arxiv(
            query="*",
            max_results=100,
            categories=categories,
            date_from=date_from,
        )
        
        # Enrich with citation data
        enriched = []
        for paper in papers:
            try:
                enriched.append(await self.enrich_with_semantic_scholar(paper))
            except Exception as e:
                logger.warning(f"Failed to enrich {paper.id}: {e}")
                enriched.append(paper)
        
        # Sort by citations
        enriched.sort(key=lambda p: p.citation_count, reverse=True)
        return enriched[:limit]
    
    async def close(self) -> None:
        """Close HTTP client and cleanup."""
        await self.http.aclose()
        self.cache.close()


# Global instance
_client: APIClientManager | None = None


def get_api_client() -> APIClientManager:
    """Get or create API client instance."""
    global _client
    if _client is None:
        _client = APIClientManager()
    return _client


def reset_api_client() -> None:
    """Reset API client instance."""
    global _client
    _client = None

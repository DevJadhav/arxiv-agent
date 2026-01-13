"""Fetcher agent for retrieving papers from external APIs."""

from datetime import datetime, timedelta

from loguru import logger

from arxiv_agent.agents.state import AgentState
from arxiv_agent.config.settings import get_settings
from arxiv_agent.core.api_client import get_api_client
from arxiv_agent.data.storage import get_db


class FetcherAgent:
    """Agent responsible for fetching papers from external APIs.
    
    Handles:
    - arXiv search queries
    - Single paper fetching
    - Daily digest paper collection
    - Semantic Scholar enrichment
    - PDF downloads
    """
    
    def __init__(self):
        """Initialize fetcher agent."""
        self.api_client = get_api_client()
        self.settings = get_settings()
        self.db = get_db()
    
    async def run(self, state: AgentState) -> AgentState:
        """Execute fetcher agent.
        
        Args:
            state: Current agent state
        
        Returns:
            Updated agent state
        """
        state.current_step = "fetching"
        logger.info(f"Fetcher agent running for task: {state.task_type}")
        
        try:
            if state.task_type == "search":
                await self._search_papers(state)
            elif state.task_type == "digest":
                await self._fetch_digest_papers(state)
            elif state.paper_id:
                await self._fetch_single_paper(state)
            else:
                state.add_error("No valid fetch operation specified")
        except Exception as e:
            logger.error(f"Fetcher error: {e}")
            state.add_error(f"Fetch failed: {str(e)}")
        
        return state
    
    async def _search_papers(self, state: AgentState) -> None:
        """Search for papers matching query.
        
        Args:
            state: Current agent state
        """
        if not state.query:
            state.add_error("No search query provided")
            return
        
        # Get search options
        limit = state.options.get("limit", 10)
        categories = state.options.get("categories") or self.settings.digest.categories
        since = state.options.get("since")
        
        date_from = None
        if since:
            if isinstance(since, str):
                date_from = datetime.fromisoformat(since)
            else:
                date_from = since
        
        logger.info(f"Searching arXiv for: {state.query}")
        
        papers = await self.api_client.search_arxiv(
            query=state.query,
            max_results=limit,
            categories=categories if categories else None,
            date_from=date_from,
        )
        
        # Enrich with Semantic Scholar data
        enriched = await self._enrich_papers(papers)
        
        state.papers = enriched
        
        # Save to database
        for paper in enriched:
            self.db.save_paper(paper)
            self.db.record_interaction(paper.id, "viewed")
        
        logger.info(f"Found and saved {len(enriched)} papers")
    
    async def _fetch_digest_papers(self, state: AgentState) -> None:
        """Fetch papers for daily digest.
        
        Args:
            state: Current agent state
        """
        date_from = datetime.now() - timedelta(days=1)
        
        # Build query from configured keywords
        keywords = self.settings.digest.keywords
        query = " OR ".join(keywords) if keywords else "*"
        
        logger.info(f"Fetching digest papers with keywords: {keywords}")
        
        papers = await self.api_client.search_arxiv(
            query=query,
            max_results=self.settings.digest.max_papers * 2,  # Get extra for filtering
            categories=self.settings.digest.categories,
            date_from=date_from,
        )
        
        # Enrich with citation data
        enriched = await self._enrich_papers(papers)
        
        # Filter by minimum citation count if configured
        if self.settings.digest.min_citation_count > 0:
            enriched = [
                p for p in enriched
                if p.citation_count >= self.settings.digest.min_citation_count
            ]
        
        # Sort by citations and take top N
        enriched.sort(key=lambda p: p.citation_count, reverse=True)
        state.papers = enriched[:self.settings.digest.max_papers]
        
        # Save papers
        for paper in state.papers:
            self.db.save_paper(paper)
        
        logger.info(f"Digest prepared with {len(state.papers)} papers")
    
    async def _fetch_single_paper(self, state: AgentState) -> None:
        """Fetch a single paper by ID.
        
        Args:
            state: Current agent state
        """
        paper_id = state.paper_id
        if not paper_id:
            state.add_error("No paper ID provided")
            return
        
        # Normalize ID
        if not paper_id.startswith("arxiv:"):
            paper_id = f"arxiv:{paper_id}"
        
        # Check database first
        existing = self.db.get_paper(paper_id)
        if existing:
            logger.debug(f"Found paper in database: {paper_id}")
            state.current_paper = existing
            state.papers = [existing]
            self.db.record_interaction(paper_id, "viewed")
            return
        
        # Fetch from API
        logger.info(f"Fetching paper from arXiv: {paper_id}")
        paper = await self.api_client.get_paper_by_id(paper_id)
        
        if paper:
            paper = await self.api_client.enrich_with_semantic_scholar(paper)
            self.db.save_paper(paper)
            state.current_paper = paper
            state.papers = [paper]
            self.db.record_interaction(paper_id, "viewed")
        else:
            state.add_error(f"Paper not found: {paper_id}")
    
    async def _enrich_papers(self, papers: list) -> list:
        """Enrich papers with Semantic Scholar data.
        
        Args:
            papers: List of Paper objects
        
        Returns:
            List of enriched Paper objects
        """
        enriched = []
        for paper in papers:
            try:
                enriched.append(
                    await self.api_client.enrich_with_semantic_scholar(paper)
                )
            except Exception as e:
                logger.warning(f"Failed to enrich {paper.id}: {e}")
                enriched.append(paper)
        return enriched
    
    async def download_pdf_for_paper(self, state: AgentState) -> str | None:
        """Download PDF for the current paper.
        
        Args:
            state: Current agent state
        
        Returns:
            Path to downloaded PDF or None
        """
        paper = state.current_paper
        if not paper:
            return None
        
        if paper.pdf_path:
            from pathlib import Path
            if Path(paper.pdf_path).exists():
                return paper.pdf_path
        
        # Determine output directory
        if paper.published_date:
            pdf_dir = self.settings.get_pdf_dir(
                paper.published_date.year,
                paper.published_date.month,
            )
        else:
            pdf_dir = self.settings.data_dir / "pdfs" / "unknown"
        
        try:
            pdf_path = await self.api_client.download_pdf(paper, pdf_dir)
            paper.pdf_path = pdf_path
            self.db.save_paper(paper)
            return pdf_path
        except Exception as e:
            logger.error(f"Failed to download PDF: {e}")
            return None

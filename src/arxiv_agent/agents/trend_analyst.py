"""Trend Analyst agent for discovering trending topics."""

from datetime import datetime, timedelta

from loguru import logger

from arxiv_agent.agents.state import AgentState
from arxiv_agent.config.settings import get_settings
from arxiv_agent.core.api_client import get_api_client
from arxiv_agent.data.storage import get_db


class TrendAnalystAgent:
    """Agent for trend discovery and recommendations.
    
    Handles:
    - Trending topics analysis
    - Citation-based paper ranking
    - Personalized recommendations
    """
    
    def __init__(self):
        """Initialize trend analyst agent."""
        self.api_client = get_api_client()
        self.settings = get_settings()
        self.db = get_db()
    
    async def run(self, state: AgentState) -> AgentState:
        """Execute trend analyst agent.
        
        Args:
            state: Current agent state
        
        Returns:
            Updated agent state
        """
        state.current_step = "trends"
        logger.info("Trend analyst agent running")
        
        try:
            action = state.options.get("action", "trending")
            
            if action == "trending":
                await self._get_trending_papers(state)
            elif action == "recommend":
                await self._get_recommendations(state)
            else:
                state.add_error(f"Unknown trends action: {action}")
                
        except Exception as e:
            logger.error(f"Trend analyst error: {e}")
            state.add_error(f"Trends analysis failed: {str(e)}")
        
        return state
    
    async def _get_trending_papers(self, state: AgentState) -> None:
        """Get trending papers.
        
        Args:
            state: Current agent state
        """
        categories = state.options.get("categories") or self.settings.digest.categories
        days = state.options.get("days", 7)
        limit = state.options.get("limit", 20)
        
        logger.info(f"Getting trending papers from last {days} days")
        
        papers = await self.api_client.get_trending_papers(
            categories=categories,
            days=days,
            limit=limit,
        )
        
        state.papers = papers
        
        # Save papers to database
        for paper in papers:
            self.db.save_paper(paper)
        
        logger.info(f"Found {len(papers)} trending papers")
    
    async def _get_recommendations(self, state: AgentState) -> None:
        """Get personalized recommendations based on reading history.
        
        Args:
            state: Current agent state
        """
        based_on = state.options.get("based_on", "library")
        limit = state.options.get("limit", 10)
        
        # Get user's papers from library
        library_papers = self.db.list_papers(limit=50)
        
        if not library_papers:
            # Fall back to trending if no library
            await self._get_trending_papers(state)
            return
        
        # Build preference keywords from library
        keywords = set()
        categories = set()
        
        for paper in library_papers[:20]:
            # Extract keywords from titles
            title_words = paper.title.lower().split()
            keywords.update(w for w in title_words if len(w) > 4)
            
            # Collect categories
            categories.update(paper.categories)
        
        # Search for similar papers
        query = " OR ".join(list(keywords)[:10])
        
        papers = await self.api_client.search_arxiv(
            query=query,
            max_results=limit * 2,
            categories=list(categories) if categories else None,
            date_from=datetime.now() - timedelta(days=30),
        )
        
        # Filter out papers already in library
        library_ids = {p.id for p in library_papers}
        new_papers = [p for p in papers if p.id not in library_ids]
        
        # Enrich and sort
        enriched = []
        for paper in new_papers[:limit]:
            try:
                enriched.append(
                    await self.api_client.enrich_with_semantic_scholar(paper)
                )
            except Exception:
                enriched.append(paper)
        
        enriched.sort(key=lambda p: p.citation_count, reverse=True)
        state.papers = enriched[:limit]
        
        logger.info(f"Generated {len(state.papers)} recommendations")

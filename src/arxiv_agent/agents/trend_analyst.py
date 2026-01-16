"""Trend Analyst agent for discovering trending topics."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

from arxiv_agent.agents.state import AgentState
from arxiv_agent.config.settings import get_settings
from arxiv_agent.core.api_client import get_api_client
from arxiv_agent.data.models import Paper
from arxiv_agent.data.storage import get_db


@dataclass
class TopicInfo:
    """Information about an extracted topic."""
    
    id: int
    name: str
    keywords: list[str]
    paper_count: int
    representative_papers: list[str]  # Paper IDs
    trend_score: float = 0.0
    
    def __repr__(self) -> str:
        return f"Topic({self.id}: {self.name}, {self.paper_count} papers)"


class BERTopicExtractor:
    """Topic extraction using BERTopic.
    
    Uses sentence-transformers for embeddings and BERTopic for clustering
    to discover emerging research topics from paper abstracts.
    """
    
    def __init__(self, min_topic_size: int = 5, nr_topics: Optional[int] = None):
        """Initialize BERTopic extractor.
        
        Args:
            min_topic_size: Minimum number of papers per topic
            nr_topics: Target number of topics (None for auto)
        """
        self.min_topic_size = min_topic_size
        self.nr_topics = nr_topics
        self._model = None
    
    @property
    def model(self):
        """Lazy load BERTopic model with optimized settings."""
        if self._model is None:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            
            logger.info("Initializing BERTopic model...")
            
            # Use the same embedding model as vector store for consistency
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            
            # Configure BERTopic
            self._model = BERTopic(
                embedding_model=embedding_model,
                min_topic_size=self.min_topic_size,
                nr_topics=self.nr_topics,
                calculate_probabilities=True,
                verbose=False,
            )
        return self._model
    
    def extract_topics(
        self,
        papers: list[Paper],
        use_abstracts: bool = True,
    ) -> tuple[list[TopicInfo], dict[str, int]]:
        """Extract topics from a collection of papers.
        
        Args:
            papers: List of Paper objects to analyze
            use_abstracts: Use full abstracts (True) or just titles (False)
        
        Returns:
            Tuple of (list of TopicInfo, dict mapping paper_id to topic_id)
        """
        if not papers:
            return [], {}
        
        # Prepare documents
        if use_abstracts:
            docs = [f"{p.title}\n\n{p.abstract}" for p in papers]
        else:
            docs = [p.title for p in papers]
        
        paper_ids = [p.id for p in papers]
        
        logger.info(f"Extracting topics from {len(papers)} papers...")
        
        # Fit BERTopic
        topics, probs = self.model.fit_transform(docs)
        
        # Build paper-to-topic mapping
        paper_topics = {paper_ids[i]: topics[i] for i in range(len(papers))}
        
        # Get topic info
        topic_info = self.model.get_topic_info()
        
        # Build TopicInfo objects
        extracted = []
        for _, row in topic_info.iterrows():
            topic_id = row["Topic"]
            if topic_id == -1:
                continue  # Skip outliers
            
            # Get keywords for this topic
            topic_words = self.model.get_topic(topic_id)
            keywords = [word for word, _ in topic_words[:10]] if topic_words else []
            
            # Get representative papers
            topic_papers = [pid for pid, tid in paper_topics.items() if tid == topic_id]
            
            # Generate human-readable topic name from top keywords
            name = self._generate_topic_name(keywords)
            
            extracted.append(TopicInfo(
                id=topic_id,
                name=name,
                keywords=keywords,
                paper_count=row["Count"],
                representative_papers=topic_papers[:5],
            ))
        
        # Sort by paper count
        extracted.sort(key=lambda t: t.paper_count, reverse=True)
        
        logger.info(f"Extracted {len(extracted)} topics")
        return extracted, paper_topics
    
    def _generate_topic_name(self, keywords: list[str]) -> str:
        """Generate a human-readable topic name from keywords."""
        if not keywords:
            return "Unknown Topic"
        
        # Use top 3 keywords for name
        top_words = keywords[:3]
        return " / ".join(top_words).title()
    
    def calculate_trend_scores(
        self,
        papers: list[Paper],
        paper_topics: dict[str, int],
        topics: list[TopicInfo],
        recent_days: int = 7,
    ) -> list[TopicInfo]:
        """Calculate trend scores based on recency and growth.
        
        Args:
            papers: List of papers
            paper_topics: Mapping of paper_id to topic_id
            topics: List of topics to score
            recent_days: Days to consider as "recent"
        
        Returns:
            Topics with updated trend_scores, sorted by score
        """
        from collections import defaultdict
        
        cutoff = datetime.now() - timedelta(days=recent_days)
        
        # Count recent vs older papers per topic
        recent_counts = defaultdict(int)
        older_counts = defaultdict(int)
        
        for paper in papers:
            topic_id = paper_topics.get(paper.id, -1)
            if topic_id == -1:
                continue
            
            pub_date = paper.published_date or datetime.now()
            if pub_date >= cutoff:
                recent_counts[topic_id] += 1
            else:
                older_counts[topic_id] += 1
        
        # Calculate trend scores
        for topic in topics:
            recent = recent_counts.get(topic.id, 0)
            older = older_counts.get(topic.id, 0)
            
            # Trend score: ratio of recent to older, weighted by volume
            if older > 0:
                growth = recent / older
            else:
                growth = recent * 2  # New topics get bonus
            
            # Combine growth with volume
            topic.trend_score = growth * (1 + topic.paper_count / 100)
        
        # Sort by trend score
        topics.sort(key=lambda t: t.trend_score, reverse=True)
        return topics


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
    
    async def extract_topics(
        self,
        papers: Optional[list[Paper]] = None,
        categories: Optional[list[str]] = None,
        days: int = 30,
        min_topic_size: int = 5,
    ) -> tuple[list[TopicInfo], dict[str, int]]:
        """Extract emerging topics from recent papers using BERTopic.
        
        Args:
            papers: Pre-fetched papers (if None, will fetch from API)
            categories: ArXiv categories to analyze
            days: Days of papers to analyze
            min_topic_size: Minimum papers per topic
        
        Returns:
            Tuple of (list of TopicInfo, paper-to-topic mapping)
        """
        # Fetch papers if not provided
        if papers is None:
            categories = categories or self.settings.digest.categories
            papers = await self.api_client.search_arxiv(
                query="all:*",
                max_results=500,
                categories=categories,
                date_from=datetime.now() - timedelta(days=days),
            )
        
        if len(papers) < min_topic_size * 2:
            logger.warning(f"Too few papers ({len(papers)}) for topic modeling")
            return [], {}
        
        # Use BERTopic for extraction
        extractor = BERTopicExtractor(min_topic_size=min_topic_size)
        topics, paper_topics = extractor.extract_topics(papers)
        
        # Calculate trend scores
        topics = extractor.calculate_trend_scores(papers, paper_topics, topics)
        
        return topics, paper_topics
    
    async def get_trending_topics(
        self,
        state: AgentState,
        top_n: int = 10,
    ) -> list[TopicInfo]:
        """Get trending topics for the state.
        
        Args:
            state: Agent state (will use state.papers if available)
            top_n: Number of top topics to return
        
        Returns:
            List of trending TopicInfo objects
        """
        papers = state.papers or []
        
        if not papers:
            # Fetch recent papers
            categories = state.options.get("categories") or self.settings.digest.categories
            days = state.options.get("days", 14)
            
            papers = await self.api_client.search_arxiv(
                query="all:*",
                max_results=300,
                categories=categories,
                date_from=datetime.now() - timedelta(days=days),
            )
        
        topics, _ = await self.extract_topics(
            papers=papers,
            min_topic_size=3,
        )
        
        return topics[:top_n]


# Global extractor for reuse
_bertopic_extractor: Optional[BERTopicExtractor] = None


def get_topic_extractor(min_topic_size: int = 5) -> BERTopicExtractor:
    """Get or create BERTopic extractor instance."""
    global _bertopic_extractor
    if _bertopic_extractor is None:
        _bertopic_extractor = BERTopicExtractor(min_topic_size=min_topic_size)
    return _bertopic_extractor

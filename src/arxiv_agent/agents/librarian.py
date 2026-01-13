"""Librarian agent for library management."""

from loguru import logger

from arxiv_agent.agents.state import AgentState
from arxiv_agent.config.settings import get_settings
from arxiv_agent.data.storage import get_db


class LibrarianAgent:
    """Agent for personal library management.
    
    Handles:
    - Adding/removing papers
    - Collection management
    - Tag management
    - Export/import
    """
    
    def __init__(self):
        """Initialize librarian agent."""
        self.settings = get_settings()
        self.db = get_db()
    
    async def run(self, state: AgentState) -> AgentState:
        """Execute librarian agent.
        
        Args:
            state: Current agent state
        
        Returns:
            Updated agent state
        """
        state.current_step = "library"
        logger.info(f"Librarian agent running")
        
        try:
            action = state.options.get("action", "list")
            
            if action == "add":
                await self._add_paper(state)
            elif action == "remove":
                await self._remove_paper(state)
            elif action == "list":
                await self._list_papers(state)
            elif action == "tag":
                await self._manage_tags(state)
            elif action == "collection":
                await self._manage_collection(state)
            else:
                state.add_error(f"Unknown library action: {action}")
                
        except Exception as e:
            logger.error(f"Librarian error: {e}")
            state.add_error(f"Library operation failed: {str(e)}")
        
        return state
    
    async def _add_paper(self, state: AgentState) -> None:
        """Add a paper to the library.
        
        Args:
            state: Current agent state
        """
        if not state.current_paper:
            state.add_error("No paper to add")
            return
        
        paper = state.current_paper
        
        # Save paper if not already in database
        self.db.save_paper(paper)
        
        # Add to collection if specified
        collection_name = state.options.get("collection")
        if collection_name:
            collection = self.db.get_collection_by_name(collection_name)
            if not collection:
                collection = self.db.create_collection(collection_name)
            self.db.add_paper_to_collection(paper.id, collection.id)
        
        # Add tags if specified
        tags = state.options.get("tags", [])
        for tag in tags:
            self.db.add_tag_to_paper(paper.id, tag)
        
        self.db.record_interaction(paper.id, "saved")
        
        logger.info(f"Added paper to library: {paper.id}")
    
    async def _remove_paper(self, state: AgentState) -> None:
        """Remove a paper from the library.
        
        Args:
            state: Current agent state
        """
        paper_id = state.paper_id
        if not paper_id:
            state.add_error("No paper ID specified for removal")
            return
        
        if not paper_id.startswith("arxiv:"):
            paper_id = f"arxiv:{paper_id}"
        
        if self.db.delete_paper(paper_id):
            logger.info(f"Removed paper from library: {paper_id}")
        else:
            state.add_error(f"Paper not found: {paper_id}")
    
    async def _list_papers(self, state: AgentState) -> None:
        """List papers in the library.
        
        Args:
            state: Current agent state
        """
        collection_id = state.options.get("collection_id")
        tag_id = state.options.get("tag_id")
        query = state.query
        limit = state.options.get("limit", 50)
        
        papers = self.db.search_papers(
            query=query,
            collection_id=collection_id,
            tag_id=tag_id,
            limit=limit,
        )
        
        state.papers = papers
        logger.info(f"Listed {len(papers)} papers from library")
    
    async def _manage_tags(self, state: AgentState) -> None:
        """Manage paper tags.
        
        Args:
            state: Current agent state
        """
        tag_action = state.options.get("tag_action", "list")
        paper_id = state.paper_id
        
        if tag_action == "add" and paper_id:
            tag_name = state.options.get("tag_name")
            if tag_name:
                if not paper_id.startswith("arxiv:"):
                    paper_id = f"arxiv:{paper_id}"
                self.db.add_tag_to_paper(paper_id, tag_name)
                logger.info(f"Added tag '{tag_name}' to {paper_id}")
                
        elif tag_action == "remove" and paper_id:
            tag_name = state.options.get("tag_name")
            if tag_name:
                if not paper_id.startswith("arxiv:"):
                    paper_id = f"arxiv:{paper_id}"
                self.db.remove_tag_from_paper(paper_id, tag_name)
                logger.info(f"Removed tag '{tag_name}' from {paper_id}")
                
        elif tag_action == "list":
            tags = self.db.list_tags()
            state.options["tags"] = [{"name": t.name, "auto": t.auto_generated} for t in tags]
    
    async def _manage_collection(self, state: AgentState) -> None:
        """Manage collections.
        
        Args:
            state: Current agent state
        """
        collection_action = state.options.get("collection_action", "list")
        
        if collection_action == "create":
            name = state.options.get("collection_name")
            description = state.options.get("description")
            if name:
                collection = self.db.create_collection(name, description)
                state.options["created_collection"] = {"id": collection.id, "name": collection.name}
                logger.info(f"Created collection: {name}")
                
        elif collection_action == "list":
            collections = self.db.list_collections()
            state.options["collections"] = [
                {"id": c.id, "name": c.name, "description": c.description}
                for c in collections
            ]

"""LangGraph-based multi-agent orchestrator."""

from datetime import datetime
from typing import Literal

from loguru import logger

from arxiv_agent.agents.state import AgentState
from arxiv_agent.config.settings import get_settings


class Orchestrator:
    """LangGraph-based multi-agent orchestrator.
    
    Coordinates workflow between specialized agents based on task type.
    """
    
    def __init__(self):
        """Initialize orchestrator with agents."""
        from arxiv_agent.agents.analyzer import AnalyzerAgent
        from arxiv_agent.agents.fetcher import FetcherAgent
        from arxiv_agent.agents.librarian import LibrarianAgent
        from arxiv_agent.agents.rag_chat import RAGChatAgent
        from arxiv_agent.agents.trend_analyst import TrendAnalystAgent
        
        self.settings = get_settings()
        
        # Initialize agents
        self.fetcher = FetcherAgent()
        self.analyzer = AnalyzerAgent()
        self.rag_chat = RAGChatAgent()
        self.librarian = LibrarianAgent()
        self.trend_analyst = TrendAnalystAgent()
        
        logger.debug("Orchestrator initialized")
    
    def _route(self, state: AgentState) -> AgentState:
        """Route task to appropriate agent.
        
        Args:
            state: Current agent state
        
        Returns:
            Updated state with routing info
        """
        state.status = "running"
        state.current_step = "routing"
        return state
    
    def _get_next_node(
        self, state: AgentState
    ) -> Literal["fetch", "analyze", "chat", "library", "trends", "end"]:
        """Determine next node based on task type.
        
        Args:
            state: Current agent state
        
        Returns:
            Name of next node to execute
        """
        task_mapping = {
            "search": "fetch",
            "analyze": "analyze",
            "chat": "chat",
            "digest": "fetch",
            "paper2code": "analyze",
            "trends": "trends",
            "library": "library",
        }
        return task_mapping.get(state.task_type, "end")
    
    def _finalize(self, state: AgentState) -> AgentState:
        """Finalize workflow execution.
        
        Args:
            state: Current agent state
        
        Returns:
            Finalized state
        """
        state.status = "completed" if not state.errors else "failed"
        state.completed_at = datetime.utcnow()
        state.current_step = "completed"
        return state
    
    async def run(
        self,
        task_type: str,
        paper_id: str | None = None,
        query: str | None = None,
        options: dict | None = None,
    ) -> AgentState:
        """Execute workflow.
        
        Args:
            task_type: Type of task (search, analyze, chat, etc.)
            paper_id: Optional paper ID for single-paper operations
            query: Optional search/chat query
            options: Additional options for agents
        
        Returns:
            Final agent state
        """
        # Initialize state
        state = AgentState(
            task=f"{task_type}:{paper_id or query or 'default'}",
            task_type=task_type,
            paper_id=paper_id,
            query=query,
            options=options or {},
        )
        
        logger.info(f"Starting workflow: {state.task}")
        
        # Route
        state = self._route(state)
        
        # Execute appropriate agent(s)
        try:
            if task_type == "search" or task_type == "fetch":
                state = await self.fetcher.run(state)
                
            elif task_type == "analyze":
                # First fetch if we don't have the paper
                if paper_id and not state.current_paper:
                    state = await self.fetcher.run(state)
                if not state.errors:
                    state = await self.analyzer.run(state)
                    
            elif task_type == "chat":
                # Fetch paper if needed
                if paper_id and not state.current_paper:
                    state = await self.fetcher.run(state)
                if not state.errors:
                    state = await self.rag_chat.run(state)
                    
            elif task_type == "digest":
                state = await self.fetcher.run(state)
                
            elif task_type == "paper2code":
                # Fetch paper first
                if paper_id and not state.current_paper:
                    state = await self.fetcher.run(state)
                if not state.errors:
                    state = await self.analyzer.run(state)
                    
            elif task_type == "trends":
                state = await self.trend_analyst.run(state)
                
            elif task_type == "library":
                # May need to fetch paper first for add operation
                if paper_id and state.options.get("action") == "add":
                    state = await self.fetcher.run(state)
                state = await self.librarian.run(state)
                
            else:
                state.add_error(f"Unknown task type: {task_type}")
                
        except Exception as e:
            logger.error(f"Workflow error: {e}")
            state.add_error(str(e))
        
        # Finalize
        state = self._finalize(state)
        
        logger.info(f"Workflow completed: {state.status} ({state.task})")
        
        return state
    
    async def run_with_langgraph(
        self,
        task_type: str,
        paper_id: str | None = None,
        query: str | None = None,
        thread_id: str | None = None,
    ) -> AgentState:
        """Execute workflow using LangGraph for complex multi-step tasks.
        
        This method uses LangGraph's StateGraph for proper checkpointing
        and complex conditional routing. Use this for production workloads.
        
        Args:
            task_type: Type of task
            paper_id: Optional paper ID
            query: Optional query
            thread_id: Optional thread ID for checkpointing
        
        Returns:
            Final agent state
        """
        import uuid
        
        try:
            from langgraph.graph import END, StateGraph
            
            # Build workflow dynamically
            workflow = StateGraph(AgentState)
            
            # Add nodes
            workflow.add_node("route", self._route)
            workflow.add_node("fetch", self.fetcher.run)
            workflow.add_node("analyze", self.analyzer.run)
            workflow.add_node("chat", self.rag_chat.run)
            workflow.add_node("library", self.librarian.run)
            workflow.add_node("trends", self.trend_analyst.run)
            workflow.add_node("finalize", self._finalize)
            
            # Set entry point
            workflow.set_entry_point("route")
            
            # Add conditional edges from router
            workflow.add_conditional_edges(
                "route",
                self._get_next_node,
                {
                    "fetch": "fetch",
                    "analyze": "analyze",
                    "chat": "chat",
                    "library": "library",
                    "trends": "trends",
                    "end": END,
                },
            )
            
            # Add edges to finalize
            for node in ["fetch", "analyze", "chat", "library", "trends"]:
                workflow.add_edge(node, "finalize")
            
            workflow.add_edge("finalize", END)
            
            # Compile
            app = workflow.compile()
            
            # Initial state
            initial_state = AgentState(
                task=f"{task_type}:{paper_id or query}",
                task_type=task_type,
                paper_id=paper_id,
                query=query,
            )
            
            config = {
                "configurable": {
                    "thread_id": thread_id or str(uuid.uuid4()),
                }
            }
            
            result = await app.ainvoke(initial_state, config)
            return result
            
        except ImportError:
            logger.warning("LangGraph not available, falling back to simple orchestration")
            return await self.run(task_type, paper_id, query)


# Global instance
_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    """Get or create orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


def reset_orchestrator() -> None:
    """Reset orchestrator instance."""
    global _orchestrator
    _orchestrator = None

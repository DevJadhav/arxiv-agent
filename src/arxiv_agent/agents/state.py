"""Agent state definition for multi-agent workflow."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from arxiv_agent.data.models import Paper


@dataclass
class AgentState:
    """Shared state for multi-agent workflow.
    
    This state is passed between agents during workflow execution
    and maintains all context needed for the current task.
    """
    
    # Input parameters
    task: str = ""
    task_type: Literal[
        "search", "analyze", "chat", "digest", 
        "paper2code", "trends", "library"
    ] = "search"
    paper_id: str | None = None
    query: str | None = None
    
    # Papers data
    papers: list[Paper] = field(default_factory=list)
    current_paper: Paper | None = None
    
    # Analysis results
    analysis: dict[str, Any] = field(default_factory=dict)
    
    # Chat context
    chat_history: list[dict[str, str]] = field(default_factory=list)
    retrieved_chunks: list[dict] = field(default_factory=list)
    chat_session_id: str | None = None
    
    # Code generation
    code_plan: dict[str, Any] | None = None
    generated_code: dict[str, str] = field(default_factory=dict)
    
    # Workflow status
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    current_step: str = ""
    errors: list[str] = field(default_factory=list)
    
    # Metadata
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    
    # Additional options
    options: dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: str) -> None:
        """Add an error message to the state."""
        self.errors.append(error)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "task": self.task,
            "task_type": self.task_type,
            "paper_id": self.paper_id,
            "query": self.query,
            "papers": [p.id for p in self.papers],
            "current_paper": self.current_paper.id if self.current_paper else None,
            "analysis": self.analysis,
            "chat_history": self.chat_history,
            "status": self.status,
            "current_step": self.current_step,
            "errors": self.errors,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

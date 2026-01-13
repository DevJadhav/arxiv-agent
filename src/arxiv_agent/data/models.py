"""SQLModel database models for ArXiv Agent."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Column, Index
from sqlmodel import JSON, Field, Relationship, SQLModel

if TYPE_CHECKING:
    pass


class Paper(SQLModel, table=True):
    """Research paper metadata."""
    
    __tablename__ = "papers"
    
    id: str = Field(primary_key=True)  # arxiv:2401.12345
    title: str
    authors: list[str] = Field(sa_column=Column(JSON), default_factory=list)
    abstract: str
    categories: list[str] = Field(sa_column=Column(JSON), default_factory=list)
    published_date: datetime | None = None
    pdf_path: str | None = None
    pdf_hash: str | None = None
    citation_count: int = 0
    tldr: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    analyses: list["Analysis"] = Relationship(back_populates="paper")
    chat_sessions: list["ChatSession"] = Relationship(back_populates="paper")
    
    __table_args__ = (
        Index("idx_papers_published", "published_date"),
    )


class Analysis(SQLModel, table=True):
    """Paper analysis results."""
    
    __tablename__ = "analyses"
    
    id: int | None = Field(default=None, primary_key=True)
    paper_id: str = Field(foreign_key="papers.id", index=True)
    analysis_type: str  # 'summary', 'methodology', 'full', 'code_plan'
    content: dict = Field(sa_column=Column(JSON), default_factory=dict)
    model_used: str | None = None
    token_count: int | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    paper: Paper | None = Relationship(back_populates="analyses")


class Collection(SQLModel, table=True):
    """User-defined paper collections."""
    
    __tablename__ = "collections"
    
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)
    description: str | None = None
    color: str = "#3B82F6"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PaperCollection(SQLModel, table=True):
    """Many-to-many relationship between papers and collections."""
    
    __tablename__ = "paper_collections"
    
    paper_id: str = Field(foreign_key="papers.id", primary_key=True)
    collection_id: int = Field(foreign_key="collections.id", primary_key=True)
    added_at: datetime = Field(default_factory=datetime.utcnow)


class ChatSession(SQLModel, table=True):
    """Chat session metadata."""
    
    __tablename__ = "chat_sessions"
    
    id: str = Field(primary_key=True)  # UUID
    paper_id: str = Field(foreign_key="papers.id", index=True)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)
    message_count: int = 0
    context_summary: str | None = None
    
    paper: Paper | None = Relationship(back_populates="chat_sessions")
    messages: list["ChatMessage"] = Relationship(back_populates="session")


class ChatMessage(SQLModel, table=True):
    """Individual chat messages."""
    
    __tablename__ = "chat_messages"
    
    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key="chat_sessions.id", index=True)
    role: str  # 'user', 'assistant'
    content: str
    retrieved_chunks: list[dict] | None = Field(sa_column=Column(JSON), default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    session: ChatSession | None = Relationship(back_populates="messages")


class Tag(SQLModel, table=True):
    """Paper tags."""
    
    __tablename__ = "tags"
    
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)
    auto_generated: bool = False


class PaperTag(SQLModel, table=True):
    """Many-to-many relationship between papers and tags."""
    
    __tablename__ = "paper_tags"
    
    paper_id: str = Field(foreign_key="papers.id", primary_key=True)
    tag_id: int = Field(foreign_key="tags.id", primary_key=True)
    confidence: float | None = None


class ReadingHistory(SQLModel, table=True):
    """User reading/interaction history for learning."""
    
    __tablename__ = "reading_history"
    
    id: int | None = Field(default=None, primary_key=True)
    paper_id: str = Field(foreign_key="papers.id", index=True)
    action: str  # 'viewed', 'analyzed', 'chatted', 'saved', 'code_generated'
    duration_sec: int | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UserPreference(SQLModel, table=True):
    """User preference key-value store."""
    
    __tablename__ = "user_preferences"
    
    key: str = Field(primary_key=True)
    value: dict = Field(sa_column=Column(JSON))
    updated_at: datetime = Field(default_factory=datetime.utcnow)

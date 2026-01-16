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
    paper_id: str | None = Field(foreign_key="papers.id", index=True, default=None)
    name: str | None = None  # Optional session name
    paper_ids: list[str] | None = Field(sa_column=Column(JSON), default=None)  # Multiple papers
    started_at: datetime | None = Field(default_factory=datetime.utcnow)
    created_at: datetime | None = Field(default=None)  # Alias for started_at
    last_active: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime | None = Field(default=None)  # Alias for last_active
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
    sources: str | None = None  # JSON string of sources (alias for retrieved_chunks)
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


# ==================== FTS5 Full-Text Search ====================
# Note: FTS5 virtual tables are created via raw SQL in DatabaseManager.init_fts()
# The following constants define the schema for documentation purposes.

FTS5_PAPERS_TABLE = "papers_fts"
FTS5_PAPERS_COLUMNS = ["id", "title", "abstract", "authors_text"]

# SQL to create FTS5 virtual table for papers (standalone table, not content-sync)
# We use a standalone table because the papers table stores authors as JSON,
# while we need it as text for full-text search.
FTS5_CREATE_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    id UNINDEXED,
    title,
    abstract,
    authors_text,
    tokenize='porter unicode61'
);
"""

# Triggers to keep FTS index synchronized with papers table
# For standalone FTS5 tables, we use regular INSERT/DELETE operations
FTS5_TRIGGER_INSERT = """
CREATE TRIGGER IF NOT EXISTS papers_fts_insert AFTER INSERT ON papers BEGIN
    INSERT INTO papers_fts(id, title, abstract, authors_text)
    VALUES (NEW.id, NEW.title, NEW.abstract, 
            (SELECT group_concat(value, ' ') FROM json_each(NEW.authors)));
END
"""

FTS5_TRIGGER_DELETE = """
CREATE TRIGGER IF NOT EXISTS papers_fts_delete AFTER DELETE ON papers BEGIN
    DELETE FROM papers_fts WHERE id = OLD.id;
END
"""

FTS5_TRIGGER_UPDATE = """
CREATE TRIGGER IF NOT EXISTS papers_fts_update AFTER UPDATE ON papers BEGIN
    DELETE FROM papers_fts WHERE id = OLD.id;
    INSERT INTO papers_fts(id, title, abstract, authors_text)
    VALUES (NEW.id, NEW.title, NEW.abstract,
            (SELECT group_concat(value, ' ') FROM json_each(NEW.authors)));
END
"""

# List of all trigger SQL statements (for backward compat)
FTS5_TRIGGERS_SQL = [FTS5_TRIGGER_INSERT, FTS5_TRIGGER_DELETE, FTS5_TRIGGER_UPDATE]

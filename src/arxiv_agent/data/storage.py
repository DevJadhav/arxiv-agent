"""Database storage manager for ArXiv Agent."""

import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator

from loguru import logger
from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine, select

from arxiv_agent.config.settings import get_settings
from arxiv_agent.data.models import (
    Analysis,
    ChatMessage,
    ChatSession,
    Collection,
    Paper,
    PaperCollection,
    PaperTag,
    ReadingHistory,
    Tag,
    UserPreference,
)


class DatabaseManager:
    """SQLite database manager with connection pooling and WAL mode."""
    
    def __init__(self, db_path: Path | None = None):
        """Initialize database manager.
        
        Args:
            db_path: Optional custom database path. Uses settings default if None.
        """
        settings = get_settings()
        self.db_path = db_path or settings.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
        
        # Enable WAL mode for better concurrency
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()
        
        # Create tables
        SQLModel.metadata.create_all(self.engine)
        logger.debug(f"Database initialized at {self.db_path}")
    
    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Get a database session with automatic commit/rollback."""
        with Session(self.engine, expire_on_commit=False) as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
    
    # ==================== Paper Operations ====================
    
    def get_paper(self, paper_id: str) -> Paper | None:
        """Get a paper by ID."""
        with self.session() as session:
            return session.get(Paper, paper_id)
    
    def save_paper(self, paper: Paper) -> Paper:
        """Save or update a paper."""
        with self.session() as session:
            # Check if exists
            existing = session.get(Paper, paper.id)
            if existing:
                # Update existing
                existing.title = paper.title
                existing.authors = paper.authors
                existing.abstract = paper.abstract
                existing.categories = paper.categories
                existing.published_date = paper.published_date
                existing.pdf_path = paper.pdf_path or existing.pdf_path
                existing.pdf_hash = paper.pdf_hash or existing.pdf_hash
                existing.citation_count = paper.citation_count
                existing.tldr = paper.tldr or existing.tldr
                existing.updated_at = datetime.utcnow()
                session.add(existing)
                session.commit()
                session.refresh(existing)
                return existing
            else:
                session.add(paper)
                session.commit()
                session.refresh(paper)
                return paper
    
    def delete_paper(self, paper_id: str) -> bool:
        """Delete a paper and all related data."""
        with self.session() as session:
            paper = session.get(Paper, paper_id)
            if not paper:
                return False
            session.delete(paper)
            return True
    
    def search_papers(
        self,
        query: str | None = None,
        collection_id: int | None = None,
        tag_id: int | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Paper]:
        """Search papers with optional filters."""
        with self.session() as session:
            statement = select(Paper)
            
            if query:
                # Simple text search in title and abstract
                query_lower = f"%{query.lower()}%"
                statement = statement.where(
                    (Paper.title.ilike(query_lower)) | (Paper.abstract.ilike(query_lower))
                )
            
            if collection_id:
                # Join with paper_collections
                statement = (
                    statement
                    .join(PaperCollection)
                    .where(PaperCollection.collection_id == collection_id)
                )
            
            if tag_id:
                # Join with paper_tags
                statement = (
                    statement
                    .join(PaperTag)
                    .where(PaperTag.tag_id == tag_id)
                )
            
            statement = statement.order_by(Paper.published_date.desc())
            statement = statement.offset(offset).limit(limit)
            
            return list(session.exec(statement).all())
    
    def list_papers(self, limit: int = 100, offset: int = 0) -> list[Paper]:
        """List all papers."""
        with self.session() as session:
            statement = (
                select(Paper)
                .order_by(Paper.created_at.desc())
                .offset(offset)
                .limit(limit)
            )
            return list(session.exec(statement).all())
    
    def count_papers(self) -> int:
        """Count total papers."""
        with self.session() as session:
            statement = select(Paper)
            return len(list(session.exec(statement).all()))
    
    # ==================== Analysis Operations ====================
    
    def save_analysis(self, analysis: Analysis) -> Analysis:
        """Save an analysis result."""
        with self.session() as session:
            session.add(analysis)
            session.commit()
            session.refresh(analysis)
            return analysis
    
    def get_latest_analysis(self, paper_id: str, analysis_type: str) -> Analysis | None:
        """Get the most recent analysis of a specific type for a paper."""
        with self.session() as session:
            statement = (
                select(Analysis)
                .where(Analysis.paper_id == paper_id)
                .where(Analysis.analysis_type == analysis_type)
                .order_by(Analysis.created_at.desc())
                .limit(1)
            )
            return session.exec(statement).first()
    
    def get_all_analyses(self, paper_id: str) -> list[Analysis]:
        """Get all analyses for a paper."""
        with self.session() as session:
            statement = (
                select(Analysis)
                .where(Analysis.paper_id == paper_id)
                .order_by(Analysis.created_at.desc())
            )
            return list(session.exec(statement).all())
    
    # ==================== Collection Operations ====================
    
    def create_collection(self, name: str, description: str | None = None, color: str = "#3B82F6") -> Collection:
        """Create a new collection."""
        with self.session() as session:
            collection = Collection(name=name, description=description, color=color)
            session.add(collection)
            session.commit()
            session.refresh(collection)
            return collection
    
    def get_collection(self, collection_id: int) -> Collection | None:
        """Get a collection by ID."""
        with self.session() as session:
            return session.get(Collection, collection_id)
    
    def get_collection_by_name(self, name: str) -> Collection | None:
        """Get a collection by name."""
        with self.session() as session:
            statement = select(Collection).where(Collection.name == name)
            return session.exec(statement).first()
    
    def list_collections(self) -> list[Collection]:
        """List all collections."""
        with self.session() as session:
            statement = select(Collection).order_by(Collection.name)
            return list(session.exec(statement).all())
    
    def add_paper_to_collection(self, paper_id: str, collection_id: int) -> bool:
        """Add a paper to a collection."""
        with self.session() as session:
            # Check if already exists
            statement = (
                select(PaperCollection)
                .where(PaperCollection.paper_id == paper_id)
                .where(PaperCollection.collection_id == collection_id)
            )
            if session.exec(statement).first():
                return False  # Already in collection
            
            pc = PaperCollection(paper_id=paper_id, collection_id=collection_id)
            session.add(pc)
            return True
    
    def remove_paper_from_collection(self, paper_id: str, collection_id: int) -> bool:
        """Remove a paper from a collection."""
        with self.session() as session:
            statement = (
                select(PaperCollection)
                .where(PaperCollection.paper_id == paper_id)
                .where(PaperCollection.collection_id == collection_id)
            )
            pc = session.exec(statement).first()
            if pc:
                session.delete(pc)
                return True
            return False
    
    # ==================== Chat Operations ====================
    
    def get_or_create_chat_session(self, paper_id: str) -> ChatSession:
        """Get existing active session or create new one."""
        settings = get_settings()
        cutoff = datetime.utcnow() - timedelta(hours=settings.chat.session_timeout_hours)
        
        with self.session() as session:
            # Find recent active session
            statement = (
                select(ChatSession)
                .where(ChatSession.paper_id == paper_id)
                .where(ChatSession.last_active > cutoff)
                .order_by(ChatSession.last_active.desc())
                .limit(1)
            )
            existing = session.exec(statement).first()
            
            if existing:
                existing.last_active = datetime.utcnow()
                session.add(existing)
                session.commit()
                session.refresh(existing)
                session.expunge(existing)
                return existing
            
            # Create new session
            new_session = ChatSession(
                id=str(uuid.uuid4()),
                paper_id=paper_id,
            )
            session.add(new_session)
            session.commit()
            session.refresh(new_session)
            session.expunge(new_session)
            return new_session
    
    def get_chat_session(self, session_id: str) -> ChatSession | None:
        """Get a chat session by ID."""
        with self.session() as session:
            return session.get(ChatSession, session_id)
    
    def add_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        retrieved_chunks: list[dict] | None = None,
    ) -> ChatMessage:
        """Add a message to a chat session."""
        with self.session() as session:
            message = ChatMessage(
                session_id=session_id,
                role=role,
                content=content,
                retrieved_chunks=retrieved_chunks,
            )
            session.add(message)
            
            # Update session
            chat_session = session.get(ChatSession, session_id)
            if chat_session:
                chat_session.message_count += 1
                chat_session.last_active = datetime.utcnow()
                session.add(chat_session)
            
            session.commit()
            session.refresh(message)
            return message
    
    def get_chat_history(self, session_id: str, limit: int = 100) -> list[ChatMessage]:
        """Get chat history for a session."""
        with self.session() as session:
            statement = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
                .limit(limit)
            )
            return list(session.exec(statement).all())
    
    def clear_chat_session(self, session_id: str) -> bool:
        """Clear all messages in a chat session."""
        with self.session() as session:
            statement = select(ChatMessage).where(ChatMessage.session_id == session_id)
            messages = session.exec(statement).all()
            for msg in messages:
                session.delete(msg)
            
            chat_session = session.get(ChatSession, session_id)
            if chat_session:
                chat_session.message_count = 0
                session.add(chat_session)
            return True
    
    # ==================== Tag Operations ====================
    
    def get_or_create_tag(self, name: str, auto_generated: bool = False) -> Tag:
        """Get existing tag or create new one."""
        with self.session() as session:
            statement = select(Tag).where(Tag.name == name)
            existing = session.exec(statement).first()
            if existing:
                session.expunge(existing)
                return existing
            
            tag = Tag(name=name, auto_generated=auto_generated)
            session.add(tag)
            session.commit()
            session.refresh(tag)
            session.expunge(tag)
            return tag
    
    def add_tag_to_paper(self, paper_id: str, tag_name: str, confidence: float | None = None) -> bool:
        """Add a tag to a paper."""
        tag = self.get_or_create_tag(tag_name)
        
        with self.session() as session:
            # Check if already exists
            statement = (
                select(PaperTag)
                .where(PaperTag.paper_id == paper_id)
                .where(PaperTag.tag_id == tag.id)
            )
            if session.exec(statement).first():
                return False
            
            pt = PaperTag(paper_id=paper_id, tag_id=tag.id, confidence=confidence)
            session.add(pt)
            return True
    
    def remove_tag_from_paper(self, paper_id: str, tag_name: str) -> bool:
        """Remove a tag from a paper."""
        with self.session() as session:
            statement = select(Tag).where(Tag.name == tag_name)
            tag = session.exec(statement).first()
            if not tag:
                return False
            
            statement = (
                select(PaperTag)
                .where(PaperTag.paper_id == paper_id)
                .where(PaperTag.tag_id == tag.id)
            )
            pt = session.exec(statement).first()
            if pt:
                session.delete(pt)
                return True
            return False
    
    def get_paper_tags(self, paper_id: str) -> list[Tag]:
        """Get all tags for a paper."""
        with self.session() as session:
            statement = (
                select(Tag)
                .join(PaperTag)
                .where(PaperTag.paper_id == paper_id)
            )
            return list(session.exec(statement).all())
    
    def list_tags(self) -> list[Tag]:
        """List all tags."""
        with self.session() as session:
            statement = select(Tag).order_by(Tag.name)
            return list(session.exec(statement).all())
    
    # ==================== Reading History ====================
    
    def record_interaction(
        self,
        paper_id: str,
        action: str,
        duration_sec: int | None = None,
    ) -> None:
        """Record a user interaction for learning."""
        with self.session() as session:
            history = ReadingHistory(
                paper_id=paper_id,
                action=action,
                duration_sec=duration_sec,
            )
            session.add(history)
    
    def get_reading_history(self, paper_id: str | None = None, limit: int = 100) -> list[ReadingHistory]:
        """Get reading history, optionally filtered by paper."""
        with self.session() as session:
            statement = select(ReadingHistory)
            if paper_id:
                statement = statement.where(ReadingHistory.paper_id == paper_id)
            statement = statement.order_by(ReadingHistory.created_at.desc()).limit(limit)
            return list(session.exec(statement).all())
    
    # ==================== User Preferences ====================
    
    def get_preference(self, key: str) -> dict | None:
        """Get a user preference value."""
        with self.session() as session:
            pref = session.get(UserPreference, key)
            return pref.value if pref else None
    
    def set_preference(self, key: str, value: dict) -> None:
        """Set a user preference value."""
        with self.session() as session:
            existing = session.get(UserPreference, key)
            if existing:
                existing.value = value
                existing.updated_at = datetime.utcnow()
                session.add(existing)
            else:
                pref = UserPreference(key=key, value=value)
                session.add(pref)


# Global instance
_db: DatabaseManager | None = None


def get_db() -> DatabaseManager:
    """Get or create database manager instance."""
    global _db
    if _db is None:
        _db = DatabaseManager()
    return _db


def reset_db() -> None:
    """Reset database manager instance (useful for testing)."""
    global _db
    _db = None

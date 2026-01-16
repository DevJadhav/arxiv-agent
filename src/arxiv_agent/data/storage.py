"""Database storage manager for ArXiv Agent."""

import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator

from loguru import logger
from sqlalchemy import event, text
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
    FTS5_CREATE_SQL,
    FTS5_TRIGGERS_SQL,
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
        
        # Run migrations for new columns
        self._run_migrations()
        
        # Track FTS5 initialization state
        self._fts_initialized = False
    
    def _run_migrations(self) -> None:
        """Run database migrations for schema changes."""
        with self.engine.connect() as conn:
            # Get existing columns in chat_sessions
            result = conn.execute(text("PRAGMA table_info(chat_sessions)"))
            existing_columns = {row[1] for row in result.fetchall()}
            
            # Add missing columns to chat_sessions
            migrations = [
                ("name", "ALTER TABLE chat_sessions ADD COLUMN name TEXT"),
                ("paper_ids", "ALTER TABLE chat_sessions ADD COLUMN paper_ids TEXT"),
                ("created_at", "ALTER TABLE chat_sessions ADD COLUMN created_at TIMESTAMP"),
                ("updated_at", "ALTER TABLE chat_sessions ADD COLUMN updated_at TIMESTAMP"),
            ]
            
            for col_name, sql in migrations:
                if col_name not in existing_columns:
                    try:
                        conn.execute(text(sql))
                        logger.debug(f"Added column {col_name} to chat_sessions")
                    except Exception as e:
                        logger.debug(f"Migration for {col_name} skipped: {e}")
            
            # Get existing columns in chat_messages
            result = conn.execute(text("PRAGMA table_info(chat_messages)"))
            existing_columns = {row[1] for row in result.fetchall()}
            
            # Add missing columns to chat_messages
            if "sources" not in existing_columns:
                try:
                    conn.execute(text("ALTER TABLE chat_messages ADD COLUMN sources TEXT"))
                    logger.debug("Added column sources to chat_messages")
                except Exception as e:
                    logger.debug(f"Migration for sources skipped: {e}")
            
            conn.commit()
    
    def init_fts(self) -> None:
        """Initialize FTS5 full-text search tables and triggers.
        
        This creates the FTS5 virtual table and triggers to keep it
        synchronized with the papers table. Should be called after
        the database is created or when FTS search is first needed.
        """
        if self._fts_initialized:
            return
        
        with self.engine.connect() as conn:
            # Create FTS5 virtual table
            conn.execute(text(FTS5_CREATE_SQL))
            
            # Create synchronization triggers (now a list)
            for trigger_sql in FTS5_TRIGGERS_SQL:
                trigger_sql = trigger_sql.strip()
                if trigger_sql:
                    conn.execute(text(trigger_sql))
            
            conn.commit()
        
        # Index existing papers
        self._rebuild_fts_index()
        self._fts_initialized = True
        logger.debug("FTS5 full-text search initialized")
    
    def _rebuild_fts_index(self) -> None:
        """Rebuild FTS index from existing papers.
        
        This populates the FTS table with all existing papers.
        Useful after initial FTS setup or for index corruption recovery.
        """
        with self.engine.connect() as conn:
            # Clear existing FTS data
            conn.execute(text("DELETE FROM papers_fts"))
            
            # Reindex all papers (standalone FTS table)
            conn.execute(text("""
                INSERT INTO papers_fts(id, title, abstract, authors_text)
                SELECT id, title, abstract,
                       (SELECT group_concat(value, ' ') FROM json_each(authors))
                FROM papers
            """))
            conn.commit()
        
        logger.debug("FTS5 index rebuilt")
    
    def fts_search(
        self,
        query: str,
        limit: int = 50,
        snippet_size: int = 64,
    ) -> list[dict]:
        """Perform full-text search using FTS5.
        
        Args:
            query: Search query (supports FTS5 query syntax)
            limit: Maximum results to return
            snippet_size: Size of snippet excerpts in tokens
        
        Returns:
            List of dicts with paper_id, rank, and snippet
            
        Example queries:
            - "transformer attention" - Match both words
            - "transformer OR attention" - Match either word
            - "transformer*" - Prefix match
            - '"attention mechanism"' - Exact phrase
            - "NEAR(transformer attention, 5)" - Words within 5 tokens
        """
        # Ensure FTS is initialized
        if not self._fts_initialized:
            self.init_fts()
        
        # Escape special characters but allow FTS5 operators
        safe_query = query.replace("'", "''")
        
        with self.engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT 
                    id,
                    rank,
                    snippet(papers_fts, 1, '<mark>', '</mark>', '...', {snippet_size}) as title_snippet,
                    snippet(papers_fts, 2, '<mark>', '</mark>', '...', {snippet_size}) as abstract_snippet
                FROM papers_fts
                WHERE papers_fts MATCH :query
                ORDER BY rank
                LIMIT :limit
            """), {"query": safe_query, "limit": limit})
            
            rows = result.fetchall()
            
        return [
            {
                "paper_id": row[0],
                "rank": row[1],
                "title_snippet": row[2],
                "abstract_snippet": row[3],
            }
            for row in rows
        ]
    
    def fts_search_papers(
        self,
        query: str,
        limit: int = 50,
    ) -> list[Paper]:
        """Perform full-text search and return Paper objects.
        
        Args:
            query: Search query (supports FTS5 query syntax)
            limit: Maximum results to return
        
        Returns:
            List of Paper objects ranked by relevance
        """
        fts_results = self.fts_search(query, limit=limit)
        
        paper_ids = [r["paper_id"] for r in fts_results]
        
        with self.session() as session:
            statement = select(Paper).where(Paper.id.in_(paper_ids))
            papers = {p.id: p for p in session.exec(statement).all()}
        
        # Return in rank order
        return [papers[pid] for pid in paper_ids if pid in papers]
    
    def fts_autocomplete(
        self,
        prefix: str,
        limit: int = 10,
    ) -> list[str]:
        """Get autocomplete suggestions from FTS index.
        
        Args:
            prefix: Search prefix
            limit: Maximum suggestions
        
        Returns:
            List of matching terms from the corpus
        """
        if not self._fts_initialized:
            self.init_fts()
        
        safe_prefix = prefix.replace("'", "''")
        
        with self.engine.connect() as conn:
            # Use FTS5 vocab table for term suggestions
            result = conn.execute(text(f"""
                SELECT DISTINCT term 
                FROM papers_fts_vocab
                WHERE term LIKE :prefix || '%'
                ORDER BY doc DESC
                LIMIT :limit
            """), {"prefix": safe_prefix.lower(), "limit": limit})
            
            return [row[0] for row in result.fetchall()]
    
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
    
    def list_chat_sessions(self, limit: int = 50) -> list[ChatSession]:
        """List all chat sessions, most recent first."""
        with self.session() as session:
            statement = (
                select(ChatSession)
                .order_by(ChatSession.last_active.desc())
                .limit(limit)
            )
            return list(session.exec(statement).all())
    
    def get_chat_messages(self, session_id: str) -> list[ChatMessage]:
        """Get all messages for a chat session (alias for get_chat_history)."""
        return self.get_chat_history(session_id, limit=10000)
    
    def export_chat_session(self, session_id: str, format: str = "markdown") -> str | dict:
        """Export a chat session to specified format.
        
        Args:
            session_id: ID of the session to export
            format: Output format - "markdown" or "json"
            
        Returns:
            Formatted string (markdown) or dict (json)
        """
        import json as json_lib
        
        chat_session = self.get_chat_session(session_id)
        if not chat_session:
            raise ValueError(f"Chat session not found: {session_id}")
        
        # Use get_chat_messages which is mockable in tests
        messages = self.get_chat_messages(session_id)
        
        # Get paper info if available
        paper = None
        paper_id = getattr(chat_session, 'paper_id', None)
        if paper_id:
            paper = self.get_paper(paper_id)
        
        # Handle name/paper_ids from test fixtures
        session_name = getattr(chat_session, 'name', None) or paper_id or str(session_id)
        
        if format == "json":
            created_at = getattr(chat_session, 'created_at', None)
            last_active = getattr(chat_session, 'last_active', None) or getattr(chat_session, 'updated_at', None)
            return {
                "session": {
                    "id": chat_session.id,
                    "paper_id": getattr(chat_session, 'paper_id', None),
                    "name": getattr(chat_session, 'name', None),
                    "created_at": created_at.isoformat() if created_at else None,
                    "last_active": last_active.isoformat() if last_active else None,
                },
                "paper": {
                    "id": paper.id,
                    "title": paper.title,
                } if paper else None,
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "created_at": msg.created_at.isoformat() if msg.created_at else None,
                        "sources": getattr(msg, 'sources', None) or getattr(msg, 'retrieved_chunks', None),
                    }
                    for msg in messages
                ]
            }
        
        # Markdown format
        title = paper.title if paper else session_name
        lines = [
            f"# Chat Export: {title}",
            "",
            f"**Session ID:** {session_id}",
        ]
        
        if paper_id:
            lines.append(f"**Paper ID:** {paper_id}")
        
        created_at = getattr(chat_session, 'created_at', None)
        if created_at:
            lines.append(f"**Created:** {created_at.strftime('%Y-%m-%d %H:%M')}")
        
        last_active = getattr(chat_session, 'last_active', None) or getattr(chat_session, 'updated_at', None)
        if last_active:
            lines.append(f"**Last Active:** {last_active.strftime('%Y-%m-%d %H:%M')}")
        
        lines.extend(["", "---", ""])
        
        for msg in messages:
            if msg.role == "user":
                lines.append("**👤 You:**")
            else:
                lines.append("**🤖 Assistant:**")
            
            lines.extend([
                "",
                msg.content,
                "",
            ])
            
            # Include sources if available (handle both sources and retrieved_chunks)
            sources_data = getattr(msg, 'sources', None) or getattr(msg, 'retrieved_chunks', None)
            if sources_data:
                import json as json_lib
                # Parse if string
                if isinstance(sources_data, str):
                    try:
                        sources_data = json_lib.loads(sources_data)
                    except:
                        sources_data = []
                
                if isinstance(sources_data, list) and sources_data:
                    lines.append("*Sources:*")
                    for src in sources_data[:5]:
                        if isinstance(src, dict):
                            src_paper_id = src.get('paper_id', 'Unknown')
                            section = src.get('section', '')
                            lines.append(f"- {src_paper_id} {f'({section})' if section else ''}")
                    lines.append("")
            
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)
    
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

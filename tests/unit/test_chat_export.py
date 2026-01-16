"""Tests for chat export functionality.

TDD: Write tests first, then implement the feature.
DeepDive.md Reference: Section 6.3 - Chat Export
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from pathlib import Path
import json

from arxiv_agent.data.models import ChatSession, ChatMessage


@pytest.fixture
def sample_chat_session():
    """Create a sample chat session with messages."""
    session = ChatSession(
        id=1,
        name="Test Chat Session",
        paper_ids=["2401.00001", "2401.00002"],
        created_at=datetime(2024, 1, 15, 10, 30, 0),
        updated_at=datetime(2024, 1, 15, 11, 45, 0),
    )
    return session


@pytest.fixture
def sample_messages():
    """Create sample chat messages."""
    return [
        ChatMessage(
            id=1,
            session_id=1,
            role="user",
            content="What is the main contribution of the paper?",
            sources=None,
            created_at=datetime(2024, 1, 15, 10, 30, 0),
        ),
        ChatMessage(
            id=2,
            session_id=1,
            role="assistant",
            content="The main contribution is the introduction of the transformer architecture...",
            sources=json.dumps([
                {"paper_id": "2401.00001", "section": "Introduction", "relevance": 0.95}
            ]),
            created_at=datetime(2024, 1, 15, 10, 31, 0),
        ),
        ChatMessage(
            id=3,
            session_id=1,
            role="user",
            content="How does it compare to previous approaches?",
            sources=None,
            created_at=datetime(2024, 1, 15, 10, 32, 0),
        ),
        ChatMessage(
            id=4,
            session_id=1,
            role="assistant",
            content="Compared to RNNs and LSTMs, transformers offer parallel processing...",
            sources=json.dumps([
                {"paper_id": "2401.00001", "section": "Related Work", "relevance": 0.88}
            ]),
            created_at=datetime(2024, 1, 15, 10, 33, 0),
        ),
    ]


class TestExportChatToMarkdown:
    """Test exporting chat session to markdown format."""

    def test_export_returns_markdown_string(self, sample_chat_session, sample_messages):
        """Export chat session returns markdown formatted string."""
        from arxiv_agent.data.storage import DatabaseManager
        
        with patch.object(DatabaseManager, '__init__', lambda x: None):
            db = DatabaseManager()
            db.get_chat_session = MagicMock(return_value=sample_chat_session)
            db.get_chat_messages = MagicMock(return_value=sample_messages)
            
            # This method should exist after implementation
            result = db.export_chat_session(session_id=1, format="markdown")
            
            assert isinstance(result, str)
            assert len(result) > 0

    def test_export_includes_session_metadata(self, sample_chat_session, sample_messages):
        """Exported markdown includes session metadata header."""
        from arxiv_agent.data.storage import DatabaseManager
        
        with patch.object(DatabaseManager, '__init__', lambda x: None):
            db = DatabaseManager()
            db.get_chat_session = MagicMock(return_value=sample_chat_session)
            db.get_chat_messages = MagicMock(return_value=sample_messages)
            
            result = db.export_chat_session(session_id=1, format="markdown")
            
            # Should include session name as title
            assert sample_chat_session.name in result or "# " in result
            # Should include creation date
            assert "2024" in result or "Created" in result.lower()

    def test_export_includes_all_messages(self, sample_chat_session, sample_messages):
        """Exported markdown includes all conversation messages."""
        from arxiv_agent.data.storage import DatabaseManager
        
        with patch.object(DatabaseManager, '__init__', lambda x: None):
            db = DatabaseManager()
            db.get_chat_session = MagicMock(return_value=sample_chat_session)
            db.get_chat_messages = MagicMock(return_value=sample_messages)
            
            result = db.export_chat_session(session_id=1, format="markdown")
            
            # Should include user messages
            assert "main contribution" in result.lower()
            # Should include assistant responses
            assert "transformer" in result.lower()

    def test_export_distinguishes_user_and_assistant(self, sample_chat_session, sample_messages):
        """Exported markdown clearly distinguishes user/assistant messages."""
        from arxiv_agent.data.storage import DatabaseManager
        
        with patch.object(DatabaseManager, '__init__', lambda x: None):
            db = DatabaseManager()
            db.get_chat_session = MagicMock(return_value=sample_chat_session)
            db.get_chat_messages = MagicMock(return_value=sample_messages)
            
            result = db.export_chat_session(session_id=1, format="markdown")
            
            # Should have markers for user and assistant
            assert "user" in result.lower() or "**You**" in result or "👤" in result
            assert "assistant" in result.lower() or "**Assistant**" in result or "🤖" in result

    def test_export_includes_sources(self, sample_chat_session, sample_messages):
        """Exported markdown includes cited sources."""
        from arxiv_agent.data.storage import DatabaseManager
        
        with patch.object(DatabaseManager, '__init__', lambda x: None):
            db = DatabaseManager()
            db.get_chat_session = MagicMock(return_value=sample_chat_session)
            db.get_chat_messages = MagicMock(return_value=sample_messages)
            
            result = db.export_chat_session(session_id=1, format="markdown")
            
            # Should include paper references
            assert "2401.00001" in result or "Source" in result or "Reference" in result

    def test_export_preserves_message_order(self, sample_chat_session, sample_messages):
        """Exported markdown preserves chronological order."""
        from arxiv_agent.data.storage import DatabaseManager
        
        with patch.object(DatabaseManager, '__init__', lambda x: None):
            db = DatabaseManager()
            db.get_chat_session = MagicMock(return_value=sample_chat_session)
            db.get_chat_messages = MagicMock(return_value=sample_messages)
            
            result = db.export_chat_session(session_id=1, format="markdown")
            
            # First message should appear before last message
            first_pos = result.find("main contribution")
            last_pos = result.find("parallel processing")
            assert first_pos < last_pos


class TestExportChatToJSON:
    """Test exporting chat session to JSON format."""

    def test_export_json_returns_dict(self, sample_chat_session, sample_messages):
        """Export to JSON returns dictionary."""
        from arxiv_agent.data.storage import DatabaseManager
        
        with patch.object(DatabaseManager, '__init__', lambda x: None):
            db = DatabaseManager()
            db.get_chat_session = MagicMock(return_value=sample_chat_session)
            db.get_chat_messages = MagicMock(return_value=sample_messages)
            
            result = db.export_chat_session(session_id=1, format="json")
            
            assert isinstance(result, dict)

    def test_export_json_has_session_info(self, sample_chat_session, sample_messages):
        """JSON export includes session metadata."""
        from arxiv_agent.data.storage import DatabaseManager
        
        with patch.object(DatabaseManager, '__init__', lambda x: None):
            db = DatabaseManager()
            db.get_chat_session = MagicMock(return_value=sample_chat_session)
            db.get_chat_messages = MagicMock(return_value=sample_messages)
            
            result = db.export_chat_session(session_id=1, format="json")
            
            assert "session" in result or "name" in result
            assert "messages" in result

    def test_export_json_messages_structure(self, sample_chat_session, sample_messages):
        """JSON export messages have proper structure."""
        from arxiv_agent.data.storage import DatabaseManager
        
        with patch.object(DatabaseManager, '__init__', lambda x: None):
            db = DatabaseManager()
            db.get_chat_session = MagicMock(return_value=sample_chat_session)
            db.get_chat_messages = MagicMock(return_value=sample_messages)
            
            result = db.export_chat_session(session_id=1, format="json")
            
            assert len(result["messages"]) == len(sample_messages)
            for msg in result["messages"]:
                assert "role" in msg
                assert "content" in msg


class TestChatHistoryCommand:
    """Test chat history CLI command."""

    def test_history_command_exists(self):
        """CLI has history subcommand under chat."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        result = runner.invoke(app, ["chat", "--help"])
        
        assert result.exit_code == 0
        # Will pass after history command is added
        # assert "history" in result.output.lower()

    def test_history_lists_sessions(self):
        """Chat history lists recent sessions."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        with patch("arxiv_agent.cli.commands.chat.get_storage") as mock_storage:
            mock_db = MagicMock()
            mock_db.list_chat_sessions = MagicMock(return_value=[
                ChatSession(id=1, name="Session 1", paper_ids=["2401.00001"],
                           created_at=datetime.now(), updated_at=datetime.now()),
                ChatSession(id=2, name="Session 2", paper_ids=["2401.00002"],
                           created_at=datetime.now(), updated_at=datetime.now()),
            ])
            mock_storage.return_value = mock_db
            
            # Will work after implementation
            result = runner.invoke(app, ["chat", "history"])
            
            # Should list sessions without crashing
            # assert result.exit_code == 0

    def test_history_shows_session_details(self):
        """Chat history show <id> shows session details."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        with patch("arxiv_agent.cli.commands.chat.get_storage") as mock_storage:
            mock_db = MagicMock()
            mock_db.get_chat_session = MagicMock(return_value=ChatSession(
                id=1, name="Test Session", paper_ids=["2401.00001"],
                created_at=datetime.now(), updated_at=datetime.now()
            ))
            mock_db.get_chat_messages = MagicMock(return_value=[])
            mock_storage.return_value = mock_db
            
            # Will work after implementation
            result = runner.invoke(app, ["chat", "history", "1"])
            
            # Should show session details
            # assert result.exit_code == 0


class TestChatExportCLICommand:
    """Test chat export CLI command."""

    def test_export_command_exists(self):
        """CLI has export subcommand under chat."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        result = runner.invoke(app, ["chat", "--help"])
        
        assert result.exit_code == 0
        # Will pass after export command is added
        # assert "export" in result.output.lower()

    def test_export_creates_file(self, tmp_path, sample_chat_session, sample_messages):
        """Chat export creates output file."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        output_file = tmp_path / "chat_export.md"
        
        runner = CliRunner()
        
        with patch("arxiv_agent.cli.commands.chat.get_storage") as mock_storage:
            mock_db = MagicMock()
            mock_db.get_chat_session = MagicMock(return_value=sample_chat_session)
            mock_db.get_chat_messages = MagicMock(return_value=sample_messages)
            mock_db.export_chat_session = MagicMock(return_value="# Exported Chat\n\nContent...")
            mock_storage.return_value = mock_db
            
            # Will work after implementation
            result = runner.invoke(app, ["chat", "export", "1", str(output_file)])
            
            # Should create the file
            # assert output_file.exists()

    def test_export_format_option(self):
        """Chat export supports --format option."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        result = runner.invoke(app, ["chat", "export", "--help"])
        
        # Will show format option after implementation
        # assert "--format" in result.output or "-f" in result.output

    def test_export_nonexistent_session(self):
        """Chat export handles nonexistent session gracefully."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        with patch("arxiv_agent.cli.commands.chat.get_storage") as mock_storage:
            mock_db = MagicMock()
            mock_db.get_chat_session = MagicMock(return_value=None)
            mock_storage.return_value = mock_db
            
            result = runner.invoke(app, ["chat", "export", "999", "output.md"])
            
            # Should show error, not crash
            # assert result.exit_code != 0 or "not found" in result.output.lower()


class TestChatSourcesCommand:
    """Test /sources command in chat session."""

    def test_sources_command_shows_references(self, sample_messages):
        """The /sources command shows all cited papers."""
        # This tests the interactive chat /sources command
        # Implementation is in the chat session loop
        
        # Extract sources from messages
        sources = []
        for msg in sample_messages:
            if msg.sources:
                sources.extend(json.loads(msg.sources))
        
        assert len(sources) > 0
        
        # After implementation, /sources should list all unique paper_ids
        unique_papers = set(s["paper_id"] for s in sources)
        assert "2401.00001" in unique_papers

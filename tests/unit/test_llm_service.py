"""Unit tests for LLM service with streaming support."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

from arxiv_agent.core.llm_service import LLMService, LLMResponse, TokenTracker


class TestTokenTracker:
    """Tests for token usage tracking."""
    
    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a token tracker with temp storage."""
        return TokenTracker(tmp_path / "usage.json")
    
    def test_track_usage(self, tracker):
        """Test tracking token usage."""
        tracker.track("anthropic", "claude-3-5-sonnet", 100, 50)
        
        # Check daily tracking
        import datetime
        today = datetime.date.today().isoformat()
        assert tracker.data["daily"].get(today, 0) == 150
    
    def test_check_daily_limit_under(self, tracker):
        """Test daily limit check when under limit."""
        tracker.track("anthropic", "claude-3-5-sonnet", 100, 50)
        
        # Should not exceed limit of 1000
        assert tracker.check_daily_limit(1000) is False
    
    def test_check_daily_limit_exceeded(self, tracker):
        """Test daily limit check when exceeded."""
        tracker.track("anthropic", "claude-3-5-sonnet", 500, 600)
        
        # Should exceed limit of 1000
        assert tracker.check_daily_limit(1000) is True
    
    def test_persistence(self, tmp_path):
        """Test tracker persists data across instances."""
        path = tmp_path / "usage.json"
        
        tracker1 = TokenTracker(path)
        tracker1.track("anthropic", "claude-3-5-sonnet", 100, 50)
        
        # Create new instance
        tracker2 = TokenTracker(path)
        
        import datetime
        today = datetime.date.today().isoformat()
        assert tracker2.data["daily"].get(today, 0) == 150


class TestLLMService:
    """Tests for LLM service."""
    
    @pytest.fixture
    def mock_settings(self, tmp_path):
        """Create mock settings."""
        mock = MagicMock()
        mock.data_dir = tmp_path
        mock.llm.default_provider = "anthropic"
        mock.llm.get_provider_model.return_value = "claude-3-5-sonnet-20241022"
        mock.llm.get_agent_config.return_value = ("anthropic", "claude-3-5-sonnet-20241022")
        mock.llm.max_tokens = 4096
        mock.llm.temperature = 0.7
        mock.guardrails.warn_on_costly_request = False
        mock.guardrails.max_daily_tokens = 100000
        mock.guardrails.enable_json_repair = False
        return mock
    
    @pytest.fixture
    def mock_key_storage(self):
        """Create mock key storage."""
        mock = MagicMock()
        mock.get_key.return_value = "test-api-key"
        return mock
    
    @pytest.fixture
    def service(self, mock_settings, mock_key_storage):
        """Create LLM service with mocks."""
        with patch("arxiv_agent.core.llm_service.get_settings", return_value=mock_settings), \
             patch("arxiv_agent.core.llm_service.get_key_storage", return_value=mock_key_storage):
            return LLMService()
    
    def test_initialization(self, service):
        """Test service initializes correctly."""
        assert service.provider == "anthropic"
        assert service.model == "claude-3-5-sonnet-20241022"
    
    def test_initialization_with_agent(self, mock_settings, mock_key_storage):
        """Test initialization with agent-specific config."""
        mock_settings.llm.get_agent_config.return_value = ("openai", "gpt-4o")
        
        with patch("arxiv_agent.core.llm_service.get_settings", return_value=mock_settings), \
             patch("arxiv_agent.core.llm_service.get_key_storage", return_value=mock_key_storage):
            service = LLMService(agent="analyzer")
        
        assert service.provider == "openai"
        assert service.model == "gpt-4o"
    
    def test_check_context_window(self, service):
        """Test context window validation."""
        # Short prompt should not warn
        service.check_context_window("Short prompt")
        
        # Very long prompt should trigger warning (but not raise)
        long_prompt = "x" * 500000  # ~125k tokens estimated
        with patch("arxiv_agent.core.llm_service.logger") as mock_logger:
            service.check_context_window(long_prompt)
            # Warning should be logged for very long prompts
    
    def test_history_management(self, service):
        """Test conversation history management."""
        assert service.history == []
        
        service.history.append({"role": "user", "content": "Hello"})
        service.history.append({"role": "assistant", "content": "Hi there!"})
        
        assert len(service.history) == 2
        
        service.clear_history()
        assert service.history == []
    
    def test_set_history(self, service):
        """Test setting history from external source."""
        history = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
        ]
        
        service.set_history(history)
        
        assert len(service.history) == 2
        assert service.history[0]["content"] == "First question"


class TestLLMServiceGeneration:
    """Tests for LLM generation methods."""
    
    @pytest.fixture
    def mock_anthropic_client(self):
        """Create mock Anthropic client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Generated response")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        return mock_client
    
    @pytest.fixture
    def service_with_mock_client(self, mock_settings, mock_key_storage, mock_anthropic_client):
        """Create service with mocked client."""
        mock_settings.data_dir = Path("/tmp/test")
        mock_settings.data_dir.mkdir(parents=True, exist_ok=True)
        
        with patch("arxiv_agent.core.llm_service.get_settings", return_value=mock_settings), \
             patch("arxiv_agent.core.llm_service.get_key_storage", return_value=mock_key_storage):
            service = LLMService()
            service._anthropic_client = mock_anthropic_client
            return service
    
    @pytest.fixture
    def mock_settings(self, tmp_path):
        """Create mock settings."""
        mock = MagicMock()
        mock.data_dir = tmp_path
        mock.llm.default_provider = "anthropic"
        mock.llm.get_provider_model.return_value = "claude-3-5-sonnet-20241022"
        mock.llm.get_agent_config.return_value = ("anthropic", "claude-3-5-sonnet-20241022")
        mock.llm.max_tokens = 4096
        mock.llm.temperature = 0.7
        mock.guardrails.warn_on_costly_request = False
        mock.guardrails.max_daily_tokens = 100000
        mock.guardrails.enable_json_repair = False
        return mock
    
    @pytest.fixture
    def mock_key_storage(self):
        """Create mock key storage."""
        mock = MagicMock()
        mock.get_key.return_value = "test-api-key"
        return mock
    
    def test_generate_anthropic(self, service_with_mock_client):
        """Test synchronous generation with Anthropic."""
        response = service_with_mock_client.generate(
            prompt="Test prompt",
            system_prompt="You are a helpful assistant.",
        )
        
        assert isinstance(response, LLMResponse)
        assert response.content == "Generated response"
        assert response.provider == "anthropic"
        assert response.input_tokens == 100
        assert response.output_tokens == 50


class TestLLMServiceStreaming:
    """Tests for LLM streaming functionality."""
    
    @pytest.fixture
    def mock_settings(self, tmp_path):
        """Create mock settings."""
        mock = MagicMock()
        mock.data_dir = tmp_path
        mock.llm.default_provider = "anthropic"
        mock.llm.get_provider_model.return_value = "claude-3-5-sonnet-20241022"
        mock.llm.get_agent_config.return_value = ("anthropic", "claude-3-5-sonnet-20241022")
        mock.llm.max_tokens = 4096
        mock.llm.temperature = 0.7
        mock.guardrails.warn_on_costly_request = False
        return mock
    
    @pytest.fixture
    def mock_key_storage(self):
        """Create mock key storage."""
        mock = MagicMock()
        mock.get_key.return_value = "test-api-key"
        return mock
    
    @pytest.mark.asyncio
    async def test_stream_anthropic(self, mock_settings, mock_key_storage):
        """Test streaming with Anthropic."""
        with patch("arxiv_agent.core.llm_service.get_settings", return_value=mock_settings), \
             patch("arxiv_agent.core.llm_service.get_key_storage", return_value=mock_key_storage):
            service = LLMService()
        
        # Mock the streaming method
        async def mock_stream(*args, **kwargs):
            chunks = ["Hello", " ", "world", "!"]
            for chunk in chunks:
                yield chunk
        
        with patch.object(service, "_stream_anthropic", mock_stream):
            collected = []
            async for chunk in service.stream("Test prompt"):
                collected.append(chunk)
            
            assert collected == ["Hello", " ", "world", "!"]
            assert "".join(collected) == "Hello world!"
    
    @pytest.mark.asyncio
    async def test_stream_openai(self, mock_settings, mock_key_storage):
        """Test streaming with OpenAI."""
        mock_settings.llm.default_provider = "openai"
        mock_settings.llm.get_provider_model.return_value = "gpt-4o"
        
        with patch("arxiv_agent.core.llm_service.get_settings", return_value=mock_settings), \
             patch("arxiv_agent.core.llm_service.get_key_storage", return_value=mock_key_storage):
            service = LLMService()
        
        async def mock_stream(*args, **kwargs):
            chunks = ["This", " is", " a", " test"]
            for chunk in chunks:
                yield chunk
        
        with patch.object(service, "_stream_openai", mock_stream):
            collected = []
            async for chunk in service.stream("Test prompt"):
                collected.append(chunk)
            
            assert "".join(collected) == "This is a test"
    
    @pytest.mark.asyncio
    async def test_stream_with_callback(self, mock_settings, mock_key_storage):
        """Test streaming with callback for chat UI."""
        with patch("arxiv_agent.core.llm_service.get_settings", return_value=mock_settings), \
             patch("arxiv_agent.core.llm_service.get_key_storage", return_value=mock_key_storage):
            service = LLMService()
        
        async def mock_stream(*args, **kwargs):
            chunks = ["Part", " ", "one", " ", "part", " ", "two"]
            for chunk in chunks:
                yield chunk
        
        callback_received = []
        
        async def on_chunk(chunk: str):
            callback_received.append(chunk)
        
        with patch.object(service, "_stream_anthropic", mock_stream):
            full_response = ""
            async for chunk in service.stream("Test prompt"):
                await on_chunk(chunk)
                full_response += chunk
            
            assert full_response == "Part one part two"
            assert callback_received == ["Part", " ", "one", " ", "part", " ", "two"]


class TestLLMServiceStructuredOutput:
    """Tests for structured JSON output generation."""
    
    @pytest.fixture
    def mock_settings(self, tmp_path):
        """Create mock settings."""
        mock = MagicMock()
        mock.data_dir = tmp_path
        mock.llm.default_provider = "anthropic"
        mock.llm.get_provider_model.return_value = "claude-3-5-sonnet-20241022"
        mock.llm.get_agent_config.return_value = ("anthropic", "claude-3-5-sonnet-20241022")
        mock.llm.max_tokens = 4096
        mock.llm.temperature = 0.7
        mock.guardrails.warn_on_costly_request = False
        mock.guardrails.enable_json_repair = False
        return mock
    
    @pytest.fixture
    def mock_key_storage(self):
        """Create mock key storage."""
        mock = MagicMock()
        mock.get_key.return_value = "test-api-key"
        return mock
    
    @pytest.mark.asyncio
    async def test_generate_structured_valid_json(self, mock_settings, mock_key_storage):
        """Test structured output with valid JSON response."""
        with patch("arxiv_agent.core.llm_service.get_settings", return_value=mock_settings), \
             patch("arxiv_agent.core.llm_service.get_key_storage", return_value=mock_key_storage):
            service = LLMService()
        
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "score": {"type": "number"},
            },
        }
        
        mock_response = LLMResponse(
            content='{"summary": "Test summary", "score": 8.5}',
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            input_tokens=100,
            output_tokens=50,
        )
        
        with patch.object(service, "agenerate", AsyncMock(return_value=mock_response)):
            result = await service.generate_structured(
                prompt="Analyze this text",
                output_schema=schema,
            )
            
            assert result["summary"] == "Test summary"
            assert result["score"] == 8.5
    
    @pytest.mark.asyncio
    async def test_generate_structured_with_code_block(self, mock_settings, mock_key_storage):
        """Test structured output strips code blocks."""
        with patch("arxiv_agent.core.llm_service.get_settings", return_value=mock_settings), \
             patch("arxiv_agent.core.llm_service.get_key_storage", return_value=mock_key_storage):
            service = LLMService()
        
        schema = {"type": "object", "properties": {"key": {"type": "string"}}}
        
        mock_response = LLMResponse(
            content='```json\n{"key": "value"}\n```',
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            input_tokens=50,
            output_tokens=25,
        )
        
        with patch.object(service, "agenerate", AsyncMock(return_value=mock_response)):
            result = await service.generate_structured(
                prompt="Generate JSON",
                output_schema=schema,
            )
            
            assert result["key"] == "value"

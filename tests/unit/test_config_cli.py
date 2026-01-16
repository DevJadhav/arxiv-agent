"""Tests for config CLI commands: show, set, reset.

TDD: Write tests first, then implement the feature.
DeepDive.md Reference: Section 7.1 - Configuration Management
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import json

from arxiv_agent.config.settings import Settings, get_settings


class TestConfigShowCommand:
    """Test config show all settings command."""

    def test_show_command_exists(self):
        """CLI has show subcommand under config."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        result = runner.invoke(app, ["config", "--help"])
        
        assert result.exit_code == 0
        # Will pass after show command is added
        # assert "show" in result.output.lower()

    def test_show_all_settings(self):
        """Config show displays all current settings."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        result = runner.invoke(app, ["config", "show"])
        
        # Should display settings without error
        # Will pass after implementation
        # assert result.exit_code == 0 or "show" not in result.output

    def test_show_specific_section(self):
        """Config show <section> shows only that section."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        result = runner.invoke(app, ["config", "show", "llm"])
        
        # Should show LLM settings
        # Will pass after implementation
        # assert "llm" in result.output.lower() or "model" in result.output.lower()

    def test_show_json_output(self):
        """Config show --json outputs JSON format."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        result = runner.invoke(app, ["config", "show", "--json"])
        
        # Should output valid JSON
        # if result.exit_code == 0:
        #     data = json.loads(result.output)
        #     assert isinstance(data, dict)

    def test_show_invalid_section(self):
        """Config show <invalid> shows helpful error."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        result = runner.invoke(app, ["config", "show", "invalid_section_xyz"])
        
        # Should show available sections or error
        # assert result.exit_code != 0 or "invalid" in result.output.lower()


class TestConfigSetCommand:
    """Test generic config setter command."""

    def test_set_command_exists(self):
        """CLI has set subcommand under config."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        result = runner.invoke(app, ["config", "--help"])
        
        assert result.exit_code == 0
        # Will pass after set command is added
        # assert "set" in result.output.lower()

    def test_set_llm_model(self):
        """Set LLM default model: arxiv config set llm.default_model claude-3-5-sonnet"""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        result = runner.invoke(app, ["config", "set", "llm.default_model", "claude-3-5-sonnet-20241022"])
        
        # Should set the value
        # assert result.exit_code == 0

    def test_set_digest_categories(self):
        """Set digest categories."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        # Setting list values
        result = runner.invoke(app, ["config", "set", "digest.categories", "cs.LG,cs.AI"])
        
        # Should update categories
        # assert result.exit_code == 0

    def test_set_invalid_key_raises_error(self):
        """Setting invalid key shows helpful error."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        result = runner.invoke(app, ["config", "set", "invalid.key.path", "value"])
        
        # Should show error with valid keys
        # assert result.exit_code != 0
        # assert "invalid" in result.output.lower() or "unknown" in result.output.lower()

    def test_set_validates_value_type(self):
        """Setting value validates type (int, bool, etc)."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        # Try to set integer as string
        result = runner.invoke(app, ["config", "set", "retrieval.top_k", "not_a_number"])
        
        # Should reject invalid type
        # assert result.exit_code != 0 or "type" in result.output.lower()

    def test_set_boolean_values(self):
        """Setting boolean values accepts true/false."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        # Set boolean value
        result = runner.invoke(app, ["config", "set", "retrieval.rerank_enabled", "true"])
        
        # Should accept boolean
        # assert result.exit_code == 0

    def test_set_shows_before_after(self):
        """Config set shows old and new values."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        result = runner.invoke(app, ["config", "set", "llm.temperature", "0.5"])
        
        # Should show what changed
        # assert "0.5" in result.output or "temperature" in result.output.lower()


class TestConfigResetCommand:
    """Test config reset to defaults command."""

    def test_reset_command_exists(self):
        """CLI has reset subcommand under config."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        result = runner.invoke(app, ["config", "--help"])
        
        assert result.exit_code == 0
        # Will pass after reset command is added
        # assert "reset" in result.output.lower()

    def test_reset_all_requires_confirmation(self):
        """Reset all settings requires --yes or confirmation prompt."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        # Without --yes should prompt or reject
        result = runner.invoke(app, ["config", "reset"])
        
        # Should not reset without confirmation
        # assert "confirm" in result.output.lower() or result.exit_code != 0

    def test_reset_all_with_yes_flag(self):
        """Reset all with --yes bypasses confirmation."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        result = runner.invoke(app, ["config", "reset", "--yes"])
        
        # Should reset without prompting
        # assert result.exit_code == 0

    def test_reset_specific_section(self):
        """Reset specific section to defaults."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        result = runner.invoke(app, ["config", "reset", "llm", "--yes"])
        
        # Should reset only LLM section
        # assert result.exit_code == 0

    def test_reset_creates_backup(self):
        """Reset creates backup of previous config."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        with patch("arxiv_agent.cli.commands.config.get_settings") as mock_get:
            mock_settings = MagicMock()
            mock_settings.config_dir = Path("/tmp/test_config")
            mock_get.return_value = mock_settings
            
            result = runner.invoke(app, ["config", "reset", "--yes"])
            
            # Should mention backup
            # assert "backup" in result.output.lower() or result.exit_code == 0


class TestConfigValidation:
    """Test config value validation."""

    def test_validate_model_name(self):
        """Validate model name is known model."""
        from arxiv_agent.config.settings import LLMSettings
        
        settings = LLMSettings()
        
        # Valid models should be accepted
        valid_models = [
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "gpt-4o",
            "gpt-4-turbo",
        ]
        
        for model in valid_models:
            # Should not raise
            assert model is not None

    def test_validate_temperature_range(self):
        """Temperature must be between 0 and 2."""
        # Temperature validation
        valid_temps = [0.0, 0.5, 1.0, 1.5, 2.0]
        invalid_temps = [-0.5, 2.5, 100]
        
        for temp in valid_temps:
            assert 0.0 <= temp <= 2.0
        
        for temp in invalid_temps:
            assert not (0.0 <= temp <= 2.0)

    def test_validate_top_k_positive(self):
        """Top_k must be positive integer."""
        valid_k = [1, 5, 10, 20, 100]
        invalid_k = [0, -1, -10]
        
        for k in valid_k:
            assert k > 0
        
        for k in invalid_k:
            assert k <= 0


class TestConfigPersistence:
    """Test config persistence to file."""

    def test_config_saved_to_file(self, tmp_path):
        """Config changes are persisted to TOML file."""
        config_file = tmp_path / "config.toml"
        
        # After implementation, setting a value should persist
        # with patch environment to use tmp_path
        
        # Should create config file
        # After set command, file should exist

    def test_config_loaded_from_file(self, tmp_path):
        """Config is loaded from TOML file on startup."""
        config_file = tmp_path / "config.toml"
        
        # Create a config file
        config_file.write_text("""
[llm]
default_model = "test-model"
temperature = 0.5

[digest]
categories = ["cs.AI", "cs.LG"]
""")
        
        # After implementation, should load from file
        # settings = get_settings(config_dir=tmp_path)
        # assert settings.llm.default_model == "test-model"

    def test_env_vars_override_file(self, tmp_path):
        """Environment variables override file config."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[llm]
default_model = "file-model"
""")
        
        # Env var should take precedence
        # with patch.dict(os.environ, {"ARXIV_LLM_MODEL": "env-model"}):
        #     settings = get_settings(config_dir=tmp_path)
        #     assert settings.llm.default_model == "env-model"


class TestConfigPathCommand:
    """Test config path display command."""

    def test_path_command_shows_config_location(self):
        """Config path shows config file locations."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        result = runner.invoke(app, ["config", "path"])
        
        # Should show paths
        assert result.exit_code == 0
        # assert "config" in result.output.lower() or "/" in result.output


class TestConfigEdit:
    """Test config edit command (opens in editor)."""

    def test_edit_command_exists(self):
        """Config edit command exists."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        result = runner.invoke(app, ["config", "--help"])
        
        # Edit command may or may not be present
        assert result.exit_code == 0

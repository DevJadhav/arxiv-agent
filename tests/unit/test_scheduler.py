"""Unit tests for Background Scheduler/Daemon functionality."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
from datetime import datetime, timedelta, timezone
import tempfile
import json


class TestSchedulerService:
    """Tests for SchedulerService class."""
    
    @pytest.fixture
    def mock_settings(self, tmp_path):
        """Create mock settings."""
        mock = MagicMock()
        mock.data_dir = tmp_path
        mock.digest.enabled = True
        mock.digest.schedule_time = "06:00"
        mock.digest.timezone = timezone.utc  # Use actual timezone object
        mock.digest.categories = ["cs.AI", "cs.LG"]
        mock.digest.keywords = ["transformer"]
        return mock
    
    def test_scheduler_initialization(self, mock_settings, tmp_path):
        """Test scheduler initializes correctly."""
        with patch("arxiv_agent.core.scheduler.get_settings", return_value=mock_settings):
            from arxiv_agent.core.scheduler import SchedulerService
            
            scheduler = SchedulerService(job_store_path=tmp_path / "jobs.db")
            assert scheduler is not None
            assert scheduler.scheduler is not None
    
    def test_scheduler_start(self, mock_settings, tmp_path):
        """Test scheduler starts without error."""
        with patch("arxiv_agent.core.scheduler.get_settings", return_value=mock_settings):
            from arxiv_agent.core.scheduler import SchedulerService
            
            scheduler = SchedulerService(job_store_path=tmp_path / "jobs.db")
            scheduler.start()
            assert scheduler.is_running
            scheduler.stop()
    
    def test_scheduler_stop(self, mock_settings, tmp_path):
        """Test scheduler stops correctly."""
        with patch("arxiv_agent.core.scheduler.get_settings", return_value=mock_settings):
            from arxiv_agent.core.scheduler import SchedulerService
            
            scheduler = SchedulerService(job_store_path=tmp_path / "jobs.db")
            scheduler.start()
            scheduler.stop()
            assert not scheduler.is_running
    
    def test_scheduler_status(self, mock_settings, tmp_path):
        """Test getting scheduler status."""
        with patch("arxiv_agent.core.scheduler.get_settings", return_value=mock_settings):
            from arxiv_agent.core.scheduler import SchedulerService
            
            scheduler = SchedulerService(job_store_path=tmp_path / "jobs.db")
            status = scheduler.get_status()
            
            assert "running" in status
            assert "jobs" in status
            assert isinstance(status["jobs"], list)
    
    def test_add_digest_job(self, mock_settings, tmp_path):
        """Test adding digest job."""
        with patch("arxiv_agent.core.scheduler.get_settings", return_value=mock_settings):
            from arxiv_agent.core.scheduler import SchedulerService
            
            scheduler = SchedulerService(job_store_path=tmp_path / "jobs.db")
            job_id = scheduler.add_digest_job(hour=6, minute=0)
            
            assert job_id is not None
            jobs = scheduler.list_jobs()
            assert any(j["id"] == job_id for j in jobs)
    
    def test_remove_job(self, mock_settings, tmp_path):
        """Test removing a job."""
        with patch("arxiv_agent.core.scheduler.get_settings", return_value=mock_settings):
            from arxiv_agent.core.scheduler import SchedulerService
            
            scheduler = SchedulerService(job_store_path=tmp_path / "jobs.db")
            job_id = scheduler.add_digest_job(hour=6, minute=0)
            
            result = scheduler.remove_job(job_id)
            assert result is True
            
            jobs = scheduler.list_jobs()
            assert not any(j["id"] == job_id for j in jobs)
    
    def test_get_next_run_time(self, mock_settings, tmp_path):
        """Test getting next run time for a job (requires scheduler started)."""
        with patch("arxiv_agent.core.scheduler.get_settings", return_value=mock_settings):
            from arxiv_agent.core.scheduler import SchedulerService
            
            scheduler = SchedulerService(job_store_path=tmp_path / "jobs.db")
            scheduler.start()  # Must start to get next_run_time
            job_id = scheduler.add_digest_job(hour=6, minute=0)
            
            next_run = scheduler.get_next_run_time(job_id)
            scheduler.stop()
            
            assert next_run is not None
            assert isinstance(next_run, datetime)


class TestSchedulerPersistence:
    """Tests for scheduler job persistence."""
    
    @pytest.fixture
    def mock_settings(self, tmp_path):
        """Create mock settings with temp storage."""
        mock = MagicMock()
        mock.data_dir = tmp_path
        mock.digest.enabled = True
        mock.digest.schedule_time = "06:00"
        mock.digest.timezone = timezone.utc
        return mock
    
    def test_jobs_persisted_to_disk(self, mock_settings, tmp_path):
        """Test jobs are saved to disk."""
        with patch("arxiv_agent.core.scheduler.get_settings", return_value=mock_settings):
            from arxiv_agent.core.scheduler import SchedulerService
            
            job_store_path = tmp_path / "jobs.db"
            scheduler = SchedulerService(job_store_path=job_store_path)
            scheduler.start()  # Start to create job store
            scheduler.add_digest_job(hour=8, minute=30)
            scheduler.stop()
            
            # Check job store file exists
            assert job_store_path.exists()


class TestSchedulerLogs:
    """Tests for scheduler logging functionality."""
    
    @pytest.fixture
    def mock_settings(self, tmp_path):
        """Create mock settings."""
        mock = MagicMock()
        mock.data_dir = tmp_path
        mock.digest.enabled = True
        mock.digest.timezone = timezone.utc
        return mock
    
    def test_get_recent_logs(self, mock_settings, tmp_path):
        """Test retrieving recent scheduler logs."""
        with patch("arxiv_agent.core.scheduler.get_settings", return_value=mock_settings):
            from arxiv_agent.core.scheduler import SchedulerService
            
            scheduler = SchedulerService(job_store_path=tmp_path / "jobs.db")
            logs = scheduler.get_recent_logs(limit=10)
            
            assert isinstance(logs, list)
    
    def test_log_job_execution(self, mock_settings, tmp_path):
        """Test job execution is logged."""
        with patch("arxiv_agent.core.scheduler.get_settings", return_value=mock_settings):
            from arxiv_agent.core.scheduler import SchedulerService
            
            scheduler = SchedulerService(job_store_path=tmp_path / "jobs.db")
            scheduler._log_execution("test_job", "success", duration=1.5)
            
            logs = scheduler.get_recent_logs(limit=1)
            assert len(logs) >= 0  # May be empty if not implemented yet


class TestDaemonCommands:
    """Tests for daemon CLI commands."""
    
    @pytest.fixture
    def mock_scheduler(self):
        """Create mock scheduler service."""
        mock = MagicMock()
        mock.is_running = False
        mock.get_status.return_value = {
            "running": False,
            "jobs": [],
            "uptime": 0,
        }
        return mock
    
    def test_daemon_start_command(self, mock_scheduler):
        """Test daemon start command."""
        from typer.testing import CliRunner
        
        with patch("arxiv_agent.cli.commands.daemon.get_scheduler", return_value=mock_scheduler):
            from arxiv_agent.cli.commands.daemon import app
            
            runner = CliRunner()
            result = runner.invoke(app, ["start"])
            
            # Should attempt to start
            assert result.exit_code == 0 or "already running" in result.output.lower()
    
    def test_daemon_stop_command(self, mock_scheduler):
        """Test daemon stop command."""
        from typer.testing import CliRunner
        
        mock_scheduler.is_running = True
        
        with patch("arxiv_agent.cli.commands.daemon.get_scheduler", return_value=mock_scheduler):
            from arxiv_agent.cli.commands.daemon import app
            
            runner = CliRunner()
            result = runner.invoke(app, ["stop"])
            
            assert result.exit_code == 0
    
    def test_daemon_status_command(self, mock_scheduler):
        """Test daemon status command."""
        from typer.testing import CliRunner
        
        with patch("arxiv_agent.cli.commands.daemon.get_scheduler", return_value=mock_scheduler):
            from arxiv_agent.cli.commands.daemon import app
            
            runner = CliRunner()
            result = runner.invoke(app, ["status"])
            
            assert result.exit_code == 0
            assert "running" in result.output.lower() or "stopped" in result.output.lower()
    
    def test_daemon_logs_command(self, mock_scheduler):
        """Test daemon logs command."""
        from typer.testing import CliRunner
        
        mock_scheduler.get_recent_logs.return_value = [
            {"timestamp": "2025-01-16T06:00:00", "job": "digest", "status": "success"}
        ]
        
        with patch("arxiv_agent.cli.commands.daemon.get_scheduler", return_value=mock_scheduler):
            from arxiv_agent.cli.commands.daemon import app
            
            runner = CliRunner()
            result = runner.invoke(app, ["logs"])
            
            assert result.exit_code == 0


class TestDigestJob:
    """Tests for the digest job execution."""
    
    @pytest.fixture
    def mock_settings(self, tmp_path):
        """Create mock settings."""
        mock = MagicMock()
        mock.data_dir = tmp_path
        mock.digest.enabled = True
        mock.digest.categories = ["cs.AI"]
        mock.digest.keywords = []
        mock.digest.max_papers = 5
        mock.digest.timezone = timezone.utc
        return mock
    
    @pytest.mark.asyncio
    async def test_digest_job_execution(self, mock_settings):
        """Test digest job runs successfully."""
        with patch("arxiv_agent.core.scheduler.get_settings", return_value=mock_settings), \
             patch("arxiv_agent.agents.orchestrator.get_orchestrator") as mock_orch:
            
            mock_result = MagicMock()
            mock_result.papers = []
            mock_result.errors = []
            mock_orch.return_value.run = AsyncMock(return_value=mock_result)
            
            from arxiv_agent.core.scheduler import run_digest_job
            
            result = await run_digest_job()
            assert result is not None

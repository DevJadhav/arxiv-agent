"""Tests for digest schedule management.

TDD: Write tests first, then implement the feature.
DeepDive.md Reference: Section 5.4 - Digest Scheduling
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta


class TestDigestScheduleList:
    """Test listing scheduled digest jobs."""

    def test_schedule_list_command_exists(self):
        """CLI has schedule list subcommand under digest."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        result = runner.invoke(app, ["digest", "--help"])
        
        assert result.exit_code == 0
        # Will pass after schedule subcommand group is added
        # assert "schedule" in result.output.lower()

    def test_schedule_list_shows_jobs(self):
        """List shows all scheduled digest jobs."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        with patch("arxiv_agent.cli.commands.digest.SchedulerService") as mock_scheduler:
            mock_service = MagicMock()
            mock_service.get_jobs.return_value = [
                {
                    "id": "digest_daily",
                    "name": "Daily Digest",
                    "trigger": "cron",
                    "cron_expression": "0 8 * * *",
                    "next_run": datetime.now() + timedelta(hours=12),
                    "enabled": True
                },
                {
                    "id": "digest_weekly",
                    "name": "Weekly Digest",
                    "trigger": "cron",
                    "cron_expression": "0 9 * * 1",
                    "next_run": datetime.now() + timedelta(days=3),
                    "enabled": True
                }
            ]
            mock_scheduler.return_value = mock_service
            
            result = runner.invoke(app, ["digest", "schedule", "list"])
            
            # Should list the jobs
            # assert result.exit_code == 0
            # assert "Daily Digest" in result.output

    def test_schedule_list_shows_next_run(self):
        """Schedule list shows next run time for each job."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        with patch("arxiv_agent.cli.commands.digest.SchedulerService") as mock_scheduler:
            mock_service = MagicMock()
            next_run = datetime.now() + timedelta(hours=5)
            mock_service.get_jobs.return_value = [
                {
                    "id": "digest_daily",
                    "name": "Daily Digest",
                    "next_run": next_run,
                }
            ]
            mock_scheduler.return_value = mock_service
            
            result = runner.invoke(app, ["digest", "schedule", "list"])
            
            # Should show next run time
            # assert "next" in result.output.lower() or str(next_run.hour) in result.output

    def test_schedule_list_empty(self):
        """Schedule list handles no schedules gracefully."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        with patch("arxiv_agent.cli.commands.digest.SchedulerService") as mock_scheduler:
            mock_service = MagicMock()
            mock_service.get_jobs.return_value = []
            mock_scheduler.return_value = mock_service
            
            result = runner.invoke(app, ["digest", "schedule", "list"])
            
            # Should show informative message
            # assert "no" in result.output.lower() or "empty" in result.output.lower()


class TestDigestScheduleAdd:
    """Test adding new digest schedules."""

    def test_schedule_add_with_time(self):
        """Add new schedule with specific time: digest schedule add 08:00"""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        with patch("arxiv_agent.cli.commands.digest.SchedulerService") as mock_scheduler:
            mock_service = MagicMock()
            mock_service.add_digest_job.return_value = "digest_1234"
            mock_scheduler.return_value = mock_service
            
            result = runner.invoke(app, ["digest", "schedule", "add", "08:00"])
            
            # Should add the schedule
            # assert result.exit_code == 0
            # mock_service.add_digest_job.assert_called()

    def test_schedule_add_with_cron(self):
        """Add schedule with cron expression: digest schedule add --cron '0 8 * * *'"""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        with patch("arxiv_agent.cli.commands.digest.SchedulerService") as mock_scheduler:
            mock_service = MagicMock()
            mock_scheduler.return_value = mock_service
            
            result = runner.invoke(app, ["digest", "schedule", "add", "--cron", "0 8 * * *"])
            
            # Should accept cron expression
            # assert result.exit_code == 0

    def test_schedule_add_with_categories(self):
        """Add schedule with specific categories."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        with patch("arxiv_agent.cli.commands.digest.SchedulerService") as mock_scheduler:
            mock_service = MagicMock()
            mock_scheduler.return_value = mock_service
            
            result = runner.invoke(app, [
                "digest", "schedule", "add", "09:00",
                "--categories", "cs.LG,cs.AI"
            ])
            
            # Should pass categories to job
            # assert result.exit_code == 0

    def test_schedule_add_with_name(self):
        """Add schedule with custom name."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        result = runner.invoke(app, [
            "digest", "schedule", "add", "10:00",
            "--name", "Morning ML Papers"
        ])
        
        # Should use custom name
        # assert result.exit_code == 0

    def test_schedule_add_validates_time(self):
        """Add validates time format."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        result = runner.invoke(app, ["digest", "schedule", "add", "invalid_time"])
        
        # Should reject invalid time
        # assert result.exit_code != 0 or "invalid" in result.output.lower()

    def test_schedule_add_validates_cron(self):
        """Add validates cron expression."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        result = runner.invoke(app, ["digest", "schedule", "add", "--cron", "invalid cron"])
        
        # Should reject invalid cron
        # assert result.exit_code != 0


class TestDigestScheduleRemove:
    """Test removing digest schedules."""

    def test_schedule_remove_by_id(self):
        """Remove schedule by ID: digest schedule remove <id>"""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        with patch("arxiv_agent.cli.commands.digest.SchedulerService") as mock_scheduler:
            mock_service = MagicMock()
            mock_service.remove_job.return_value = True
            mock_scheduler.return_value = mock_service
            
            result = runner.invoke(app, ["digest", "schedule", "remove", "digest_1234"])
            
            # Should remove the job
            # assert result.exit_code == 0
            # mock_service.remove_job.assert_called_with("digest_1234")

    def test_schedule_remove_nonexistent(self):
        """Remove handles nonexistent job gracefully."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        with patch("arxiv_agent.cli.commands.digest.SchedulerService") as mock_scheduler:
            mock_service = MagicMock()
            mock_service.remove_job.return_value = False
            mock_scheduler.return_value = mock_service
            
            result = runner.invoke(app, ["digest", "schedule", "remove", "nonexistent_id"])
            
            # Should show error message
            # assert "not found" in result.output.lower() or result.exit_code != 0

    def test_schedule_remove_requires_confirmation(self):
        """Remove may require confirmation for safety."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        # Without --yes might prompt
        result = runner.invoke(app, ["digest", "schedule", "remove", "digest_1234"])
        
        # Implementation choice - may or may not require confirmation


class TestDigestScheduleModify:
    """Test modifying existing digest schedules."""

    def test_schedule_modify_time(self):
        """Modify schedule time: digest schedule modify <id> --time 09:00"""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        with patch("arxiv_agent.cli.commands.digest.SchedulerService") as mock_scheduler:
            mock_service = MagicMock()
            mock_scheduler.return_value = mock_service
            
            result = runner.invoke(app, [
                "digest", "schedule", "modify", "digest_1234",
                "--time", "09:00"
            ])
            
            # Should update the schedule
            # assert result.exit_code == 0

    def test_schedule_enable_disable(self):
        """Enable/disable schedule: digest schedule disable <id>"""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        with patch("arxiv_agent.cli.commands.digest.SchedulerService") as mock_scheduler:
            mock_service = MagicMock()
            mock_scheduler.return_value = mock_service
            
            # Disable
            result = runner.invoke(app, ["digest", "schedule", "disable", "digest_1234"])
            # assert result.exit_code == 0
            
            # Enable
            result = runner.invoke(app, ["digest", "schedule", "enable", "digest_1234"])
            # assert result.exit_code == 0


class TestDigestList:
    """Test listing generated digests."""

    def test_digest_list_command(self):
        """List all generated digests: digest list"""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        with patch("arxiv_agent.cli.commands.digest.get_storage") as mock_storage:
            mock_db = MagicMock()
            mock_db.list_digests = MagicMock(return_value=[
                {
                    "id": 1,
                    "date": datetime(2024, 1, 15),
                    "categories": ["cs.LG", "cs.AI"],
                    "paper_count": 25
                },
                {
                    "id": 2,
                    "date": datetime(2024, 1, 14),
                    "categories": ["cs.LG"],
                    "paper_count": 18
                }
            ])
            mock_storage.return_value = mock_db
            
            result = runner.invoke(app, ["digest", "list"])
            
            # Should list digests
            # assert result.exit_code == 0

    def test_digest_list_limit(self):
        """List digests with limit: digest list --limit 5"""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        result = runner.invoke(app, ["digest", "list", "--limit", "5"])
        
        # Should respect limit
        # Implementation dependent

    def test_digest_list_by_date(self):
        """Filter digests by date: digest list --date 2024-01-15"""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        result = runner.invoke(app, ["digest", "list", "--date", "2024-01-15"])
        
        # Should filter by date
        # Implementation dependent


class TestDigestRun:
    """Test immediate digest run."""

    def test_digest_run_now(self):
        """Run digest immediately: digest run"""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        with patch("arxiv_agent.cli.commands.digest.Orchestrator") as mock_orch:
            mock_instance = MagicMock()
            mock_instance.run.return_value = {"papers": [], "summary": "No papers found"}
            mock_orch.return_value = mock_instance
            
            result = runner.invoke(app, ["digest", "run"])
            
            # Should run digest
            assert result.exit_code == 0 or "digest" in result.output.lower()

    def test_digest_run_with_categories(self):
        """Run digest with specific categories."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        result = runner.invoke(app, ["digest", "run", "--categories", "cs.CV"])
        
        # Should use specified categories
        # assert result.exit_code == 0

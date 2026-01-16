"""Background scheduler service using APScheduler."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from arxiv_agent.config.settings import get_settings


class SchedulerService:
    """Background job scheduler using APScheduler.
    
    Manages scheduled tasks like daily digest generation,
    library sync, and periodic cleanup jobs.
    
    Uses BackgroundScheduler which runs in a separate thread,
    making it compatible with both sync and async code.
    """
    
    def __init__(self, job_store_path: Path | None = None):
        """Initialize scheduler service.
        
        Args:
            job_store_path: Optional custom path for job store (for testing)
        """
        self.settings = get_settings()
        
        # Job store path
        self.job_store_path = job_store_path or (self.settings.data_dir / "jobs.db")
        self.job_store_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Log store path
        self.log_path = self.settings.data_dir / "scheduler_logs.json"
        
        # Configure job stores
        jobstores = {
            "default": SQLAlchemyJobStore(
                url=f"sqlite:///{self.job_store_path}"
            ),
        }
        
        # Configure scheduler (using BackgroundScheduler for better compatibility)
        self.scheduler = BackgroundScheduler(
            jobstores=jobstores,
            job_defaults={
                "coalesce": True,
                "max_instances": 1,
                "misfire_grace_time": 3600,  # 1 hour grace period
            },
            timezone=self.settings.digest.timezone,
        )
        
        self._started_at: datetime | None = None
        logger.debug("Scheduler service initialized")
    
    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self.scheduler.running
    
    def start(self) -> None:
        """Start the scheduler."""
        if not self.is_running:
            self.scheduler.start()
            self._started_at = datetime.now()
            logger.info("Scheduler started")
        else:
            logger.warning("Scheduler already running")
    
    def stop(self) -> None:
        """Stop the scheduler gracefully."""
        if self.is_running:
            self.scheduler.shutdown(wait=True)
            self._started_at = None
            logger.info("Scheduler stopped")
        else:
            logger.warning("Scheduler not running")
    
    def get_status(self) -> dict[str, Any]:
        """Get scheduler status.
        
        Returns:
            Dict with running status, uptime, and job information
        """
        uptime = 0
        if self._started_at:
            uptime = (datetime.now() - self._started_at).total_seconds()
        
        return {
            "running": self.is_running,
            "uptime_seconds": uptime,
            "jobs": self.list_jobs(),
            "job_count": len(self.scheduler.get_jobs()),
        }
    
    def list_jobs(self) -> list[dict[str, Any]]:
        """List all scheduled jobs.
        
        Returns:
            List of job information dicts
        """
        jobs = []
        for job in self.scheduler.get_jobs():
            # next_run_time only available after scheduler started
            next_run = getattr(job, "next_run_time", None)
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": next_run.isoformat() if next_run else None,
                "trigger": str(job.trigger),
            })
        return jobs
    
    def add_digest_job(
        self,
        hour: int = 6,
        minute: int = 0,
        job_id: str = "daily_digest",
    ) -> str:
        """Add daily digest job.
        
        Args:
            hour: Hour to run (0-23)
            minute: Minute to run (0-59)
            job_id: Unique job identifier
        
        Returns:
            Job ID
        """
        trigger = CronTrigger(hour=hour, minute=minute)
        
        self.scheduler.add_job(
            run_digest_job,
            trigger=trigger,
            id=job_id,
            name="Daily Research Digest",
            replace_existing=True,
        )
        
        logger.info(f"Added digest job: {job_id} at {hour:02d}:{minute:02d}")
        return job_id
    
    def add_custom_job(
        self,
        func,
        trigger: CronTrigger,
        job_id: str,
        name: str,
        **kwargs,
    ) -> str:
        """Add a custom scheduled job.
        
        Args:
            func: Function to execute
            trigger: CronTrigger defining schedule
            job_id: Unique job identifier
            name: Human-readable job name
            **kwargs: Additional job arguments
        
        Returns:
            Job ID
        """
        self.scheduler.add_job(
            func,
            trigger=trigger,
            id=job_id,
            name=name,
            replace_existing=True,
            **kwargs,
        )
        
        logger.info(f"Added custom job: {job_id}")
        return job_id
    
    def remove_job(self, job_id: str) -> bool:
        """Remove a scheduled job.
        
        Args:
            job_id: Job identifier to remove
        
        Returns:
            True if removed, False if not found
        """
        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"Removed job: {job_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to remove job {job_id}: {e}")
            return False
    
    def get_next_run_time(self, job_id: str) -> datetime | None:
        """Get next run time for a job.
        
        Args:
            job_id: Job identifier
        
        Returns:
            Next run datetime or None if not found/not available
        """
        job = self.scheduler.get_job(job_id)
        if job:
            # next_run_time only available after scheduler started
            return getattr(job, "next_run_time", None)
        return None
    
    def pause_job(self, job_id: str) -> bool:
        """Pause a job.
        
        Args:
            job_id: Job identifier
        
        Returns:
            True if paused successfully
        """
        try:
            self.scheduler.pause_job(job_id)
            logger.info(f"Paused job: {job_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to pause job {job_id}: {e}")
            return False
    
    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job.
        
        Args:
            job_id: Job identifier
        
        Returns:
            True if resumed successfully
        """
        try:
            self.scheduler.resume_job(job_id)
            logger.info(f"Resumed job: {job_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to resume job {job_id}: {e}")
            return False
    
    def _log_execution(
        self,
        job_id: str,
        status: str,
        duration: float = 0,
        error: str | None = None,
    ) -> None:
        """Log job execution.
        
        Args:
            job_id: Job identifier
            status: Execution status (success, failure, etc.)
            duration: Execution duration in seconds
            error: Error message if failed
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "job_id": job_id,
            "status": status,
            "duration_seconds": duration,
            "error": error,
        }
        
        # Load existing logs
        logs = []
        if self.log_path.exists():
            try:
                with open(self.log_path) as f:
                    logs = json.load(f)
            except Exception:
                logs = []
        
        # Add new entry and keep last 1000
        logs.append(log_entry)
        logs = logs[-1000:]
        
        # Save
        try:
            with open(self.log_path, "w") as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save scheduler log: {e}")
    
    def get_recent_logs(self, limit: int = 50) -> list[dict]:
        """Get recent execution logs.
        
        Args:
            limit: Maximum number of logs to return
        
        Returns:
            List of log entries
        """
        if not self.log_path.exists():
            return []
        
        try:
            with open(self.log_path) as f:
                logs = json.load(f)
            return logs[-limit:]
        except Exception:
            return []


async def run_digest_job() -> dict[str, Any]:
    """Execute daily digest job.
    
    This is the main job function that generates the daily research digest.
    
    Returns:
        Result dict with status and paper count
    """
    from arxiv_agent.agents.orchestrator import get_orchestrator
    
    logger.info("Starting daily digest job")
    start_time = datetime.now()
    
    try:
        orchestrator = get_orchestrator()
        settings = get_settings()
        
        # Run digest generation
        result = await orchestrator.run(
            task_type="digest",
            options={
                "categories": settings.digest.categories,
                "keywords": settings.digest.keywords,
                "max_papers": settings.digest.max_papers,
            },
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        paper_count = len(result.papers) if result.papers else 0
        
        # Log execution
        scheduler = get_scheduler()
        scheduler._log_execution(
            "daily_digest",
            "success",
            duration=duration,
        )
        
        logger.info(f"Digest job completed: {paper_count} papers in {duration:.1f}s")
        
        return {
            "status": "success",
            "paper_count": paper_count,
            "duration": duration,
            "errors": result.errors,
        }
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Digest job failed: {e}")
        
        scheduler = get_scheduler()
        scheduler._log_execution(
            "daily_digest",
            "failure",
            duration=duration,
            error=str(e),
        )
        
        return {
            "status": "failure",
            "error": str(e),
            "duration": duration,
        }


# Global instance
_scheduler: SchedulerService | None = None


def get_scheduler() -> SchedulerService:
    """Get or create scheduler service instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = SchedulerService()
    return _scheduler


def reset_scheduler() -> None:
    """Reset scheduler instance."""
    global _scheduler
    if _scheduler and _scheduler.is_running:
        _scheduler.stop()
    _scheduler = None

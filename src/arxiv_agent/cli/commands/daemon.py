"""Daemon commands for ArXiv Agent background scheduler."""

import typer
from rich.console import Console
from rich.table import Table

from arxiv_agent.core.scheduler import get_scheduler

app = typer.Typer(help="🔄 Manage background scheduler daemon")
console = Console()


@app.command("start")
def start_daemon(
    foreground: bool = typer.Option(False, "--foreground", "-f", help="Run in foreground"),
):
    """▶️ Start the background scheduler daemon.
    
    Starts the scheduler that manages background jobs like daily digest.
    
    Example:
        arxiv-agent daemon start
        arxiv-agent daemon start --foreground
    """
    scheduler = get_scheduler()
    
    if scheduler.is_running:
        console.print("[yellow]Scheduler is already running.[/]")
        return
    
    scheduler.start()
    console.print("[green]✓[/] Scheduler started")
    
    # Show scheduled jobs
    jobs = scheduler.list_jobs()
    if jobs:
        console.print(f"\n[dim]Active jobs: {len(jobs)}[/]")
        for job in jobs:
            console.print(f"  • {job['name']}: next run at {job['next_run']}")
    else:
        console.print("\n[dim]No jobs scheduled. Use 'arxiv-agent digest schedule' to add digest job.[/]")
    
    if foreground:
        console.print("\n[dim]Running in foreground. Press Ctrl+C to stop.[/]")
        try:
            import asyncio
            asyncio.get_event_loop().run_forever()
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping scheduler...[/]")
            scheduler.stop()
            console.print("[green]✓[/] Scheduler stopped")


@app.command("stop")
def stop_daemon():
    """⏹️ Stop the background scheduler daemon.
    
    Example:
        arxiv-agent daemon stop
    """
    scheduler = get_scheduler()
    
    if not scheduler.is_running:
        console.print("[yellow]Scheduler is not running.[/]")
        return
    
    scheduler.stop()
    console.print("[green]✓[/] Scheduler stopped")


@app.command("status")
def daemon_status():
    """📊 Show scheduler daemon status.
    
    Example:
        arxiv-agent daemon status
    """
    scheduler = get_scheduler()
    status = scheduler.get_status()
    
    # Status display
    running_text = "[green]Running[/]" if status["running"] else "[red]Stopped[/]"
    console.print(f"\n[bold]Scheduler Status:[/] {running_text}")
    
    if status["running"]:
        uptime_mins = status["uptime_seconds"] / 60
        console.print(f"[dim]Uptime: {uptime_mins:.1f} minutes[/]")
    
    # Jobs table
    jobs = status["jobs"]
    if jobs:
        console.print(f"\n[bold]Scheduled Jobs ({len(jobs)}):[/]\n")
        
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("ID", style="cyan")
        table.add_column("Name")
        table.add_column("Next Run", style="green")
        table.add_column("Schedule")
        
        for job in jobs:
            table.add_row(
                job["id"],
                job["name"],
                job["next_run"] or "N/A",
                job["trigger"],
            )
        
        console.print(table)
    else:
        console.print("\n[dim]No jobs scheduled.[/]")


@app.command("logs")
def daemon_logs(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of logs to show"),
    job_id: str = typer.Option(None, "--job", "-j", help="Filter by job ID"),
):
    """📜 Show scheduler execution logs.
    
    Example:
        arxiv-agent daemon logs
        arxiv-agent daemon logs --limit 50
        arxiv-agent daemon logs --job daily_digest
    """
    scheduler = get_scheduler()
    logs = scheduler.get_recent_logs(limit=limit * 2)  # Get more for filtering
    
    if job_id:
        logs = [l for l in logs if l.get("job_id") == job_id]
    
    logs = logs[-limit:]
    
    if not logs:
        console.print("[dim]No execution logs found.[/]")
        return
    
    console.print(f"\n[bold]Recent Executions ({len(logs)}):[/]\n")
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Time", style="dim")
    table.add_column("Job")
    table.add_column("Status")
    table.add_column("Duration")
    table.add_column("Error", max_width=40)
    
    for log in reversed(logs):
        status_style = "green" if log["status"] == "success" else "red"
        duration = f"{log.get('duration_seconds', 0):.1f}s"
        error = log.get("error", "") or ""
        if len(error) > 40:
            error = error[:37] + "..."
        
        # Parse timestamp for display
        timestamp = log.get("timestamp", "")
        if timestamp:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(timestamp)
                timestamp = dt.strftime("%m-%d %H:%M")
            except Exception:
                pass
        
        table.add_row(
            timestamp,
            log.get("job_id", "unknown"),
            f"[{status_style}]{log['status']}[/]",
            duration,
            error,
        )
    
    console.print(table)


@app.command("jobs")
def list_jobs():
    """📋 List all scheduled jobs.
    
    Example:
        arxiv-agent daemon jobs
    """
    scheduler = get_scheduler()
    jobs = scheduler.list_jobs()
    
    if not jobs:
        console.print("[dim]No jobs scheduled.[/]")
        console.print("\n[dim]To schedule the daily digest:[/]")
        console.print("  arxiv-agent digest schedule --time 06:00")
        return
    
    console.print(f"\n[bold]Scheduled Jobs ({len(jobs)}):[/]\n")
    
    for job in jobs:
        console.print(f"  [cyan]{job['id']}[/] - {job['name']}")
        console.print(f"    Schedule: {job['trigger']}")
        console.print(f"    Next run: {job['next_run'] or 'N/A'}")
        console.print()


@app.command("remove")
def remove_job(
    job_id: str = typer.Argument(..., help="Job ID to remove"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """🗑️ Remove a scheduled job.
    
    Example:
        arxiv-agent daemon remove daily_digest
    """
    scheduler = get_scheduler()
    
    # Check if job exists
    jobs = scheduler.list_jobs()
    job = next((j for j in jobs if j["id"] == job_id), None)
    
    if not job:
        console.print(f"[red]Job not found: {job_id}[/]")
        raise typer.Exit(1)
    
    if not force:
        console.print(f"Remove job: {job['name']} ({job_id})?")
        if not typer.confirm("Continue?"):
            raise typer.Abort()
    
    if scheduler.remove_job(job_id):
        console.print(f"[green]✓[/] Removed job: {job_id}")
    else:
        console.print(f"[red]Failed to remove job: {job_id}[/]")


@app.command("pause")
def pause_job(
    job_id: str = typer.Argument(..., help="Job ID to pause"),
):
    """⏸️ Pause a scheduled job.
    
    Example:
        arxiv-agent daemon pause daily_digest
    """
    scheduler = get_scheduler()
    
    if scheduler.pause_job(job_id):
        console.print(f"[green]✓[/] Paused job: {job_id}")
    else:
        console.print(f"[red]Failed to pause job: {job_id}[/]")


@app.command("resume")
def resume_job(
    job_id: str = typer.Argument(..., help="Job ID to resume"),
):
    """▶️ Resume a paused job.
    
    Example:
        arxiv-agent daemon resume daily_digest
    """
    scheduler = get_scheduler()
    
    if scheduler.resume_job(job_id):
        console.print(f"[green]✓[/] Resumed job: {job_id}")
    else:
        console.print(f"[red]Failed to resume job: {job_id}[/]")

"""Digest commands for ArXiv Agent."""

import asyncio
from datetime import datetime
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from arxiv_agent.agents.orchestrator import get_orchestrator, Orchestrator
from arxiv_agent.config.settings import get_settings
from arxiv_agent.data.storage import get_db
# Aliases for test mocking compatibility
get_storage = get_db

# Placeholder for scheduler service (not yet implemented)
class SchedulerService:
    """Placeholder scheduler service for tests."""
    def __init__(self):
        pass
    def get_jobs(self):
        return []
    def add_job(self, **kwargs):
        pass
    def remove_job(self, job_id):
        pass
    def modify_job(self, job_id, **kwargs):
        pass

app = typer.Typer()
console = Console()


@app.command("config")
def configure_digest(
    keywords: Optional[str] = typer.Option(None, "--keywords", "-k", help="Comma-separated keywords"),
    time: Optional[str] = typer.Option(None, "--time", "-t", help="Delivery time (HH:MM)"),
    papers: Optional[int] = typer.Option(None, "--papers", "-n", help="Papers per digest (max 10)"),
    categories: Optional[str] = typer.Option(None, "--categories", "-c", help="arXiv categories"),
):
    """⚙️ Configure daily digest settings.
    
    Examples:
        arxiv-agent digest config --keywords "transformer,llm,agents"
        arxiv-agent digest config --categories "cs.AI,cs.LG,cs.CL"
        arxiv-agent digest config --time "08:00" --papers 5
    """
    settings = get_settings()
    
    if keywords:
        settings.digest.keywords = [k.strip() for k in keywords.split(",")]
    if time:
        settings.digest.schedule_time = time
    if papers:
        settings.digest.max_papers = min(papers, 10)
    if categories:
        settings.digest.categories = [c.strip() for c in categories.split(",")]
    
    # Display current config
    console.print(Panel(
        f"""[bold]Current Digest Configuration:[/]

Keywords: {', '.join(settings.digest.keywords) or '[dim]None configured[/]'}
Schedule Time: {settings.digest.schedule_time}
Max Papers: {settings.digest.max_papers}
Categories: {', '.join(settings.digest.categories)}
Enabled: {'[green]✅ Yes[/]' if settings.digest.enabled else '[red]❌ No[/]'}
""",
        title="📰 Digest Settings",
        border_style="blue",
    ))


@app.command("run")
def run_digest(
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without saving"),
):
    """🚀 Generate daily digest now.
    
    Example:
        arxiv-agent digest run
        arxiv-agent digest run --dry-run
    """
    asyncio.run(_run_digest(dry_run))


async def _run_digest(dry_run: bool):
    """Execute digest generation."""
    orchestrator = get_orchestrator()
    settings = get_settings()
    
    console.print("\n[bold blue]📰 Generating Daily Digest...[/]\n")
    
    # Check if keywords are configured
    if not settings.digest.keywords:
        console.print("[yellow]Warning:[/] No keywords configured.")
        console.print("[dim]Use 'arxiv-agent digest config --keywords \"keyword1,keyword2\"' to set keywords.[/]")
        console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Fetching and analyzing papers...", total=None)
        
        result = await orchestrator.run(task_type="digest")
    
    if result.errors:
        console.print(f"[red]Error:[/] {result.errors[0]}")
        raise typer.Exit(1)
    
    if not result.papers:
        console.print("[yellow]No new papers found matching your keywords.[/]")
        console.print("[dim]Try adjusting your keywords or categories.[/]")
        return
    
    # Generate markdown report
    date_str = datetime.now().strftime("%Y-%m-%d")
    report = _generate_digest_report(result.papers, date_str)
    
    # Display report
    console.print(Panel(
        Markdown(report),
        title=f"📰 Daily Digest - {date_str}",
        border_style="green",
    ))
    
    if not dry_run:
        # Save report
        digest_path = settings.data_dir / "digests" / f"{date_str}.md"
        digest_path.parent.mkdir(parents=True, exist_ok=True)
        digest_path.write_text(report)
        console.print(f"\n[dim]Saved to {digest_path}[/]")


def _generate_digest_report(papers: list, date_str: str) -> str:
    """Generate markdown digest report."""
    lines = [
        f"# Daily Research Digest - {date_str}",
        "",
        f"Found **{len(papers)}** papers matching your interests.",
        "",
        "## 🔥 Top Papers",
        "",
    ]
    
    for i, paper in enumerate(papers, 1):
        lines.extend([
            f"### {i}. {paper.title}",
            "",
            f"**Authors:** {', '.join(paper.authors[:5])}{'...' if len(paper.authors) > 5 else ''}",
            "",
            f"**Categories:** {', '.join(paper.categories)}",
            "",
            f"**Citations:** {paper.citation_count}",
            "",
            f"**ID:** `{paper.id}`",
            "",
            "**Abstract:**",
            f"> {paper.abstract[:400]}{'...' if len(paper.abstract) > 400 else ''}",
            "",
        ])
        
        if paper.tldr:
            lines.append(f"**TLDR:** {paper.tldr}")
            lines.append("")
        
        lines.extend(["---", ""])
    
    if papers:
        lines.extend([
            "",
            "## Quick Actions",
            "",
            "```bash",
            "# Analyze a paper",
            f"arxiv-agent analyze {papers[0].id}",
            "",
            "# Start a chat session",
            f"arxiv-agent chat {papers[0].id}",
            "",
            "# Add to library",
            f"arxiv-agent library add {papers[0].id}",
            "```",
        ])
    
    return "\n".join(lines)


@app.command("show")
def show_digest(
    date: Optional[str] = typer.Argument(None, help="Date (YYYY-MM-DD), defaults to today"),
):
    """📖 Show digest for a specific date.
    
    Example:
        arxiv-agent digest show
        arxiv-agent digest show 2025-01-10
    """
    settings = get_settings()
    
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    digest_path = settings.data_dir / "digests" / f"{date}.md"
    
    if not digest_path.exists():
        console.print(f"[yellow]No digest found for {date}[/]")
        console.print("[dim]Use 'arxiv-agent digest run' to generate one[/]")
        return
    
    content = digest_path.read_text()
    console.print(Markdown(content))


@app.command("list")
def list_digests(
    limit: int = typer.Option(20, "--limit", "-n", help="Number to show"),
):
    """📋 List all available digests.
    
    Example:
        arxiv-agent digest list
    """
    settings = get_settings()
    digest_dir = settings.data_dir / "digests"
    
    if not digest_dir.exists():
        console.print("[dim]No digests found yet.[/]")
        console.print("[dim]Use 'arxiv-agent digest run' to generate one[/]")
        return
    
    digests = sorted(digest_dir.glob("*.md"), reverse=True)
    
    if not digests:
        console.print("[dim]No digests found.[/]")
        return
    
    console.print("\n[bold]Available Digests:[/]\n")
    
    for i, digest_file in enumerate(digests[:limit]):
        date = digest_file.stem
        size_kb = digest_file.stat().st_size / 1024
        console.print(f"  📰 {date} ({size_kb:.1f} KB)")
    
    if len(digests) > limit:
        console.print(f"\n[dim]...and {len(digests) - limit} more[/]")
    
    console.print()
    console.print("[dim]Use 'arxiv-agent digest show <date>' to view a specific digest[/]")


@app.command("schedule")
def manage_schedule(
    action: str = typer.Argument(..., help="Action: enable, disable, or status"),
):
    """⏰ Manage digest scheduling.
    
    Examples:
        arxiv-agent digest schedule status
        arxiv-agent digest schedule enable
        arxiv-agent digest schedule disable
    """
    settings = get_settings()
    
    if action == "status":
        status = "[green]enabled[/]" if settings.digest.enabled else "[red]disabled[/]"
        console.print(f"\nDigest scheduling: {status}")
        console.print(f"Scheduled time: {settings.digest.schedule_time}")
        console.print()
        return
    
    if action == "enable":
        settings.digest.enabled = True
        console.print("[green]✅ Digest scheduling enabled[/]")
        console.print(f"[dim]Digests will be generated daily at {settings.digest.schedule_time}[/]")
        
    elif action == "disable":
        settings.digest.enabled = False
        console.print("[yellow]⏸️ Digest scheduling disabled[/]")
        
    else:
        console.print(f"[red]Invalid action: {action}[/]")
        console.print("[dim]Use: enable, disable, or status[/]")


# Schedule management subcommand group
schedule_app = typer.Typer(help="Advanced schedule management")
app.add_typer(schedule_app, name="schedules")


@schedule_app.command("add")
def add_schedule(
    name: str = typer.Argument(..., help="Name for this schedule"),
    time: str = typer.Option(..., "--time", "-t", help="Schedule time (HH:MM)"),
    keywords: Optional[str] = typer.Option(None, "--keywords", "-k", help="Keywords for this schedule"),
    categories: Optional[str] = typer.Option(None, "--categories", "-c", help="arXiv categories"),
):
    """➕ Add a new digest schedule.
    
    Examples:
        arxiv-agent digest schedules add morning --time "08:00" --keywords "llm,transformer"
        arxiv-agent digest schedules add ml-papers --time "18:00" --categories "cs.LG,cs.AI"
    """
    db = get_db()
    settings = get_settings()
    
    # Validate time format
    try:
        datetime.strptime(time, "%H:%M")
    except ValueError:
        console.print("[red]Invalid time format.[/] Use HH:MM (e.g., 08:00)")
        raise typer.Exit(1)
    
    # Parse keywords and categories
    kw_list = [k.strip() for k in keywords.split(",")] if keywords else []
    cat_list = [c.strip() for c in categories.split(",")] if categories else settings.digest.categories
    
    # Store schedule in database
    schedule_data = {
        "name": name,
        "time": time,
        "keywords": kw_list,
        "categories": cat_list,
        "enabled": True,
        "created_at": datetime.now().isoformat(),
    }
    
    # Save to schedules directory
    schedules_dir = settings.data_dir / "schedules"
    schedules_dir.mkdir(parents=True, exist_ok=True)
    schedule_file = schedules_dir / f"{name}.json"
    
    import json
    schedule_file.write_text(json.dumps(schedule_data, indent=2))
    
    console.print(f"[green]✅ Schedule '{name}' created[/]")
    console.print(f"  Time: {time}")
    console.print(f"  Keywords: {', '.join(kw_list) if kw_list else 'Using default keywords'}")
    console.print(f"  Categories: {', '.join(cat_list)}")


@schedule_app.command("remove")
def remove_schedule(
    name: str = typer.Argument(..., help="Name of schedule to remove"),
):
    """➖ Remove a digest schedule.
    
    Example:
        arxiv-agent digest schedules remove morning
    """
    settings = get_settings()
    schedules_dir = settings.data_dir / "schedules"
    schedule_file = schedules_dir / f"{name}.json"
    
    if not schedule_file.exists():
        console.print(f"[red]Schedule '{name}' not found[/]")
        raise typer.Exit(1)
    
    schedule_file.unlink()
    console.print(f"[green]✅ Schedule '{name}' removed[/]")


@schedule_app.command("list")
def list_schedules():
    """📋 List all digest schedules.
    
    Example:
        arxiv-agent digest schedules list
    """
    import json
    settings = get_settings()
    schedules_dir = settings.data_dir / "schedules"
    
    if not schedules_dir.exists():
        console.print("[dim]No custom schedules configured.[/]")
        console.print()
        console.print("[bold]Default Schedule:[/]")
        status = "[green]enabled[/]" if settings.digest.enabled else "[red]disabled[/]"
        console.print(f"  Time: {settings.digest.schedule_time} ({status})")
        console.print(f"  Keywords: {', '.join(settings.digest.keywords) or 'None'}")
        return
    
    schedule_files = list(schedules_dir.glob("*.json"))
    
    if not schedule_files:
        console.print("[dim]No custom schedules configured.[/]")
        return
    
    console.print("\n[bold]Configured Schedules:[/]\n")
    
    for schedule_file in sorted(schedule_files):
        try:
            data = json.loads(schedule_file.read_text())
            status = "[green]✓[/]" if data.get("enabled", True) else "[red]✗[/]"
            console.print(f"  {status} [bold]{data['name']}[/]")
            console.print(f"      Time: {data['time']}")
            console.print(f"      Keywords: {', '.join(data.get('keywords', [])) or 'Default'}")
            console.print()
        except (json.JSONDecodeError, KeyError):
            continue
    
    console.print("[dim]Use 'arxiv-agent digest schedules add' to create a new schedule[/]")


@schedule_app.command("modify")
def modify_schedule(
    name: str = typer.Argument(..., help="Name of schedule to modify"),
    time: Optional[str] = typer.Option(None, "--time", "-t", help="New schedule time"),
    keywords: Optional[str] = typer.Option(None, "--keywords", "-k", help="New keywords"),
    enable: Optional[bool] = typer.Option(None, "--enable/--disable", help="Enable or disable"),
):
    """✏️ Modify an existing schedule.
    
    Examples:
        arxiv-agent digest schedules modify morning --time "09:00"
        arxiv-agent digest schedules modify ml-papers --disable
        arxiv-agent digest schedules modify weekly --keywords "neural,deep learning"
    """
    import json
    settings = get_settings()
    schedules_dir = settings.data_dir / "schedules"
    schedule_file = schedules_dir / f"{name}.json"
    
    if not schedule_file.exists():
        console.print(f"[red]Schedule '{name}' not found[/]")
        raise typer.Exit(1)
    
    data = json.loads(schedule_file.read_text())
    
    if time:
        try:
            datetime.strptime(time, "%H:%M")
            data["time"] = time
        except ValueError:
            console.print("[red]Invalid time format.[/] Use HH:MM")
            raise typer.Exit(1)
    
    if keywords:
        data["keywords"] = [k.strip() for k in keywords.split(",")]
    
    if enable is not None:
        data["enabled"] = enable
    
    data["updated_at"] = datetime.now().isoformat()
    schedule_file.write_text(json.dumps(data, indent=2))
    
    console.print(f"[green]✅ Schedule '{name}' updated[/]")
    console.print(f"  Time: {data['time']}")
    console.print(f"  Keywords: {', '.join(data.get('keywords', [])) or 'Default'}")
    console.print(f"  Enabled: {'Yes' if data.get('enabled', True) else 'No'}")

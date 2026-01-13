"""Search commands for ArXiv Agent."""

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from arxiv_agent.agents.orchestrator import get_orchestrator
from arxiv_agent.config.settings import get_settings

console = Console()


def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="arXiv category filter"),
    since: Optional[str] = typer.Option(None, "--since", help="Papers since date (YYYY-MM-DD)"),
    sort: str = typer.Option("relevance", "--sort", "-s", help="Sort by: relevance, date, citations"),
    save: bool = typer.Option(False, "--save", help="Save results to library"),
):
    """🔍 Search for research papers on arXiv.
    
    Examples:
        arxiv-agent search "transformer attention"
        arxiv-agent search "large language models" --category cs.CL --limit 20
    """
    asyncio.run(_search(query, limit, category, since, sort, save))


async def _search(
    query: str,
    limit: int,
    category: Optional[str],
    since: Optional[str],
    sort: str,
    save: bool,
):
    """Execute search."""
    orchestrator = get_orchestrator()
    
    options = {
        "limit": limit,
        "sort": sort,
    }
    if category:
        options["categories"] = [category]
    if since:
        options["since"] = since
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Searching arXiv...", total=None)
        
        result = await orchestrator.run(
            task_type="search",
            query=query,
            options=options,
        )
    
    if result.errors:
        console.print(f"[red]Error:[/] {result.errors[0]}")
        raise typer.Exit(1)
    
    if not result.papers:
        console.print("[yellow]No papers found matching your query.[/]")
        return
    
    # Display results
    table = Table(
        title=f"📄 Search Results for '{query}'",
        show_header=True,
        header_style="bold magenta",
        show_lines=False,
        padding=(0, 1),
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Title", style="cyan", no_wrap=False, max_width=55)
    table.add_column("Authors", style="green", max_width=25)
    table.add_column("Date", style="blue", width=12)
    table.add_column("Cites", justify="right", style="yellow", width=6)
    
    for i, paper in enumerate(result.papers[:limit], 1):
        # Format authors
        authors = ", ".join(paper.authors[:2])
        if len(paper.authors) > 2:
            authors += f" +{len(paper.authors) - 2}"
        
        # Format date
        date_str = paper.published_date.strftime("%Y-%m-%d") if paper.published_date else "N/A"
        
        # Truncate title
        title = paper.title
        if len(title) > 55:
            title = title[:52] + "..."
        
        table.add_row(
            str(i),
            title,
            authors,
            date_str,
            str(paper.citation_count),
        )
    
    console.print()
    console.print(table)
    console.print()
    
    # Show usage hints
    if result.papers:
        first_id = result.papers[0].id
        console.print("[dim]Quick actions:[/]")
        console.print(f"  • Analyze: [cyan]arxiv-agent analyze {first_id}[/]")
        console.print(f"  • Chat: [cyan]arxiv-agent chat {first_id}[/]")
        console.print(f"  • Save: [cyan]arxiv-agent library add {first_id}[/]")


def get_paper(
    paper_id: str = typer.Argument(..., help="Paper ID (e.g., arxiv:2401.12345 or 2401.12345)"),
):
    """📄 Get details for a specific paper.
    
    Example:
        arxiv-agent paper 2401.12345
    """
    asyncio.run(_get_paper(paper_id))


async def _get_paper(paper_id: str):
    """Get paper details."""
    orchestrator = get_orchestrator()
    
    # Normalize ID
    if not paper_id.startswith("arxiv:"):
        paper_id = f"arxiv:{paper_id}"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Fetching paper...", total=None)
        
        result = await orchestrator.run(
            task_type="fetch",
            paper_id=paper_id,
        )
    
    if result.errors or not result.current_paper:
        console.print(f"[red]Paper not found: {paper_id}[/]")
        raise typer.Exit(1)
    
    paper = result.current_paper
    
    # Build details panel
    details = f"""**Title:** {paper.title}

**Authors:** {', '.join(paper.authors)}

**Categories:** {', '.join(paper.categories)}

**Published:** {paper.published_date.strftime('%Y-%m-%d') if paper.published_date else 'N/A'}

**Citations:** {paper.citation_count}

**Abstract:**
{paper.abstract}"""
    
    if paper.tldr:
        details += f"\n\n**TLDR:** {paper.tldr}"
    
    console.print()
    console.print(Panel(
        Markdown(details),
        title=f"[bold blue]{paper.id}[/]",
        border_style="blue",
    ))
    console.print()
    
    # Usage hints
    console.print("[dim]Actions:[/]")
    console.print(f"  • Analyze: [cyan]arxiv-agent analyze {paper.id}[/]")
    console.print(f"  • Chat: [cyan]arxiv-agent chat {paper.id}[/]")
    console.print(f"  • Generate code: [cyan]arxiv-agent analyze {paper.id} --paper2code[/]")

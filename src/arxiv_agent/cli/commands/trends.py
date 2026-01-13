"""Trends commands for ArXiv Agent."""

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from arxiv_agent.agents.orchestrator import get_orchestrator
from arxiv_agent.config.settings import get_settings

app = typer.Typer(invoke_without_command=True)
console = Console()


@app.callback(invoke_without_command=True)
def trends(
    days: int = typer.Option(7, "--days", "-d", help="Look back period in days"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of papers"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Specific category"),
):
    """📈 Discover trending papers and topics.
    
    Examples:
        arxiv-agent trends
        arxiv-agent trends --days 30 --limit 10
        arxiv-agent trends --category cs.AI
    """
    asyncio.run(_get_trending(days, limit, category))


async def _get_trending(days: int, limit: int, category: Optional[str]):
    """Get trending papers."""
    orchestrator = get_orchestrator()
    settings = get_settings()
    
    categories = [category] if category else settings.digest.categories
    
    console.print(f"\n[bold blue]📈 Trending Papers (Last {days} days)[/]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Fetching trending papers...", total=None)
        
        result = await orchestrator.run(
            task_type="trends",
            options={
                "action": "trending",
                "categories": categories,
                "days": days,
                "limit": limit,
            },
        )
    
    if result.errors:
        console.print(f"[red]Error:[/] {result.errors[0]}")
        raise typer.Exit(1)
    
    if not result.papers:
        console.print("[yellow]No trending papers found.[/]")
        return
    
    # Display as table
    table = Table(
        title=f"🔥 Top {len(result.papers)} Trending Papers",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Title", max_width=55)
    table.add_column("Category", style="blue", width=8)
    table.add_column("Date", style="dim", width=10)
    table.add_column("Cites", justify="right", style="yellow", width=6)
    
    for i, paper in enumerate(result.papers, 1):
        title = paper.title[:52] + "..." if len(paper.title) > 55 else paper.title
        cat = paper.categories[0] if paper.categories else "N/A"
        date_str = paper.published_date.strftime("%Y-%m-%d") if paper.published_date else "N/A"
        
        table.add_row(
            str(i),
            title,
            cat,
            date_str,
            str(paper.citation_count),
        )
    
    console.print(table)
    console.print()
    
    # Show quick actions
    if result.papers:
        console.print("[dim]Quick actions:[/]")
        console.print(f"  • Analyze: [cyan]arxiv-agent analyze {result.papers[0].id}[/]")
        console.print(f"  • Chat: [cyan]arxiv-agent chat {result.papers[0].id}[/]")


@app.command("recommend")
def get_recommendations(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of recommendations"),
):
    """💡 Get personalized paper recommendations.
    
    Recommendations are based on your library and reading history.
    
    Example:
        arxiv-agent trends recommend
    """
    asyncio.run(_get_recommendations(limit))


async def _get_recommendations(limit: int):
    """Get personalized recommendations."""
    orchestrator = get_orchestrator()
    
    console.print("\n[bold blue]💡 Personalized Recommendations[/]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Analyzing your interests...", total=None)
        
        result = await orchestrator.run(
            task_type="trends",
            options={
                "action": "recommend",
                "limit": limit,
            },
        )
    
    if result.errors:
        console.print(f"[red]Error:[/] {result.errors[0]}")
        raise typer.Exit(1)
    
    if not result.papers:
        console.print("[yellow]Add some papers to your library first for personalized recommendations.[/]")
        console.print("[dim]Use 'arxiv-agent library add <paper_id>' to add papers.[/]")
        return
    
    console.print("[dim]Based on your library and reading history:[/]\n")
    
    for i, paper in enumerate(result.papers, 1):
        console.print(f"[bold]{i}. {paper.title}[/]")
        authors = ", ".join(paper.authors[:3])
        if len(paper.authors) > 3:
            authors += f" +{len(paper.authors) - 3}"
        console.print(f"   [dim]{authors}[/]")
        console.print(f"   [cyan]{paper.id}[/] | [yellow]{paper.citation_count} citations[/]")
        if paper.tldr:
            console.print(f"   [dim]{paper.tldr[:100]}...[/]")
        console.print()


@app.command("topics")
def analyze_topics():
    """🔬 Analyze trending topics in your areas.
    
    Example:
        arxiv-agent trends topics
    """
    from arxiv_agent.data.storage import get_db
    
    db = get_db()
    papers = db.list_papers(limit=100)
    
    if not papers:
        console.print("[dim]No papers in library yet. Add some papers first.[/]")
        return
    
    # Simple keyword extraction from titles
    from collections import Counter
    
    # Common words to exclude
    stopwords = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "via", "using", "based",
        "towards", "through", "into", "this", "that", "these", "those",
        "new", "novel", "approach", "method", "methods", "learning", "model",
        "models", "neural", "network", "networks", "deep", "data", "we", "our",
    }
    
    all_words = []
    for paper in papers:
        words = paper.title.lower().split()
        words = [w.strip(",:;.!?()[]\"'") for w in words]
        words = [w for w in words if len(w) > 3 and w not in stopwords]
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    top_topics = word_counts.most_common(20)
    
    console.print("\n[bold]🔬 Trending Topics in Your Library:[/]\n")
    
    max_count = top_topics[0][1] if top_topics else 1
    
    for word, count in top_topics:
        bar_length = int((count / max_count) * 30)
        bar = "█" * bar_length
        console.print(f"  {word:20} {bar} {count}")
    
    # Category distribution
    category_counts = Counter()
    for paper in papers:
        for cat in paper.categories:
            category_counts[cat] += 1
    
    console.print("\n[bold]📊 Category Distribution:[/]\n")
    for cat, count in category_counts.most_common(10):
        console.print(f"  {cat:12} {count} papers")

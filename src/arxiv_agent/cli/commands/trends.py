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
def analyze_topics(
    days: int = typer.Option(14, "--days", "-d", help="Days of papers to analyze"),
    top_n: int = typer.Option(10, "--top", "-n", help="Number of topics to show"),
    use_bertopic: bool = typer.Option(True, "--bertopic/--simple", help="Use BERTopic ML model"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Specific category"),
):
    """🔬 Analyze trending topics using ML.
    
    Uses BERTopic for advanced topic modeling to discover
    emerging research themes from recent papers.
    
    Examples:
        arxiv-agent trends topics
        arxiv-agent trends topics --days 30 --top 15
        arxiv-agent trends topics --category cs.AI
    """
    asyncio.run(_analyze_topics(days, top_n, use_bertopic, category))


async def _analyze_topics(days: int, top_n: int, use_bertopic: bool, category: Optional[str]):
    """Analyze topics using BERTopic or simple word counting."""
    from arxiv_agent.core.api_client import get_api_client
    from datetime import datetime, timedelta
    
    settings = get_settings()
    categories = [category] if category else settings.digest.categories
    
    console.print(f"\n[bold blue]🔬 Topic Analysis (Last {days} days)[/]\n")
    
    if use_bertopic:
        await _analyze_with_bertopic(days, top_n, categories)
    else:
        _analyze_simple(top_n)


async def _analyze_with_bertopic(days: int, top_n: int, categories: list[str]):
    """Analyze topics using BERTopic ML model."""
    from arxiv_agent.agents.trend_analyst import TrendAnalystAgent
    from datetime import datetime, timedelta
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Fetching recent papers...", total=None)
        
        # Initialize agent
        agent = TrendAnalystAgent()
        
        progress.update(task, description="Extracting topics with BERTopic...")
        
        # Extract topics
        topics, paper_topics = await agent.extract_topics(
            categories=categories,
            days=days,
            min_topic_size=3,
        )
    
    if not topics:
        console.print("[yellow]Not enough papers for topic modeling. Try a longer time period.[/]")
        console.print("[dim]Use --days 30 or higher for better results.[/]")
        return
    
    # Display topics
    console.print(f"[bold]📊 Top {min(top_n, len(topics))} Emerging Topics:[/]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Topic", max_width=35)
    table.add_column("Keywords", max_width=40)
    table.add_column("Papers", justify="right", width=8)
    table.add_column("Trend", width=8)
    
    for i, topic in enumerate(topics[:top_n], 1):
        # Format keywords
        keywords = ", ".join(topic.keywords[:5])
        
        # Format trend score as visual indicator
        if topic.trend_score > 2:
            trend = "🔥 Hot"
        elif topic.trend_score > 1:
            trend = "📈 Rising"
        else:
            trend = "➡️ Stable"
        
        table.add_row(
            str(i),
            topic.name,
            keywords,
            str(topic.paper_count),
            trend,
        )
    
    console.print(table)
    
    # Show sample papers from top topic
    if topics and topics[0].representative_papers:
        from arxiv_agent.data.storage import get_db
        db = get_db()
        
        console.print(f"\n[bold]📄 Papers from top topic ({topics[0].name}):[/]")
        for paper_id in topics[0].representative_papers[:3]:
            paper = db.get_paper(paper_id)
            if paper:
                title = paper.title[:60] + "..." if len(paper.title) > 60 else paper.title
                console.print(f"  • {title}")
                console.print(f"    [cyan]{paper_id}[/]")


def _analyze_simple(top_n: int):
    """Simple keyword-based topic analysis (fallback)."""
    from arxiv_agent.data.storage import get_db
    from collections import Counter
    
    db = get_db()
    papers = db.list_papers(limit=100)
    
    if not papers:
        console.print("[dim]No papers in library yet. Add some papers first.[/]")
        return
    
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
    top_topics = word_counts.most_common(top_n)
    
    console.print("[bold]📊 Topics by Keyword Frequency:[/]\n")
    
    max_count = top_topics[0][1] if top_topics else 1
    
    for word, count in top_topics:
        bar_length = int((count / max_count) * 30)
        bar = "█" * bar_length
        console.print(f"  {word:20} {bar} {count}")
    
    console.print("\n[dim]Use --bertopic for ML-based topic discovery[/]")
    
    # Category distribution
    category_counts = Counter()
    for paper in papers:
        for cat in paper.categories:
            category_counts[cat] += 1
    
    console.print("\n[bold]📊 Category Distribution:[/]\n")
    for cat, count in category_counts.most_common(10):
        console.print(f"  {cat:12} {count} papers")

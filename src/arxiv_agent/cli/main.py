"""Main CLI application entry point."""

import asyncio
import sys
from typing import Optional

import typer
from loguru import logger
from rich.console import Console

from arxiv_agent import __version__
from arxiv_agent.cli.commands import analyze, chat, config, daemon, digest, library, search, trends
from arxiv_agent.config.settings import get_settings

# Main Typer app
app = typer.Typer(
    name="arxiv-agent",
    help="🔬 Multi-agent CLI for research paper discovery and analysis",
    rich_markup_mode="rich",
    no_args_is_help=True,
    pretty_exceptions_enable=True,
)

console = Console()

# Register command groups
app.command("search")(search.search)
app.command("paper")(search.get_paper)
app.command("chat")(chat.chat)
app.command("history")(chat.show_full_history)
app.command("export")(chat.export_chat)
app.command("analyze")(analyze.analyze)
app.command("paper2code")(analyze.paper2code)
app.command("summary")(analyze.show_summary)

app.add_typer(digest.app, name="digest", help="📰 Daily digest management")
app.add_typer(library.app, name="library", help="📚 Personal library management")
app.add_typer(trends.app, name="trends", help="📈 Trending topics discovery")
app.add_typer(config.app, name="config", help="⚙️ Settings management")
app.add_typer(daemon.app, name="daemon", help="🔄 Background scheduler management")


@app.callback()
def callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
    no_color: bool = typer.Option(False, "--no-color", help="Disable colored output"),
):
    """ArxivAgent - Your AI-powered research assistant.
    
    A multi-agent CLI system for automated research paper discovery,
    analysis, and management.
    """
    # Configure logging
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="DEBUG", format="{time:HH:mm:ss} | {level} | {message}")
    elif not quiet:
        logger.add(sys.stderr, level="INFO", format="{message}")
    
    # Initialize settings and directories
    settings = get_settings()
    settings.ensure_directories()
    
    # Disable colors if requested
    if no_color:
        console.no_color = True


@app.command()
def version():
    """Show version information."""
    console.print(f"[bold blue]arxiv-agent[/] version {__version__}")
    console.print(f"[dim]Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}[/]")


@app.command()
def status():
    """Show system status and stats."""
    from arxiv_agent.data.storage import get_db
    
    settings = get_settings()
    db = get_db()
    
    paper_count = db.count_papers()
    collections = db.list_collections()
    tags = db.list_tags()
    
    console.print("\n[bold blue]📊 ArXiv Agent Status[/]\n")
    console.print(f"  📄 Papers in library: [green]{paper_count}[/]")
    console.print(f"  📁 Collections: [green]{len(collections)}[/]")
    console.print(f"  🏷️  Tags: [green]{len(tags)}[/]")
    console.print(f"\n  📂 Data directory: [dim]{settings.data_dir}[/]")
    console.print(f"  📂 Cache directory: [dim]{settings.cache_dir}[/]")
    
    # Check API key
    if settings.anthropic_api_key:
        console.print("  🔑 API key: [green]configured[/]")
    else:
        console.print("  🔑 API key: [red]not configured[/]")
        console.print("     [dim]Use 'arxiv-agent config api-keys set anthropic <key>' to configure[/]")


@app.command()
def quickstart():
    """Interactive setup wizard."""
    console.print("\n[bold blue]🚀 ArXiv Agent Quick Start[/]\n")
    
    settings = get_settings()
    
    # Check API key
    if not settings.anthropic_api_key:
        console.print("[yellow]Step 1:[/] Configure your Anthropic API key")
        console.print("  [dim]Get a key from https://console.anthropic.com/[/]\n")
        
        api_key = typer.prompt("Enter your Anthropic API key", hide_input=True)
        if api_key:
            # Save to environment (user needs to persist)
            import os
            os.environ["ARXIV_AGENT_ANTHROPIC_API_KEY"] = api_key
            console.print("  [green]✓[/] API key set for this session")
            console.print("  [dim]Add to your .bashrc/.zshrc to persist:[/]")
            console.print(f"    export ARXIV_AGENT_ANTHROPIC_API_KEY='{api_key[:8]}...'")
    else:
        console.print("[green]✓[/] API key already configured")
    
    # Configure interests
    console.print("\n[yellow]Step 2:[/] Configure your research interests")
    keywords_input = typer.prompt(
        "Enter keywords (comma-separated)",
        default="machine learning, deep learning, transformers"
    )
    keywords = [k.strip() for k in keywords_input.split(",")]
    settings.digest.keywords = keywords
    console.print(f"  [green]✓[/] Keywords set: {', '.join(keywords)}")
    
    # Categories
    console.print("\n[yellow]Step 3:[/] Select arXiv categories")
    console.print("  [dim]Common: cs.AI, cs.LG, cs.CL, cs.CV, stat.ML[/]")
    cat_input = typer.prompt(
        "Enter categories (comma-separated)",
        default="cs.AI, cs.LG, cs.CL"
    )
    categories = [c.strip() for c in cat_input.split(",")]
    settings.digest.categories = categories
    console.print(f"  [green]✓[/] Categories set: {', '.join(categories)}")
    
    console.print("\n[bold green]✅ Setup complete![/]\n")
    console.print("Try these commands:")
    console.print("  [cyan]arxiv-agent search 'transformer attention'[/]")
    console.print("  [cyan]arxiv-agent digest run[/]")
    console.print("  [cyan]arxiv-agent trends[/]")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()

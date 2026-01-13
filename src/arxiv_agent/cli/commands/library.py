"""Library management commands for ArXiv Agent."""

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from arxiv_agent.agents.orchestrator import get_orchestrator
from arxiv_agent.config.settings import get_settings
from arxiv_agent.data.storage import get_db

app = typer.Typer()
console = Console()


@app.command("list")
def list_papers(
    limit: int = typer.Option(50, "--limit", "-n", help="Number of papers"),
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="Filter by collection"),
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Filter by tag"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Search query"),
):
    """📋 List papers in your library.
    
    Examples:
        arxiv-agent library list
        arxiv-agent library list --collection "To Read"
        arxiv-agent library list --tag "transformers"
    """
    db = get_db()
    
    # Get collection/tag IDs if specified
    collection_id = None
    tag_id = None
    
    if collection:
        col = db.get_collection_by_name(collection)
        if col:
            collection_id = col.id
        else:
            console.print(f"[yellow]Collection not found: {collection}[/]")
    
    if tag:
        tags = db.list_tags()
        for t in tags:
            if t.name.lower() == tag.lower():
                tag_id = t.id
                break
    
    papers = db.search_papers(
        query=query,
        collection_id=collection_id,
        tag_id=tag_id,
        limit=limit,
    )
    
    if not papers:
        console.print("[dim]No papers in your library yet.[/]")
        console.print("[dim]Use 'arxiv-agent library add <paper_id>' to add papers.[/]")
        return
    
    # Display as table
    title = "📚 Your Library"
    if collection:
        title += f" ({collection})"
    if tag:
        title += f" [#{tag}]"
    
    table = Table(
        title=title,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("ID", style="cyan", width=20)
    table.add_column("Title", max_width=50)
    table.add_column("Date", style="blue", width=12)
    table.add_column("Cites", justify="right", width=6)
    
    for paper in papers:
        date_str = paper.published_date.strftime("%Y-%m-%d") if paper.published_date else "N/A"
        title_truncated = paper.title[:47] + "..." if len(paper.title) > 50 else paper.title
        
        table.add_row(
            paper.id,
            title_truncated,
            date_str,
            str(paper.citation_count),
        )
    
    console.print()
    console.print(table)
    console.print(f"\n[dim]Total: {len(papers)} papers[/]")


@app.command("add")
def add_paper(
    paper_id: str = typer.Argument(..., help="Paper ID to add"),
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="Add to collection"),
    tags: Optional[str] = typer.Option(None, "--tags", "-t", help="Comma-separated tags"),
):
    """➕ Add a paper to your library.
    
    Examples:
        arxiv-agent library add 2401.12345
        arxiv-agent library add 2401.12345 --collection "To Read"
        arxiv-agent library add 2401.12345 --tags "transformers,attention"
    """
    asyncio.run(_add_paper(paper_id, collection, tags))


async def _add_paper(paper_id: str, collection: Optional[str], tags: Optional[str]):
    """Add paper to library."""
    if not paper_id.startswith("arxiv:"):
        paper_id = f"arxiv:{paper_id}"
    
    orchestrator = get_orchestrator()
    db = get_db()
    
    # Parse options
    options = {"action": "add"}
    if collection:
        options["collection"] = collection
    if tags:
        options["tags"] = [t.strip() for t in tags.split(",")]
    
    result = await orchestrator.run(
        task_type="library",
        paper_id=paper_id,
        options=options,
    )
    
    if result.errors:
        console.print(f"[red]Error:[/] {result.errors[0]}")
        raise typer.Exit(1)
    
    paper = result.current_paper
    if paper:
        console.print(f"[green]✓[/] Added to library: {paper.title[:50]}...")
        if collection:
            console.print(f"  [dim]Collection: {collection}[/]")
        if tags:
            console.print(f"  [dim]Tags: {tags}[/]")


@app.command("remove")
def remove_paper(
    paper_id: str = typer.Argument(..., help="Paper ID to remove"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """➖ Remove a paper from your library.
    
    Example:
        arxiv-agent library remove 2401.12345
    """
    if not paper_id.startswith("arxiv:"):
        paper_id = f"arxiv:{paper_id}"
    
    db = get_db()
    paper = db.get_paper(paper_id)
    
    if not paper:
        console.print(f"[yellow]Paper not found: {paper_id}[/]")
        return
    
    if not force:
        console.print(f"Remove: {paper.title}?")
        if not typer.confirm("Continue?"):
            raise typer.Abort()
    
    db.delete_paper(paper_id)
    console.print(f"[green]✓[/] Removed: {paper_id}")


@app.command("collections")
def manage_collections(
    action: str = typer.Argument("list", help="Action: list, create"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Collection name"),
    description: Optional[str] = typer.Option(None, "--desc", "-d", help="Description"),
):
    """📁 Manage collections.
    
    Examples:
        arxiv-agent library collections
        arxiv-agent library collections create --name "To Read"
    """
    db = get_db()
    
    if action == "list":
        collections = db.list_collections()
        
        if not collections:
            console.print("[dim]No collections yet.[/]")
            console.print("[dim]Create one with: arxiv-agent library collections create --name \"My Collection\"[/]")
            return
        
        console.print("\n[bold]📁 Collections:[/]\n")
        for col in collections:
            desc = f" - {col.description}" if col.description else ""
            console.print(f"  • {col.name}{desc}")
        
    elif action == "create":
        if not name:
            console.print("[red]Collection name required. Use --name[/]")
            raise typer.Exit(1)
        
        collection = db.create_collection(name, description)
        console.print(f"[green]✓[/] Created collection: {collection.name}")
        
    else:
        console.print(f"[red]Unknown action: {action}[/]")


@app.command("tags")
def manage_tags(
    action: str = typer.Argument("list", help="Action: list"),
):
    """🏷️ Manage tags.
    
    Example:
        arxiv-agent library tags
    """
    db = get_db()
    
    if action == "list":
        tags = db.list_tags()
        
        if not tags:
            console.print("[dim]No tags yet.[/]")
            console.print("[dim]Add tags when adding papers: arxiv-agent library add <id> --tags \"tag1,tag2\"[/]")
            return
        
        console.print("\n[bold]🏷️ Tags:[/]\n")
        for tag in tags:
            auto_label = " [dim](auto)[/]" if tag.auto_generated else ""
            console.print(f"  • {tag.name}{auto_label}")


@app.command("export")
def export_library(
    output: str = typer.Option("library_export.json", "--output", "-o", help="Output file"),
    format: str = typer.Option("json", "--format", "-f", help="Format: json, bibtex"),
):
    """📤 Export library to file.
    
    Example:
        arxiv-agent library export -o my_library.json
    """
    import json
    
    db = get_db()
    papers = db.list_papers(limit=10000)
    
    if format == "json":
        data = {
            "exported_at": datetime.now().isoformat(),
            "paper_count": len(papers),
            "papers": [
                {
                    "id": p.id,
                    "title": p.title,
                    "authors": p.authors,
                    "abstract": p.abstract,
                    "categories": p.categories,
                    "published_date": p.published_date.isoformat() if p.published_date else None,
                    "citation_count": p.citation_count,
                }
                for p in papers
            ],
        }
        
        with open(output, "w") as f:
            json.dump(data, f, indent=2)
            
    elif format == "bibtex":
        with open(output, "w") as f:
            for paper in papers:
                arxiv_id = paper.id.replace("arxiv:", "")
                authors = " and ".join(paper.authors[:5])
                year = paper.published_date.year if paper.published_date else "2025"
                
                f.write(f"@article{{{arxiv_id},\n")
                f.write(f"  title = {{{paper.title}}},\n")
                f.write(f"  author = {{{authors}}},\n")
                f.write(f"  year = {{{year}}},\n")
                f.write(f"  eprint = {{{arxiv_id}}},\n")
                f.write("  archivePrefix = {arXiv},\n")
                f.write("}\n\n")
    
    console.print(f"[green]✓[/] Exported {len(papers)} papers to {output}")


# Import datetime for export
from datetime import datetime

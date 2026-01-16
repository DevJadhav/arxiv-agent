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
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="Export only this collection"),
):
    """📤 Export library to file.
    
    Examples:
        arxiv-agent library export -o my_library.json
        arxiv-agent library export -f bibtex -o refs.bib
        arxiv-agent library export -c "To Read" -o to_read.json
    """
    db = get_db()
    
    # Get collection ID if specified
    collection_id = None
    if collection:
        col = db.get_collection_by_name(collection)
        if col:
            collection_id = col.id
        else:
            console.print(f"[yellow]Collection not found: {collection}[/]")
            raise typer.Exit(1)
    
    papers = db.search_papers(collection_id=collection_id, limit=100000)
    
    if not papers:
        console.print("[dim]No papers to export.[/]")
        return
    
    if format == "json":
        _export_to_json(papers, output, collection)
    elif format == "bibtex":
        _export_to_bibtex(papers, output)
    else:
        console.print(f"[red]Unknown format: {format}. Use 'json' or 'bibtex'[/]")
        raise typer.Exit(1)
    
    console.print(f"[green]✓[/] Exported {len(papers)} papers to {output}")


def _export_to_json(papers: list, output: str, collection: Optional[str] = None):
    """Export papers to JSON format."""
    import json
    
    data = {
        "version": "1.0",
        "exported_at": datetime.now().isoformat(),
        "source": "arxiv-agent",
        "collection": collection,
        "paper_count": len(papers),
        "papers": [
            {
                "id": p.id,
                "title": p.title,
                "authors": p.authors,
                "abstract": p.abstract,
                "categories": p.categories,
                "published_date": p.published_date.isoformat() if p.published_date else None,
                "pdf_url": getattr(p, 'pdf_url', None),
                "citation_count": p.citation_count,
                "tldr": getattr(p, 'tldr', None),
            }
            for p in papers
        ],
    }
    
    with open(output, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _export_to_bibtex(papers: list, output: str):
    """Export papers to BibTeX format."""
    import re
    
    def escape_bibtex(text: str) -> str:
        """Escape special LaTeX/BibTeX characters."""
        if not text:
            return ""
        # Escape special characters
        text = text.replace("&", r"\&")
        text = text.replace("%", r"\%")
        text = text.replace("_", r"\_")
        text = text.replace("#", r"\#")
        text = text.replace("$", r"\$")
        return text
    
    def make_cite_key(paper_id: str, authors: list, year: int) -> str:
        """Generate a readable BibTeX citation key."""
        arxiv_id = paper_id.replace("arxiv:", "").replace(".", "_")
        if authors:
            # Get first author's last name
            first_author = authors[0].split()[-1].lower()
            first_author = re.sub(r'[^a-z]', '', first_author)
            return f"{first_author}{year}_{arxiv_id}"
        return f"arxiv_{arxiv_id}"
    
    with open(output, "w") as f:
        f.write("% ArXiv Agent Library Export\n")
        f.write(f"% Generated: {datetime.now().isoformat()}\n")
        f.write(f"% Papers: {len(papers)}\n\n")
        
        for paper in papers:
            arxiv_id = paper.id.replace("arxiv:", "")
            year = paper.published_date.year if paper.published_date else 2025
            cite_key = make_cite_key(paper.id, paper.authors, year)
            
            # Format authors in BibTeX style
            authors = " and ".join(paper.authors[:10])  # Limit to 10 authors
            
            f.write(f"@article{{{cite_key},\n")
            f.write(f"  title = {{{escape_bibtex(paper.title)}}},\n")
            f.write(f"  author = {{{escape_bibtex(authors)}}},\n")
            f.write(f"  year = {{{year}}},\n")
            f.write(f"  eprint = {{{arxiv_id}}},\n")
            f.write("  archivePrefix = {arXiv},\n")
            f.write("  primaryClass = {" + (paper.categories[0] if paper.categories else "cs.AI") + "},\n")
            
            if paper.abstract:
                # Truncate abstract for BibTeX
                abstract = paper.abstract[:500]
                if len(paper.abstract) > 500:
                    abstract += "..."
                f.write(f"  abstract = {{{escape_bibtex(abstract)}}},\n")
            
            f.write("}\n\n")


@app.command("import")
def import_library(
    input_file: str = typer.Argument(..., help="Input file to import"),
    format: str = typer.Option(None, "--format", "-f", help="Format: json, bibtex (auto-detected)"),
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="Import to collection"),
    skip_existing: bool = typer.Option(True, "--skip-existing/--replace", help="Skip or replace existing"),
):
    """📥 Import papers from file.
    
    Examples:
        arxiv-agent library import my_library.json
        arxiv-agent library import refs.bib
        arxiv-agent library import export.json --collection "Imported"
    """
    import os
    
    if not os.path.exists(input_file):
        console.print(f"[red]File not found: {input_file}[/]")
        raise typer.Exit(1)
    
    # Auto-detect format from extension
    if format is None:
        if input_file.endswith('.json'):
            format = 'json'
        elif input_file.endswith('.bib') or input_file.endswith('.bibtex'):
            format = 'bibtex'
        else:
            console.print("[red]Cannot detect format. Use --format json or --format bibtex[/]")
            raise typer.Exit(1)
    
    db = get_db()
    
    # Get or create collection if specified
    collection_id = None
    if collection:
        col = db.get_collection_by_name(collection)
        if not col:
            col = db.create_collection(collection, f"Imported from {input_file}")
            console.print(f"[dim]Created collection: {collection}[/]")
        collection_id = col.id
    
    if format == "json":
        imported, skipped = _import_from_json(input_file, db, collection_id, skip_existing)
    elif format == "bibtex":
        imported, skipped = _import_from_bibtex(input_file, db, collection_id, skip_existing)
    else:
        console.print(f"[red]Unknown format: {format}[/]")
        raise typer.Exit(1)
    
    console.print(f"[green]✓[/] Imported {imported} papers")
    if skipped > 0:
        console.print(f"[dim]Skipped {skipped} existing papers[/]")


def _import_from_json(input_file: str, db, collection_id: Optional[int], skip_existing: bool) -> tuple[int, int]:
    """Import papers from JSON file."""
    import json
    
    with open(input_file, "r") as f:
        data = json.load(f)
    
    papers_data = data.get("papers", [])
    if not papers_data:
        console.print("[yellow]No papers found in JSON file[/]")
        return 0, 0
    
    from arxiv_agent.data.models import Paper
    
    imported = 0
    skipped = 0
    
    with console.status("[dim]Importing papers...[/]") as status:
        for p in papers_data:
            paper_id = p.get("id", "")
            if not paper_id:
                continue
            
            # Ensure arxiv: prefix
            if not paper_id.startswith("arxiv:"):
                paper_id = f"arxiv:{paper_id}"
            
            # Check if exists
            existing = db.get_paper(paper_id)
            if existing:
                if skip_existing:
                    skipped += 1
                    continue
                else:
                    # Delete for replacement
                    db.delete_paper(paper_id)
            
            # Parse dates
            pub_date = None
            if p.get("published_date"):
                try:
                    pub_date = datetime.fromisoformat(p["published_date"].replace("Z", "+00:00"))
                except Exception:
                    pass
            
            upd_date = None
            if p.get("updated_date"):
                try:
                    upd_date = datetime.fromisoformat(p["updated_date"].replace("Z", "+00:00"))
                except Exception:
                    pass
            
            # Create paper model
            paper = Paper(
                id=paper_id,
                title=p.get("title", ""),
                authors=p.get("authors", []),
                abstract=p.get("abstract", ""),
                categories=p.get("categories", []),
                published_date=pub_date,
                updated_date=upd_date,
                pdf_url=p.get("pdf_url", ""),
                citation_count=p.get("citation_count", 0),
                read_status=p.get("read_status", "unread"),
                notes=p.get("notes", ""),
            )
            
            db.save_paper(paper)
            
            # Add to collection if specified
            if collection_id:
                db.add_paper_to_collection(paper_id, collection_id)
            
            imported += 1
            status.update(f"[dim]Importing... ({imported} done)[/]")
    
    return imported, skipped


def _import_from_bibtex(input_file: str, db, collection_id: Optional[int], skip_existing: bool) -> tuple[int, int]:
    """Import papers from BibTeX file."""
    import re
    
    with open(input_file, "r") as f:
        content = f.read()
    
    # Parse BibTeX entries
    entry_pattern = r'@\w+\{([^,]+),([^@]+)\}'
    entries = re.findall(entry_pattern, content, re.DOTALL)
    
    if not entries:
        console.print("[yellow]No BibTeX entries found[/]")
        return 0, 0
    
    from arxiv_agent.data.models import Paper
    
    imported = 0
    skipped = 0
    
    with console.status("[dim]Importing from BibTeX...[/]") as status:
        for cite_key, body in entries:
            # Parse fields
            fields = {}
            field_pattern = r'(\w+)\s*=\s*\{([^}]*)\}'
            for match in re.finditer(field_pattern, body):
                fields[match.group(1).lower()] = match.group(2)
            
            # Get arXiv ID from eprint field or cite key
            arxiv_id = fields.get("eprint", "")
            if not arxiv_id:
                # Try to extract from cite key
                arxiv_match = re.search(r'(\d{4}\.\d{4,5})', cite_key)
                if arxiv_match:
                    arxiv_id = arxiv_match.group(1)
            
            if not arxiv_id:
                continue  # Skip non-arXiv entries
            
            paper_id = f"arxiv:{arxiv_id}"
            
            # Check if exists
            existing = db.get_paper(paper_id)
            if existing:
                if skip_existing:
                    skipped += 1
                    continue
                else:
                    db.delete_paper(paper_id)
            
            # Parse authors
            authors = []
            if fields.get("author"):
                authors = [a.strip() for a in fields["author"].split(" and ")]
            
            # Parse year to date
            pub_date = None
            if fields.get("year"):
                try:
                    year = int(fields["year"])
                    pub_date = datetime(year, 1, 1)
                except Exception:
                    pass
            
            # Get categories from primaryClass
            categories = []
            if fields.get("primaryclass"):
                categories = [fields["primaryclass"]]
            
            paper = Paper(
                id=paper_id,
                title=fields.get("title", ""),
                authors=authors,
                abstract=fields.get("abstract", ""),
                categories=categories,
                published_date=pub_date,
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            )
            
            db.save_paper(paper)
            
            if collection_id:
                db.add_paper_to_collection(paper_id, collection_id)
            
            imported += 1
            status.update(f"[dim]Importing... ({imported} done)[/]")
    
    return imported, skipped


# Import datetime for export
from datetime import datetime

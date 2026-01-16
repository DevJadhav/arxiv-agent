"""Analyze commands for ArXiv Agent."""

import asyncio
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree

from arxiv_agent.agents.orchestrator import get_orchestrator
from arxiv_agent.config.settings import get_settings
from arxiv_agent.data.storage import get_db as get_storage


console = Console()
app = typer.Typer()


@app.command("paper")
def analyze(
    paper_id: str = typer.Argument(..., help="Paper ID to analyze"),
    depth: str = typer.Option("standard", "--depth", "-d", help="Analysis depth: quick, standard, full"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-analysis"),
    paper2code: bool = typer.Option(False, "--paper2code", "--code", help="Generate implementation plan"),
):
    """📊 Perform deep analysis on a research paper.
    
    Examples:
        arxiv-agent analyze 2401.12345
        arxiv-agent analyze 2401.12345 --depth full
        arxiv-agent analyze 2401.12345 --paper2code
    """
    if paper2code:
        asyncio.run(_paper2code(paper_id))
    else:
        asyncio.run(_analyze(paper_id, depth, force))


async def _analyze(paper_id: str, depth: str, force: bool):
    """Execute analysis."""
    orchestrator = get_orchestrator()
    
    # Normalize ID
    if not paper_id.startswith("arxiv:"):
        paper_id = f"arxiv:{paper_id}"
    
    options = {
        "depth": depth,
        "force": force,
    }
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Analyzing paper...", total=None)
        
        result = await orchestrator.run(
            task_type="analyze",
            paper_id=paper_id,
            options=options,
        )
    
    if result.errors:
        console.print(f"[red]Error:[/] {result.errors[0]}")
        raise typer.Exit(1)
    
    if not result.analysis:
        console.print("[yellow]No analysis generated.[/]")
        return
    
    paper = result.current_paper
    analysis = result.analysis
    
    # Display title
    console.print()
    if paper:
        console.print(f"[bold blue]📊 Analysis: {paper.title}[/]")
        console.print(f"[dim]{paper.id}[/]")
    console.print()
    
    # Display analysis
    analysis_text = analysis.get("text", "")
    console.print(Panel(
        Markdown(analysis_text),
        title=f"[bold]Analysis ({depth})[/]",
        border_style="green",
    ))
    
    # Show sections if available
    sections = analysis.get("sections", [])
    if sections:
        console.print("\n[bold]Paper Sections:[/]")
        for section in sections:
            console.print(f"  • {section['name']} (pages {section['pages']})")
    
    console.print()
    console.print("[dim]Actions:[/]")
    console.print(f"  • Chat: [cyan]arxiv-agent chat {paper_id}[/]")
    console.print(f"  • Generate code: [cyan]arxiv-agent analyze {paper_id} --paper2code[/]")


async def _paper2code(paper_id: str):
    """Generate paper-to-code implementation plan."""
    orchestrator = get_orchestrator()
    
    # Normalize ID
    if not paper_id.startswith("arxiv:"):
        paper_id = f"arxiv:{paper_id}"
    
    console.print()
    console.print(f"[bold blue]🔧 Paper-to-Code: {paper_id}[/]")
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Generating implementation plan...", total=None)
        
        result = await orchestrator.run(
            task_type="paper2code",
            paper_id=paper_id,
        )
    
    if result.errors:
        console.print(f"[red]Error:[/] {result.errors[0]}")
        raise typer.Exit(1)
    
    if not result.code_plan:
        console.print("[yellow]No implementation plan generated.[/]")
        return
    
    plan = result.code_plan
    
    # Display architecture
    console.print(Panel(
        plan.get("architecture", "No architecture description"),
        title="[bold]Architecture[/]",
        border_style="blue",
    ))
    
    # Display components as tree
    tree = Tree("📁 Project Structure")
    for component in plan.get("components", []):
        branch = tree.add(f"📄 [cyan]{component['file']}[/] - {component['name']}")
        branch.add(f"[dim]{component.get('description', '')}[/]")
        deps = component.get("dependencies", [])
        if deps:
            branch.add(f"[dim]Dependencies: {', '.join(deps)}[/]")
    
    console.print(tree)
    console.print()
    
    # Display dependencies
    deps = plan.get("python_dependencies", [])
    if deps:
        console.print(f"[bold]Python Dependencies:[/] {', '.join(deps)}")
    
    # Display implementation order
    order = plan.get("implementation_order", [])
    if order:
        console.print("\n[bold]Implementation Order:[/]")
        for i, step in enumerate(order, 1):
            console.print(f"  {i}. {step}")
    
    console.print()
    console.print("[dim]To generate full code: arxiv-agent paper2code <paper_id> --output ./implementation[/]")


def paper2code(
    paper_id: str = typer.Argument(..., help="Paper ID to analyze"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for generated code"),
):
    """🔧 Generate implementation code from a research paper.
    
    Analyzes a paper and generates a full implementation plan with code scaffolding.
    
    Examples:
        arxiv-agent paper2code 2401.12345
        arxiv-agent paper2code arxiv:2401.12345 --output ./implementation
    """
    asyncio.run(_paper2code(paper_id))
    
    if output:
        console.print(f"\n[dim]Output directory would be: {output}[/]")
        console.print("[dim]Full code generation coming in v0.2.0[/]")


def show_summary(
    paper_id: str = typer.Argument(..., help="Paper ID"),
):
    """📝 Show quick summary of a paper.
    
    Example:
        arxiv-agent summary 2401.12345
    """
    asyncio.run(_quick_summary(paper_id))


async def _quick_summary(paper_id: str):
    """Show quick summary."""
    from arxiv_agent.data.storage import get_db
    
    db = get_db()
    
    if not paper_id.startswith("arxiv:"):
        paper_id = f"arxiv:{paper_id}"
    
    paper = db.get_paper(paper_id)
    if not paper:
        console.print(f"[red]Paper not found: {paper_id}[/]")
        console.print("[dim]Use 'arxiv-agent search' to find and fetch papers first.[/]")
        raise typer.Exit(1)
    
    analyses = db.get_all_analyses(paper_id)
    
    console.print()
    console.print(f"[bold]{paper.title}[/]")
    console.print(f"[dim]{paper.id}[/]")
    console.print()
    
    if paper.tldr:
        console.print(f"[bold]TLDR:[/] {paper.tldr}")
        console.print()
    
    console.print(f"[bold]Abstract:[/]")
    console.print(f"[dim]{paper.abstract}[/]")
    
    if analyses:
        console.print(f"\n[bold]Available analyses:[/] {len(analyses)}")
        for a in analyses[:5]:
            console.print(f"  • {a.analysis_type} ({a.created_at.strftime('%Y-%m-%d')})")


@app.command("compare")
def compare_papers(
    paper_ids: List[str] = typer.Argument(..., help="Paper IDs to compare (minimum 2)"),
    format: str = typer.Option("markdown", "--format", "-f", help="Output format: json, markdown, html"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save to file"),
):
    """🔄 Compare multiple papers side by side.
    
    Examples:
        arxiv-agent analyze compare 2401.00001 2401.00002
        arxiv-agent analyze compare 2401.00001 2401.00002 --format json
        arxiv-agent analyze compare 2401.00001 2401.00002 -o comparison.md
    """
    if len(paper_ids) < 2:
        console.print("[red]Error:[/] At least 2 paper IDs required for comparison")
        raise typer.Exit(1)
    
    asyncio.run(_compare_papers(paper_ids, format, output))


async def _compare_papers(paper_ids: List[str], format: str, output: Optional[str]):
    """Execute paper comparison."""
    from arxiv_agent.agents.analyzer import AnalyzerAgent
    from arxiv_agent.data.storage import get_db
    
    db = get_db()
    
    # Normalize IDs and fetch papers
    papers = []
    for paper_id in paper_ids:
        if not paper_id.startswith("arxiv:"):
            paper_id = f"arxiv:{paper_id}"
        
        paper = db.get_paper(paper_id)
        if not paper:
            console.print(f"[red]Paper not found:[/] {paper_id}")
            console.print("[dim]Use 'arxiv-agent search' to find and fetch papers first[/]")
            raise typer.Exit(1)
        papers.append(paper)
    
    console.print()
    console.print("[bold blue]🔄 Comparing Papers[/]")
    console.print()
    
    for i, paper in enumerate(papers, 1):
        console.print(f"  {i}. {paper.title}")
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Analyzing papers for comparison...", total=None)
        
        analyzer = AnalyzerAgent()
        result = analyzer.compare_papers(*papers, output_format=format)
    
    # Display or save result
    if isinstance(result, str):
        if output:
            Path(output).write_text(result)
            console.print(f"[green]✓[/] Saved comparison to {output}")
        else:
            if format == "markdown":
                console.print(Panel(Markdown(result), title="Comparison", border_style="green"))
            else:
                console.print(result)
    else:
        import json
        json_str = json.dumps(result, indent=2)
        if output:
            Path(output).write_text(json_str)
            console.print(f"[green]✓[/] Saved comparison to {output}")
        else:
            console.print(json_str)


# Keep the original analyze as default command
def analyze_default(
    paper_id: str = typer.Argument(..., help="Paper ID to analyze"),
    depth: str = typer.Option("standard", "--depth", "-d", help="Analysis depth: quick, standard, full"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-analysis"),
    paper2code: bool = typer.Option(False, "--paper2code", "--code", help="Generate implementation plan"),
):
    """📊 Perform deep analysis on a research paper (default command)."""
    if paper2code:
        asyncio.run(_paper2code(paper_id))
    else:
        asyncio.run(_analyze(paper_id, depth, force))


# Register the callback for the default behavior
@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    paper_id: Optional[str] = typer.Argument(None, help="Paper ID to analyze"),
    depth: str = typer.Option("standard", "--depth", "-d", help="Analysis depth"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-analysis"),
    paper2code: bool = typer.Option(False, "--paper2code", "--code", help="Generate implementation plan"),
):
    """📊 Analyze research papers.
    
    Run without subcommand to analyze a paper directly.
    """
    if ctx.invoked_subcommand is None:
        if paper_id:
            if paper2code:
                asyncio.run(_paper2code(paper_id))
            else:
                asyncio.run(_analyze(paper_id, depth, force))
        else:
            console.print("Usage: arxiv analyze <paper_id> [OPTIONS]")
            console.print("\nSubcommands: paper, compare")
            console.print("\nUse --help for more information")
            raise typer.Exit(0)

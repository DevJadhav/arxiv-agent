"""Chat commands for ArXiv Agent."""

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from arxiv_agent.agents.orchestrator import get_orchestrator
from arxiv_agent.config.settings import get_settings
from arxiv_agent.data.storage import get_db

console = Console()

def chat(
    paper_id: str = typer.Argument(..., help="Paper ID to chat about"),
    question: Optional[str] = typer.Option(None, "--ask", "-q", help="Ask a single question and exit"),
):
    """💬 Start an interactive RAG chat session or ask a question.
    
    Examples:
        arxiv-agent chat 2401.12345
        arxiv-agent chat 2401.12345 --ask "What is the main contribution?"
    """
    if question:
        asyncio.run(_single_question(paper_id, question))
    else:
        asyncio.run(_chat_session(paper_id))


async def _chat_session(paper_id: str):
    """Run interactive chat session."""
    # Normalize ID
    if not paper_id.startswith("arxiv:"):
        paper_id = f"arxiv:{paper_id}"
    
    orchestrator = get_orchestrator()
    settings = get_settings()
    db = get_db()
    
    # Get paper info
    paper = db.get_paper(paper_id)
    if not paper:
        # Try to fetch it
        console.print(f"[dim]Fetching paper {paper_id}...[/]")
        result = await orchestrator.run(task_type="search", paper_id=paper_id)
        if result.errors or not result.current_paper:
            console.print(f"[red]Paper not found: {paper_id}[/]")
            console.print("[dim]Use 'arxiv-agent search' to find papers first[/]")
            raise typer.Exit(1)
        paper = result.current_paper
    
    # Display header
    console.print()
    console.print(Panel(
        f"[bold]{paper.title}[/]\n\n[dim]Type your questions below. Use /help for commands, /quit to exit.[/]",
        title="💬 Chat Session",
        border_style="green",
    ))
    
    # Setup prompt with history
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        
        history_file = settings.data_dir / "chat_history.txt"
        session = PromptSession(history=FileHistory(str(history_file)))
        
        use_prompt_toolkit = True
    except ImportError:
        use_prompt_toolkit = False
    
    while True:
        try:
            # Get user input
            if use_prompt_toolkit:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: session.prompt("\n[You] > "),
                )
            else:
                user_input = input("\n[You] > ")
            
            if not user_input.strip():
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower().strip()
                
                if cmd in ["/quit", "/exit", "/q"]:
                    console.print("[dim]Goodbye![/]")
                    break
                    
                elif cmd == "/help":
                    console.print("""
[bold]Commands:[/]
  /quit, /exit, /q - Exit chat
  /help - Show this help
  /history - Show recent chat history
  /sources - Show sources from last response
  /clear - Start fresh session
""")
                    continue
                    
                elif cmd == "/history":
                    _show_recent_history(paper_id)
                    continue
                    
                elif cmd == "/clear":
                    chat_session = db.get_or_create_chat_session(paper_id)
                    db.clear_chat_session(chat_session.id)
                    console.print("[dim]Session cleared.[/]")
                    continue
                    
                else:
                    console.print(f"[yellow]Unknown command: {cmd}. Use /help for available commands.[/]")
                    continue
            
            # Process query
            console.print()
            
            result = await orchestrator.run(
                task_type="chat",
                paper_id=paper_id,
                query=user_input,
            )
            
            if result.errors:
                console.print(f"[red]Error: {result.errors[0]}[/]")
                continue
            
            # Display response
            if result.chat_history:
                response = result.chat_history[-1]["content"]
                console.print(Panel(
                    Markdown(response),
                    title="[bold blue]Claude[/]",
                    border_style="blue",
                ))
                
        except KeyboardInterrupt:
            console.print("\n[dim]Use /quit to exit[/]")
        except EOFError:
            break


def _show_recent_history(paper_id: str):
    """Show recent chat history for paper."""
    db = get_db()
    session = db.get_or_create_chat_session(paper_id)
    messages = db.get_chat_history(session.id, limit=10)
    
    if not messages:
        console.print("[dim]No chat history yet.[/]")
        return
    
    console.print("\n[bold]Recent History:[/]")
    for msg in messages[-6:]:
        role = "[green]You[/]" if msg.role == "user" else "[blue]Claude[/]"
        content = msg.content[:150]
        if len(msg.content) > 150:
            content += "..."
        console.print(f"  {role}: {content}")


def show_full_history(
    paper_id: str = typer.Argument(..., help="Paper ID"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of messages"),
):
    """📜 Show chat history for a paper.
    
    Example:
        arxiv-agent chat history 2401.12345
    """
    if not paper_id.startswith("arxiv:"):
        paper_id = f"arxiv:{paper_id}"
    
    db = get_db()
    session = db.get_or_create_chat_session(paper_id)
    messages = db.get_chat_history(session.id, limit=limit)
    
    if not messages:
        console.print("[dim]No chat history for this paper.[/]")
        return
    
    console.print(f"\n[bold]Chat History ({len(messages)} messages)[/]\n")
    
    for msg in messages:
        role_display = "[bold green]You[/]" if msg.role == "user" else "[bold blue]Claude[/]"
        time_str = msg.created_at.strftime("%H:%M")
        console.print(f"{role_display} [{time_str}]:")
        console.print(f"  {msg.content}\n")


def export_chat(
    paper_id: str = typer.Argument(..., help="Paper ID"),
    output: str = typer.Option("chat_export.md", "--output", "-o", help="Output file"),
):
    """📤 Export chat session to markdown.
    
    Example:
        arxiv-agent chat export 2401.12345 -o my_chat.md
    """
    if not paper_id.startswith("arxiv:"):
        paper_id = f"arxiv:{paper_id}"
    
    db = get_db()
    paper = db.get_paper(paper_id)
    session = db.get_or_create_chat_session(paper_id)
    messages = db.get_chat_history(session.id, limit=1000)
    
    if not messages:
        console.print("[red]No chat history to export.[/]")
        raise typer.Exit(1)
    
    title = paper.title if paper else paper_id
    
    with open(output, "w") as f:
        f.write(f"# Chat Export: {title}\n\n")
        f.write(f"Paper ID: {paper_id}\n")
        f.write(f"Session: {session.id}\n")
        f.write(f"Exported: {messages[-1].created_at.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("---\n\n")
        
        for msg in messages:
            role = "**You**" if msg.role == "user" else "**Claude**"
            f.write(f"{role}:\n\n{msg.content}\n\n---\n\n")
    
    console.print(f"[green]✓[/] Exported {len(messages)} messages to {output}")


def single_question(
    paper_id: str = typer.Argument(..., help="Paper ID"),
    question: str = typer.Argument(..., help="Question to ask"),
):
    """❓ Ask a single question about a paper (non-interactive).
    
    Example:
        arxiv-agent chat ask 2401.12345 "What is the main contribution?"
    """
    asyncio.run(_single_question(paper_id, question))


async def _single_question(paper_id: str, question: str):
    """Ask a single question."""
    if not paper_id.startswith("arxiv:"):
        paper_id = f"arxiv:{paper_id}"
    
    orchestrator = get_orchestrator()
    
    result = await orchestrator.run(
        task_type="chat",
        paper_id=paper_id,
        query=question,
    )
    
    if result.errors:
        console.print(f"[red]Error: {result.errors[0]}[/]")
        raise typer.Exit(1)
    
    if result.chat_history:
        response = result.chat_history[-1]["content"]
        console.print()
        console.print(Markdown(response))

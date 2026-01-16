"""Chat commands for ArXiv Agent."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from arxiv_agent.agents.orchestrator import get_orchestrator
from arxiv_agent.agents.rag_chat import RAGChatAgent
from arxiv_agent.config.settings import get_settings
from arxiv_agent.data.storage import get_db
# Alias for test mocking compatibility
get_storage = get_db

console = Console()
app = typer.Typer()


@app.command("start")
def chat(
    paper_id: str = typer.Argument(..., help="Paper ID to chat about"),
    question: Optional[str] = typer.Option(None, "--ask", "-q", help="Ask a single question and exit"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Enable streaming responses"),
):
    """💬 Start an interactive RAG chat session or ask a question.
    
    Examples:
        arxiv-agent chat start 2401.12345
        arxiv-agent chat start 2401.12345 --ask "What is the main contribution?"
        arxiv-agent chat 2401.12345 --no-stream
    """
    if question:
        asyncio.run(_single_question(paper_id, question, stream=stream))
    else:
        asyncio.run(_chat_session(paper_id, stream=stream))


async def _chat_session(paper_id: str, stream: bool = True):
    """Run interactive chat session."""
    # Normalize ID
    if not paper_id.startswith("arxiv:"):
        paper_id = f"arxiv:{paper_id}"
    
    orchestrator = get_orchestrator()
    settings = get_settings()
    db = get_db()
    
    # Initialize RAG chat agent for streaming
    rag_agent = RAGChatAgent() if stream else None
    
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
            
            if stream and rag_agent:
                # Stream the response for better UX
                await _stream_chat_response(rag_agent, paper_id, user_input)
            else:
                # Use non-streaming orchestrator
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


async def _stream_chat_response(rag_agent: RAGChatAgent, paper_id: str, query: str):
    """Stream chat response with live display.
    
    Uses Rich's Live display to show response chunks as they arrive,
    providing a smooth real-time typing effect.
    """
    full_response = ""
    
    # Show thinking indicator
    with console.status("[bold blue]Thinking...[/]", spinner="dots"):
        # Get first chunk to start display
        stream = rag_agent.stream_response(paper_id, query)
        try:
            first_chunk = await stream.__anext__()
            full_response = first_chunk
        except StopAsyncIteration:
            console.print("[yellow]No response generated.[/]")
            return
    
    # Stream remaining chunks with live display
    console.print(Panel.fit(
        "[bold blue]Claude[/]",
        border_style="blue",
    ))
    
    # Print chunks as they arrive
    console.print(first_chunk, end="")
    
    async for chunk in stream:
        full_response += chunk
        console.print(chunk, end="")
    
    # Final newline and separator
    console.print("\n")


async def _stream_single_question(rag_agent: RAGChatAgent, paper_id: str, question: str):
    """Stream a single question response.
    
    Like _stream_chat_response but for single question mode.
    """
    full_response = ""
    
    with console.status("[bold blue]Thinking...[/]", spinner="dots"):
        stream = rag_agent.stream_response(paper_id, question)
        try:
            first_chunk = await stream.__anext__()
            full_response = first_chunk
        except StopAsyncIteration:
            console.print("[yellow]No response generated.[/]")
            return
    
    console.print()
    console.print(first_chunk, end="")
    
    async for chunk in stream:
        full_response += chunk
        console.print(chunk, end="")
    
    console.print("\n")


@app.command("history")
def show_full_history(
    paper_id: Optional[str] = typer.Argument(None, help="Paper ID (optional, shows all if omitted)"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of messages/sessions"),
):
    """📜 Show chat history.
    
    Examples:
        arxiv-agent chat history
        arxiv-agent chat history 2401.12345
        arxiv-agent chat history --limit 50
    """
    db = get_db()
    
    if paper_id:
        # Show messages for specific paper
        if not paper_id.startswith("arxiv:"):
            paper_id = f"arxiv:{paper_id}"
        
        session = db.get_or_create_chat_session(paper_id)
        messages = db.get_chat_history(session.id, limit=limit)
        
        if not messages:
            console.print("[dim]No chat history for this paper.[/]")
            return
        
        console.print(f"\n[bold]Chat History ({len(messages)} messages)[/]\n")
        
        for msg in messages:
            role_display = "[bold green]You[/]" if msg.role == "user" else "[bold blue]Claude[/]"
            time_str = msg.created_at.strftime("%H:%M") if msg.created_at else ""
            console.print(f"{role_display} [{time_str}]:")
            console.print(f"  {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}\n")
    else:
        # List all chat sessions
        sessions = db.list_chat_sessions(limit=limit)
        
        if not sessions:
            console.print("[dim]No chat sessions found.[/]")
            return
        
        console.print("\n[bold]Chat Sessions[/]\n")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Paper ID", style="cyan")
        table.add_column("Messages", justify="right")
        table.add_column("Last Active")
        table.add_column("Session ID", style="dim")
        
        for session in sessions:
            paper_id_display = session.paper_id or "N/A"
            if len(paper_id_display) > 25:
                paper_id_display = paper_id_display[:22] + "..."
            
            last_active = session.last_active.strftime("%Y-%m-%d %H:%M") if session.last_active else "N/A"
            
            table.add_row(
                paper_id_display,
                str(session.message_count),
                last_active,
                str(session.id)[:8] + "..."
            )
        
        console.print(table)
        console.print("\n[dim]Use 'arxiv-agent chat history <paper_id>' for session details[/]")


@app.command("export")
def export_chat(
    paper_id: str = typer.Argument(..., help="Paper ID"),
    output: str = typer.Option("chat_export.md", "--output", "-o", help="Output file"),
    format: str = typer.Option("markdown", "--format", "-f", help="Format: markdown or json"),
):
    """📤 Export chat session to file.
    
    Examples:
        arxiv-agent chat export 2401.12345
        arxiv-agent chat export 2401.12345 -o my_chat.md
        arxiv-agent chat export 2401.12345 --format json -o chat.json
    """
    if not paper_id.startswith("arxiv:"):
        paper_id = f"arxiv:{paper_id}"
    
    db = get_db()
    session = db.get_or_create_chat_session(paper_id)
    
    messages = db.get_chat_history(session.id, limit=1000)
    if not messages:
        console.print("[red]No chat history to export.[/]")
        raise typer.Exit(1)
    
    try:
        result = db.export_chat_session(session.id, format=format)
        
        if format == "json":
            import json
            output_content = json.dumps(result, indent=2)
        else:
            output_content = result
        
        Path(output).write_text(output_content)
        console.print(f"[green]✓[/] Exported {len(messages)} messages to {output}")
        
    except Exception as e:
        console.print(f"[red]Export failed:[/] {e}")
        raise typer.Exit(1)


@app.command("ask")
def single_question(
    paper_id: str = typer.Argument(..., help="Paper ID"),
    question: str = typer.Argument(..., help="Question to ask"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Enable streaming responses"),
):
    """❓ Ask a single question about a paper (non-interactive).
    
    Examples:
        arxiv-agent chat ask 2401.12345 "What is the main contribution?"
        arxiv-agent chat ask 2401.12345 "Explain methodology" --no-stream
    """
    asyncio.run(_single_question(paper_id, question, stream=stream))


async def _single_question(paper_id: str, question: str, stream: bool = True):
    """Ask a single question."""
    if not paper_id.startswith("arxiv:"):
        paper_id = f"arxiv:{paper_id}"
    
    if stream:
        # Use streaming RAG agent
        rag_agent = RAGChatAgent()
        await _stream_single_question(rag_agent, paper_id, question)
    else:
        # Use non-streaming orchestrator
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


# Default callback for backward compatibility
@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    paper_id: Optional[str] = typer.Argument(None, help="Paper ID to chat about"),
    question: Optional[str] = typer.Option(None, "--ask", "-q", help="Ask a single question"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Enable streaming"),
):
    """💬 Chat with research papers using RAG.
    
    Run without subcommand to start a chat session directly.
    """
    if ctx.invoked_subcommand is None:
        if paper_id:
            if question:
                asyncio.run(_single_question(paper_id, question, stream=stream))
            else:
                asyncio.run(_chat_session(paper_id, stream=stream))
        else:
            console.print("Usage: arxiv chat <paper_id> [OPTIONS]")
            console.print("\nSubcommands: start, history, export, ask")
            console.print("\nUse --help for more information")

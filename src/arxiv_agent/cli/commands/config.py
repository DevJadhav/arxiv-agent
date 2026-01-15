"""Config commands for ArXiv Agent with multi-provider support."""

import os
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from arxiv_agent.config.settings import get_settings, LLMProviderType
from arxiv_agent.config.keys import get_key_storage, PROVIDERS

app = typer.Typer()
console = Console()

# Sub-app for provider management
provider_app = typer.Typer(help="🔌 Manage LLM providers")
app.add_typer(provider_app, name="provider")


@provider_app.command("list")
def list_providers():
    """📋 List available LLM providers and their status.
    
    Example:
        arxiv-agent config provider list
    """
    settings = get_settings()
    key_storage = get_key_storage()
    configured = key_storage.list_configured()
    
    console.print("\n[bold blue]🔌 LLM Providers[/]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Provider", style="cyan")
    table.add_column("Status")
    table.add_column("Default Model")
    table.add_column("Active", justify="center")
    
    providers_info = [
        ("anthropic", settings.llm.anthropic_model),
        ("openai", settings.llm.openai_model),
        ("gemini", settings.llm.gemini_model),
    ]
    
    for provider, model in providers_info:
        status = "[green]✓ Configured[/]" if configured.get(provider) else "[red]✗ Not configured[/]"
        is_active = "⭐" if provider == settings.llm.default_provider else ""
        table.add_row(provider.title(), status, model, is_active)
    
    console.print(table)
    console.print(f"\n[dim]Default provider: [cyan]{settings.llm.default_provider}[/][/]")
    console.print("[dim]Use 'arxiv-agent config provider set <name>' to change default[/]")


@provider_app.command("set")
def set_default_provider(
    provider: str = typer.Argument(..., help="Provider name: anthropic, openai, or gemini"),
):
    """⭐ Set the default LLM provider.
    
    Example:
        arxiv-agent config provider set openai
    """
    provider = provider.lower()
    valid_providers = ["anthropic", "openai", "gemini"]
    
    if provider not in valid_providers:
        console.print(f"[red]Invalid provider: {provider}[/]")
        console.print(f"[dim]Valid providers: {', '.join(valid_providers)}[/]")
        raise typer.Exit(1)
    
    settings = get_settings()
    key_storage = get_key_storage()
    
    # Check if API key is configured
    if not key_storage.get_key(provider):
        console.print(f"[yellow]Warning:[/] No API key configured for {provider}")
        console.print(f"[dim]Run 'arxiv-agent config provider setup {provider}' first[/]")
        if not typer.confirm("Set as default anyway?"):
            raise typer.Abort()
    
    # Update settings (runtime only - persisted via env var or config file)
    settings.llm.default_provider = provider
    
    console.print(f"[green]✓[/] Default provider set to: [cyan]{provider}[/]")
    console.print(f"\n[dim]To persist, add to your shell profile:[/]")
    console.print(f"  export ARXIV_AGENT_LLM__DEFAULT_PROVIDER={provider}")


@provider_app.command("setup")
def setup_provider(
    provider: str = typer.Argument(..., help="Provider to setup: anthropic, openai, or gemini"),
):
    """🔧 Interactive setup for a provider (add API key).
    
    Example:
        arxiv-agent config provider setup anthropic
    """
    provider = provider.lower()
    valid_providers = ["anthropic", "openai", "gemini"]
    
    if provider not in valid_providers:
        console.print(f"[red]Invalid provider: {provider}[/]")
        console.print(f"[dim]Valid providers: {', '.join(valid_providers)}[/]")
        raise typer.Exit(1)
    
    key_storage = get_key_storage()
    
    # Show provider-specific info
    info = {
        "anthropic": "Get your API key from https://console.anthropic.com/",
        "openai": "Get your API key from https://platform.openai.com/api-keys",
        "gemini": "Get your API key from https://aistudio.google.com/apikey",
    }
    
    console.print(f"\n[bold blue]🔧 Setup {provider.title()}[/]\n")
    console.print(f"[dim]{info[provider]}[/]\n")
    
    # Check existing
    existing = key_storage.get_masked_key(provider)
    if existing:
        console.print(f"Current key: [cyan]{existing}[/]")
        if not typer.confirm("Replace existing key?"):
            raise typer.Abort()
    
    # Get new key
    api_key = typer.prompt("Enter API key", hide_input=True)
    
    if not api_key.strip():
        console.print("[red]API key cannot be empty[/]")
        raise typer.Exit(1)
    
    # Store securely
    key_storage.set_key(provider, api_key.strip())
    
    console.print(f"\n[green]✓[/] API key stored securely for {provider}")
    console.print(f"[dim]Key: {key_storage.get_masked_key(provider)}[/]")
    
    # Offer to set as default
    settings = get_settings()
    if settings.llm.default_provider != provider:
        if typer.confirm(f"\nSet {provider} as default provider?"):
            settings.llm.default_provider = provider
            console.print(f"[green]✓[/] {provider} is now the default provider")


@provider_app.command("remove")
def remove_provider_key(
    provider: str = typer.Argument(..., help="Provider to remove key for"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """🗑️ Remove API key for a provider.
    
    Example:
        arxiv-agent config provider remove openai
    """
    provider = provider.lower()
    key_storage = get_key_storage()
    
    if not key_storage.get_key(provider):
        console.print(f"[yellow]No API key configured for {provider}[/]")
        return
    
    if not force:
        if not typer.confirm(f"Remove API key for {provider}?"):
            raise typer.Abort()
    
    key_storage.delete_key(provider)
    console.print(f"[green]✓[/] Removed API key for {provider}")


# Sub-app for model configuration
models_app = typer.Typer(help="🤖 Configure models for agents")
app.add_typer(models_app, name="models")


@models_app.command("show")
def show_model_config():
    """📋 Show current model configuration for all agents.
    
    Example:
        arxiv-agent config models show
    """
    settings = get_settings()
    
    console.print("\n[bold blue]🤖 Agent Model Configuration[/]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Agent", style="cyan")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Override", justify="center")
    
    agents = ["analyzer", "chat", "code", "digest"]
    
    for agent in agents:
        provider, model = settings.llm.get_agent_config(agent)
        
        # Check if using override
        config = getattr(settings.llm, f"{agent}_config")
        has_override = config.provider is not None or config.model is not None
        override_marker = "⚙️" if has_override else ""
        
        table.add_row(agent.title(), provider, model, override_marker)
    
    console.print(table)
    console.print(f"\n[dim]Default provider: [cyan]{settings.llm.default_provider}[/][/]")
    console.print("[dim]⚙️ = custom configuration[/]")


@models_app.command("set")
def set_agent_model(
    agent: Optional[str] = typer.Argument(None, help="Agent name: analyzer, chat, code, digest"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Provider override"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model override"),
):
    """⚙️ Set custom model for a specific agent.
    
    Examples:
        arxiv-agent config models set code --provider openai --model gpt-4o
        arxiv-agent config models set analyzer --model claude-opus-4-5-20250101
    """
    valid_agents = ["analyzer", "chat", "code", "digest"]
    
    if not agent:
        import questionary
        agent = questionary.select(
            "Select agent to configure:",
            choices=valid_agents
        ).ask()
        
        if not agent:
            raise typer.Abort()
        
    agent = agent.lower()
    
    if agent not in valid_agents:
        console.print(f"[red]Invalid agent: {agent}[/]")
        console.print(f"[dim]Valid agents: {', '.join(valid_agents)}[/]")
        raise typer.Exit(1)
    
    settings = get_settings()
    config = getattr(settings.llm, f"{agent}_config")
    
    # Provider selection
    if not provider:
        # Defaults to current or global default
        current_provider = config.provider or settings.llm.default_provider
        
        # Interactive provider selection if not provided
        if not provider and not model:
            import questionary
            provider = questionary.select(
                "Select provider:",
                choices=["anthropic", "openai", "gemini"],
                default=current_provider
            ).ask()
            
            if not provider:
                raise typer.Abort()
        else:
            provider = current_provider

    if provider:
         valid_providers = ["anthropic", "openai", "gemini"]
         if provider not in valid_providers:
             console.print(f"[red]Invalid provider: {provider}[/]")
             raise typer.Exit(1)
         config.provider = provider

    # Model selection
    if not model:
        # If model not provided, prompt for it
        from arxiv_agent.core.llm_service import get_llm_service
        
        try:
            with console.status(f"Fetching available models for {provider}..."):
                # Create temp service to fetch models
                llm = get_llm_service(provider=provider)
                available_models = llm.list_models()
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to fetch models: {e}[/]")
            available_models = []

        if available_models:
             import questionary
             # Use current model as default if valid
             current_model = config.model or settings.llm.get_provider_model(provider)
             default_model = current_model if current_model in available_models else available_models[0]
             
             model = questionary.select(
                 f"Select model for {provider}:",
                 choices=available_models,
                 default=default_model
             ).ask()
             
             if not model:
                 raise typer.Abort()
        else:
             # Fallback to manual entry if list failed
             model = typer.prompt("Enter model name (listing failed)")

    if model:
        config.model = model
    
    console.print(f"[green]✓[/] Updated {agent} agent configuration:")
    new_provider, new_model = settings.llm.get_agent_config(agent)
    console.print(f"  Provider: [cyan]{new_provider}[/]")
    console.print(f"  Model: [cyan]{new_model}[/]")
    
    console.print(f"\n[dim]To persist, add to your shell profile:[/]")
    console.print(f"  export ARXIV_AGENT_LLM__{agent.upper()}_CONFIG__PROVIDER={new_provider}")
    console.print(f"  export ARXIV_AGENT_LLM__{agent.upper()}_CONFIG__MODEL={new_model}")


@models_app.command("reset")
def reset_agent_model(
    agent: str = typer.Argument(..., help="Agent to reset to defaults"),
):
    """🔄 Reset agent to use default provider/model.
    
    Example:
        arxiv-agent config models reset code
    """
    agent = agent.lower()
    valid_agents = ["analyzer", "chat", "code", "digest"]
    
    if agent not in valid_agents:
        console.print(f"[red]Invalid agent: {agent}[/]")
        raise typer.Exit(1)
    
    settings = get_settings()
    config = getattr(settings.llm, f"{agent}_config")
    config.provider = None
    config.model = None
    
    console.print(f"[green]✓[/] Reset {agent} agent to defaults")
    provider, model = settings.llm.get_agent_config(agent)
    console.print(f"  Now using: [cyan]{provider}[/] / [cyan]{model}[/]")


@app.command("show")
def show_config():
    """📋 Show current configuration.
    
    Example:
        arxiv-agent config show
    """
    settings = get_settings()
    key_storage = get_key_storage()
    configured = key_storage.list_configured()
    
    console.print("\n[bold blue]⚙️ ArXiv Agent Configuration[/]\n")
    
    # Paths
    console.print("[bold]Paths:[/]")
    console.print(f"  Data: {settings.data_dir}")
    console.print(f"  Cache: {settings.cache_dir}")
    console.print(f"  Config: {settings.config_dir}")
    
    # LLM
    console.print("\n[bold]LLM Settings:[/]")
    console.print(f"  Default Provider: [cyan]{settings.llm.default_provider}[/]")
    console.print(f"  Temperature: {settings.llm.temperature}")
    console.print(f"  Max Tokens: {settings.llm.max_tokens}")
    
    # API Keys
    console.print("\n[bold]API Keys:[/]")
    for provider in ["anthropic", "openai", "gemini", "semantic_scholar"]:
        masked = key_storage.get_masked_key(provider)
        if masked:
            console.print(f"  {provider.title():18} [green]✓[/] {masked}")
        else:
            console.print(f"  {provider.title():18} [dim]✗ Not configured[/]")
    
    # Digest
    console.print("\n[bold]Digest Settings:[/]")
    console.print(f"  Enabled: {settings.digest.enabled}")
    console.print(f"  Schedule: {settings.digest.schedule_time}")
    console.print(f"  Categories: {', '.join(settings.digest.categories)}")


@app.command("paths")
def show_paths():
    """📂 Show all data paths.
    
    Example:
        arxiv-agent config paths
    """
    settings = get_settings()
    
    table = Table(title="📂 Data Paths", show_header=True)
    table.add_column("Type", style="cyan")
    table.add_column("Path")
    table.add_column("Exists", style="green")
    
    paths = [
        ("Data Directory", settings.data_dir),
        ("Database", settings.db_path),
        ("Vector Store", settings.vector_db_path),
        ("Cache Directory", settings.cache_dir),
        ("Config Directory", settings.config_dir),
        ("PDFs", settings.data_dir / "pdfs"),
        ("Digests", settings.data_dir / "digests"),
    ]
    
    for name, path in paths:
        exists = "✓" if path.exists() else "✗"
        table.add_row(name, str(path), exists)
    
    console.print()
    console.print(table)


@app.command("reset")
def reset_config(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """🔄 Reset configuration to defaults.
    
    Example:
        arxiv-agent config reset --yes
    """
    if not confirm:
        if not typer.confirm("This will reset all settings to defaults. Continue?"):
            raise typer.Abort()
    
    from arxiv_agent.config.settings import reset_settings
    reset_settings()
    
    console.print("[green]✓[/] Configuration reset to defaults")

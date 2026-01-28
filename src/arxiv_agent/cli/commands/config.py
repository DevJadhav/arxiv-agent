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

# Sub-app for theme management
theme_app = typer.Typer(help="🎨 Manage UI themes")
app.add_typer(theme_app, name="theme")


@theme_app.command("list")
def list_themes():
    """📋 List available themes.
    
    Example:
        arxiv-agent config theme list
    """
    from arxiv_agent.config.themes import get_available_themes, get_current_theme, get_theme
    
    current = get_current_theme()
    themes = get_available_themes()
    
    console.print("\n[bold blue]🎨 Available Themes[/]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Theme", style="cyan")
    table.add_column("Description")
    table.add_column("Active", justify="center")
    
    for name in themes:
        theme = get_theme(name)
        is_current = "⭐" if name == current.name else ""
        table.add_row(theme.name, theme.description, is_current)
    
    console.print(table)
    console.print(f"\n[dim]Current theme: [cyan]{current.name}[/][/]")


@theme_app.command("set")
def set_theme(
    name: str = typer.Argument(..., help="Theme name to activate"),
):
    """⭐ Set the active theme.
    
    Example:
        arxiv-agent config theme set dark
    """
    from arxiv_agent.config.themes import get_theme_manager, get_available_themes
    
    manager = get_theme_manager()
    available = get_available_themes()
    
    if name not in available:
        console.print(f"[red]Theme not found: {name}[/]")
        console.print(f"[dim]Available themes: {', '.join(available)}[/]")
        raise typer.Exit(1)
    
    manager.set_theme(name)
    console.print(f"[green]✓[/] Theme set to: [cyan]{name}[/]")


@theme_app.command("preview")
def preview_theme(
    name: str = typer.Argument(..., help="Theme name to preview"),
):
    """👁️ Preview a theme without applying.
    
    Example:
        arxiv-agent config theme preview dracula
    """
    from arxiv_agent.config.themes import preview_theme as do_preview, get_available_themes
    
    available = get_available_themes()
    
    if name not in available:
        console.print(f"[red]Theme not found: {name}[/]")
        console.print(f"[dim]Available themes: {', '.join(available)}[/]")
        raise typer.Exit(1)
    
    preview = do_preview(name)
    console.print(Panel(preview, title=f"Theme Preview: {name}", border_style="blue"))


@theme_app.command("reset")
def reset_theme():
    """🔄 Reset to default theme.
    
    Example:
        arxiv-agent config theme reset
    """
    from arxiv_agent.config.themes import get_theme_manager
    
    manager = get_theme_manager()
    manager.set_theme("default")
    console.print("[green]✓[/] Theme reset to default")


@app.command("path")
def show_paths():
    """📁 Show configuration file paths.
    
    Example:
        arxiv-agent config path
    """
    settings = get_settings()
    
    console.print("\n[bold blue]📁 Configuration Paths[/]\n")
    
    console.print(f"[bold]Data Directory:[/]")
    console.print(f"  {settings.data_dir}")
    
    console.print(f"\n[bold]Database:[/]")
    console.print(f"  {settings.db_path}")
    
    console.print(f"\n[bold]Vector Store:[/]")
    console.print(f"  {settings.vector_db_path}")
    
    console.print(f"\n[bold]Cache Directory:[/]")
    console.print(f"  {settings.data_dir / 'cache'}")
    
    config_file = settings.data_dir / "config.json"
    console.print(f"\n[bold]Config File:[/]")
    if config_file.exists():
        console.print(f"  {config_file} [green](exists)[/]")
    else:
        console.print(f"  {config_file} [dim](not created)[/]")


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
        ("ollama", settings.llm.ollama_model),
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
    valid_providers = ["anthropic", "openai", "gemini", "ollama"]
    
    if provider not in valid_providers:
        console.print(f"[red]Invalid provider: {provider}[/]")
        console.print(f"[dim]Valid providers: {', '.join(valid_providers)}[/]")
        raise typer.Exit(1)
    
    settings = get_settings()
    key_storage = get_key_storage()
    
    # Check if API key is configured (Ollama doesn't require an API key)
    if provider != "ollama" and not key_storage.get_key(provider):
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
    valid_providers = ["anthropic", "openai", "gemini", "ollama"]
    
    if provider not in valid_providers:
        console.print(f"[red]Invalid provider: {provider}[/]")
        console.print(f"[dim]Valid providers: {', '.join(valid_providers)}[/]")
        raise typer.Exit(1)
    
    key_storage = get_key_storage()
    settings = get_settings()
    
    # Show provider-specific info
    info = {
        "anthropic": "Get your API key from https://console.anthropic.com/",
        "openai": "Get your API key from https://platform.openai.com/api-keys",
        "gemini": "Get your API key from https://aistudio.google.com/apikey",
        "ollama": "Ollama runs locally and doesn't require an API key. Configure the base URL if needed.",
    }
    
    console.print(f"\n[bold blue]🔧 Setup {provider.title()}[/]\n")
    console.print(f"[dim]{info[provider]}[/]\n")
    
    # Ollama setup is different - configure base URL instead of API key
    if provider == "ollama":
        current_url = settings.llm.ollama_base_url
        console.print(f"Current base URL: [cyan]{current_url}[/]")
        
        new_url = typer.prompt(
            "Enter Ollama base URL",
            default=current_url,
            show_default=True
        )
        
        if new_url != current_url:
            settings.llm.ollama_base_url = new_url
            console.print(f"\n[green]✓[/] Ollama base URL updated to: [cyan]{new_url}[/]")
        else:
            console.print(f"\n[green]✓[/] Ollama base URL unchanged: [cyan]{current_url}[/]")
        
        # Test connection by listing models
        console.print("\n[dim]Testing connection to Ollama server...[/]")
        try:
            from arxiv_agent.core.llm_service import get_llm_service
            llm = get_llm_service(provider="ollama")
            models = llm.list_models()
            if models:
                console.print(f"[green]✓[/] Connected! Found {len(models)} model(s): {', '.join(models[:5])}")
                if len(models) > 5:
                    console.print(f"  [dim]...and {len(models) - 5} more[/]")
            else:
                console.print("[yellow]Warning:[/] No models found. Pull a model with: ollama pull llama3.2")
        except Exception as e:
            console.print(f"[yellow]Warning:[/] Could not connect to Ollama: {e}")
            console.print("[dim]Make sure Ollama is running: ollama serve[/]")
        
        # Offer to set as default
        if settings.llm.default_provider != provider:
            if typer.confirm(f"\nSet {provider} as default provider?"):
                settings.llm.default_provider = provider
                console.print(f"[green]✓[/] {provider} is now the default provider")
        
        console.print(f"\n[dim]To persist the base URL, add to your shell profile:[/]")
        console.print(f"  export ARXIV_AGENT_LLM__OLLAMA_BASE_URL={new_url}")
        return
    
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
                choices=["anthropic", "openai", "gemini", "ollama"],
                default=current_provider
            ).ask()
            
            if not provider:
                raise typer.Abort()
        else:
            provider = current_provider

    if provider:
         valid_providers = ["anthropic", "openai", "gemini", "ollama"]
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


@app.command("set")
def set_config_value(
    key: str = typer.Argument(..., help="Config key in dot notation (e.g., llm.temperature)"),
    value: str = typer.Argument(..., help="Value to set"),
):
    """⚙️ Set a configuration value.
    
    Examples:
        arxiv-agent config set llm.temperature 0.5
        arxiv-agent config set digest.max_papers 10
        arxiv-agent config set retrieval.top_k 5
    """
    settings = get_settings()
    
    # Parse key path
    parts = key.split(".")
    if len(parts) < 2:
        console.print("[red]Invalid key format.[/] Use dot notation like 'llm.temperature'")
        raise typer.Exit(1)
    
    # Map of valid keys and their types
    valid_keys = {
        "llm.temperature": (float, "llm", "temperature"),
        "llm.max_tokens": (int, "llm", "max_tokens"),
        "llm.default_provider": (str, "llm", "default_provider"),
        "digest.max_papers": (int, "digest", "max_papers"),
        "digest.schedule_time": (str, "digest", "schedule_time"),
        "digest.enabled": (bool, "digest", "enabled"),
        "retrieval.top_k": (int, "retrieval", "top_k"),
        "retrieval.rerank_enabled": (bool, "retrieval", "rerank_enabled"),
        "chunking.chunk_size": (int, "chunking", "chunk_size"),
        "chunking.chunk_overlap": (int, "chunking", "chunk_overlap"),
    }
    
    if key not in valid_keys:
        console.print(f"[red]Unknown configuration key: {key}[/]")
        console.print("\n[bold]Valid keys:[/]")
        for k in sorted(valid_keys.keys()):
            console.print(f"  • {k}")
        raise typer.Exit(1)
    
    value_type, section, attr = valid_keys[key]
    
    # Parse and validate value
    try:
        if value_type == bool:
            parsed_value = value.lower() in ("true", "1", "yes", "on")
        elif value_type == int:
            parsed_value = int(value)
        elif value_type == float:
            parsed_value = float(value)
        else:
            parsed_value = value
    except ValueError:
        console.print(f"[red]Invalid value type.[/] Expected {value_type.__name__}")
        raise typer.Exit(1)
    
    # Get the section and set the attribute
    section_obj = getattr(settings, section)
    old_value = getattr(section_obj, attr)
    setattr(section_obj, attr, parsed_value)
    
    console.print(f"[green]✓[/] Set {key}")
    console.print(f"  Old: {old_value}")
    console.print(f"  New: {parsed_value}")
    console.print(f"\n[dim]Note: This change is for the current session only.[/]")
    console.print(f"[dim]To persist, set environment variable:[/]")
    env_key = f"ARXIV_AGENT_{section.upper()}__{attr.upper()}"
    console.print(f"  export {env_key}={parsed_value}")

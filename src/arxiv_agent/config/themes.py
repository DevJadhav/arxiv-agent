"""Theme management for ArXiv Agent CLI.

Provides customizable color themes for Rich console output.
Supports built-in themes and custom theme loading from TOML files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import threading

from rich.console import Console
from rich.style import Style
from rich.theme import Theme as RichTheme
from rich.table import Table
from rich.panel import Panel
from loguru import logger


@dataclass
class AppTheme:
    """Application theme definition.
    
    Defines color scheme for CLI output elements.
    """
    name: str
    description: str = ""
    
    # Primary colors
    primary: str = "#3b82f6"      # Blue
    secondary: str = "#8b5cf6"    # Purple
    accent: str = "#06b6d4"       # Cyan
    
    # Status colors
    success: str = "#22c55e"      # Green
    error: str = "#ef4444"        # Red
    warning: str = "#f59e0b"      # Amber
    info: str = "#3b82f6"         # Blue
    
    # Text colors
    text: str = "#f8fafc"         # Light
    text_muted: str = "#94a3b8"   # Muted
    
    # Background hints (for reference)
    background: str = "#0f172a"   # Dark blue-gray
    surface: str = "#1e293b"      # Lighter surface
    
    # Component-specific
    heading: str = "#f8fafc"
    link: str = "#3b82f6"
    code: str = "#22d3ee"
    border: str = "#334155"
    
    def to_rich_theme(self) -> RichTheme:
        """Convert to Rich Theme for console styling."""
        return RichTheme({
            "primary": Style(color=self.primary),
            "secondary": Style(color=self.secondary),
            "accent": Style(color=self.accent),
            "success": Style(color=self.success),
            "error": Style(color=self.error),
            "warning": Style(color=self.warning),
            "info": Style(color=self.info),
            "text": Style(color=self.text),
            "muted": Style(color=self.text_muted),
            "heading": Style(color=self.heading, bold=True),
            "link": Style(color=self.link, underline=True),
            "code": Style(color=self.code),
            "border": Style(color=self.border),
            # Semantic styles
            "title": Style(color=self.primary, bold=True),
            "subtitle": Style(color=self.text_muted),
            "highlight": Style(color=self.accent, bold=True),
            "dim": Style(color=self.text_muted, dim=True),
        })


# Built-in themes
BUILTIN_THEMES: Dict[str, AppTheme] = {
    "default": AppTheme(
        name="default",
        description="Default dark theme with blue accents",
        primary="#3b82f6",
        secondary="#8b5cf6",
        accent="#06b6d4",
        success="#22c55e",
        error="#ef4444",
        warning="#f59e0b",
        info="#3b82f6",
        text="#f8fafc",
        text_muted="#94a3b8",
        background="#0f172a",
    ),
    "dark": AppTheme(
        name="dark",
        description="Pure dark theme",
        primary="#60a5fa",
        secondary="#a78bfa",
        accent="#22d3ee",
        success="#4ade80",
        error="#f87171",
        warning="#fbbf24",
        info="#60a5fa",
        text="#ffffff",
        text_muted="#a1a1aa",
        background="#000000",
    ),
    "light": AppTheme(
        name="light",
        description="Light theme for bright environments",
        primary="#2563eb",
        secondary="#7c3aed",
        accent="#0891b2",
        success="#16a34a",
        error="#dc2626",
        warning="#d97706",
        info="#2563eb",
        text="#1e293b",
        text_muted="#64748b",
        background="#ffffff",
    ),
    "solarized": AppTheme(
        name="solarized",
        description="Solarized dark color scheme",
        primary="#268bd2",
        secondary="#6c71c4",
        accent="#2aa198",
        success="#859900",
        error="#dc322f",
        warning="#b58900",
        info="#268bd2",
        text="#839496",
        text_muted="#586e75",
        background="#002b36",
    ),
    "dracula": AppTheme(
        name="dracula",
        description="Dracula color scheme",
        primary="#bd93f9",
        secondary="#ff79c6",
        accent="#8be9fd",
        success="#50fa7b",
        error="#ff5555",
        warning="#ffb86c",
        info="#bd93f9",
        text="#f8f8f2",
        text_muted="#6272a4",
        background="#282a36",
    ),
    "high_contrast": AppTheme(
        name="high_contrast",
        description="High contrast for accessibility",
        primary="#00ff00",
        secondary="#ffff00",
        accent="#00ffff",
        success="#00ff00",
        error="#ff0000",
        warning="#ffff00",
        info="#00ffff",
        text="#ffffff",
        text_muted="#cccccc",
        background="#000000",
    ),
}


class ThemeManager:
    """Singleton manager for application themes."""
    
    _instance: Optional["ThemeManager"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "ThemeManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._current_theme_name = "default"
                cls._instance._custom_themes: Dict[str, AppTheme] = {}
                cls._instance._initialized = False
            return cls._instance
    
    def initialize(self, config_dir: Optional[Path] = None) -> None:
        """Initialize theme manager, loading custom themes."""
        if self._initialized:
            return
        
        if config_dir:
            themes_dir = config_dir / "themes"
            if themes_dir.exists():
                self._load_custom_themes(themes_dir)
        
        self._initialized = True
    
    @property
    def current_theme(self) -> AppTheme:
        """Get the current active theme."""
        return self.get_theme(self._current_theme_name)
    
    def get_theme(self, name: str) -> AppTheme:
        """Get a theme by name.
        
        Args:
            name: Theme name
            
        Returns:
            AppTheme instance
            
        Raises:
            KeyError: If theme not found
        """
        if name in BUILTIN_THEMES:
            return BUILTIN_THEMES[name]
        if name in self._custom_themes:
            return self._custom_themes[name]
        raise KeyError(f"Theme not found: {name}")
    
    def get_available_themes(self) -> List[str]:
        """Get list of available theme names."""
        return list(BUILTIN_THEMES.keys()) + list(self._custom_themes.keys())
    
    def set_theme(self, name: str) -> None:
        """Set the current theme.
        
        Args:
            name: Theme name to activate
            
        Raises:
            KeyError: If theme not found
        """
        # Validate theme exists
        self.get_theme(name)
        self._current_theme_name = name
        logger.debug(f"Theme set to: {name}")
    
    def add_custom_theme(self, theme: AppTheme) -> None:
        """Add a custom theme.
        
        Args:
            theme: AppTheme instance to add
        """
        self._custom_themes[theme.name] = theme
    
    def _load_custom_themes(self, themes_dir: Path) -> None:
        """Load custom themes from TOML files."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        
        for theme_file in themes_dir.glob("*.toml"):
            try:
                theme = load_theme_from_file(theme_file)
                self._custom_themes[theme.name] = theme
                logger.debug(f"Loaded custom theme: {theme.name}")
            except Exception as e:
                logger.warning(f"Failed to load theme {theme_file}: {e}")


def load_theme_from_file(path: Path) -> AppTheme:
    """Load a theme from a TOML file.
    
    Args:
        path: Path to TOML file
        
    Returns:
        AppTheme instance
    """
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    
    with open(path, "rb") as f:
        data = tomllib.load(f)
    
    return AppTheme(
        name=data.get("name", path.stem),
        description=data.get("description", ""),
        primary=data.get("primary", BUILTIN_THEMES["default"].primary),
        secondary=data.get("secondary", BUILTIN_THEMES["default"].secondary),
        accent=data.get("accent", BUILTIN_THEMES["default"].accent),
        success=data.get("success", BUILTIN_THEMES["default"].success),
        error=data.get("error", BUILTIN_THEMES["default"].error),
        warning=data.get("warning", BUILTIN_THEMES["default"].warning),
        info=data.get("info", BUILTIN_THEMES["default"].info),
        text=data.get("text", BUILTIN_THEMES["default"].text),
        text_muted=data.get("text_muted", BUILTIN_THEMES["default"].text_muted),
        background=data.get("background", BUILTIN_THEMES["default"].background),
    )


def get_theme_manager() -> ThemeManager:
    """Get the global theme manager instance."""
    return ThemeManager()


def get_current_theme() -> AppTheme:
    """Get the currently active theme."""
    return get_theme_manager().current_theme


def get_theme(name: str) -> AppTheme:
    """Get a theme by name."""
    return get_theme_manager().get_theme(name)


def get_available_themes() -> List[str]:
    """Get list of available theme names."""
    return get_theme_manager().get_available_themes()


def get_custom_themes_dir() -> Path:
    """Get the custom themes directory path."""
    from arxiv_agent.config.settings import get_settings
    return get_settings().config_dir / "themes"


def apply_theme(console: Console, theme: AppTheme) -> Console:
    """Apply a theme to a Rich console.
    
    Args:
        console: Rich Console instance
        theme: AppTheme to apply
        
    Returns:
        Console with theme applied
    """
    # Rich doesn't support changing theme after creation,
    # so we return a new console with the theme
    return Console(theme=theme.to_rich_theme())


def create_themed_table(
    title: str,
    theme: Optional[AppTheme] = None,
    **kwargs
) -> Table:
    """Create a table styled with the current theme.
    
    Args:
        title: Table title
        theme: Optional theme override
        **kwargs: Additional Table arguments
        
    Returns:
        Styled Table instance
    """
    theme = theme or get_current_theme()
    
    return Table(
        title=title,
        title_style=f"bold {theme.primary}",
        header_style=f"bold {theme.secondary}",
        border_style=theme.border,
        **kwargs
    )


def create_themed_panel(
    content: str,
    title: str = "",
    theme: Optional[AppTheme] = None,
    **kwargs
) -> Panel:
    """Create a panel styled with the current theme.
    
    Args:
        content: Panel content
        title: Panel title
        theme: Optional theme override
        **kwargs: Additional Panel arguments
        
    Returns:
        Styled Panel instance
    """
    theme = theme or get_current_theme()
    
    return Panel(
        content,
        title=f"[bold {theme.primary}]{title}[/]" if title else None,
        border_style=theme.border,
        **kwargs
    )


def preview_theme(theme_name: str) -> str:
    """Generate a preview of a theme.
    
    Args:
        theme_name: Name of theme to preview
        
    Returns:
        Rich-formatted preview string
    """
    theme = get_theme(theme_name)
    
    lines = [
        f"[bold]Theme: {theme.name}[/]",
        f"[dim]{theme.description}[/]",
        "",
        "[bold]Colors:[/]",
        f"  [{theme.primary}]Primary ████[/{theme.primary}]",
        f"  [{theme.secondary}]Secondary ████[/{theme.secondary}]",
        f"  [{theme.accent}]Accent ████[/{theme.accent}]",
        "",
        "[bold]Status:[/]",
        f"  [{theme.success}]✓ Success[/{theme.success}]",
        f"  [{theme.error}]✗ Error[/{theme.error}]",
        f"  [{theme.warning}]⚠ Warning[/{theme.warning}]",
        f"  [{theme.info}]ℹ Info[/{theme.info}]",
        "",
        "[bold]Sample Output:[/]",
        f"  [{theme.text}]Regular text[/{theme.text}]",
        f"  [{theme.text_muted}]Muted text[/{theme.text_muted}]",
        f"  [{theme.primary}][bold]Heading[/bold][/{theme.primary}]",
    ]
    
    return "\n".join(lines)

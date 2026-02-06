"""Interactive onboarding wizard for nanobot."""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import questionary
from questionary import Style as QStyle
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from nanobot import __logo__
from nanobot.config.schema import (
    AgentRole,
    ClaudeMaxConfig,
    CodexConfig,
    Config,
    ModelSpec,
    ProviderConfig,
)

# Tuned defaults per agent role — applied when creating new roles in the wizard.
_ROLE_DEFAULTS: dict[str, dict[str, Any]] = {
    "explorer": {"max_tokens": 4096, "temperature": 0.3, "max_tool_iterations": 10},
    "researcher": {"max_tokens": 8192, "temperature": 0.5, "max_tool_iterations": 15},
    "coder": {"max_tokens": 16384, "temperature": 0.2, "max_tool_iterations": 30},
    "thinker": {"max_tokens": 16384, "temperature": 0.8, "max_tool_iterations": 5},
}

# Questionary styling — matches the Rich cyan/green palette
_STYLE = QStyle(
    [
        ("qmark", "fg:ansicyan bold"),
        ("question", "bold"),
        ("answer", "fg:ansigreen bold"),
        ("pointer", "fg:ansicyan bold"),
        ("highlighted", "fg:ansicyan bold"),
        ("selected", "fg:ansigreen"),
        ("separator", "fg:ansicyan"),
        ("instruction", "fg:ansiyellow"),
    ]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mask_key(key: str) -> str:
    """Mask an API key, showing only the last 4 characters."""
    if not key:
        return "[dim]not set[/dim]"
    return f"****{key[-4:]}"


def _mask_key_plain(key: str) -> str:
    """Mask for questionary (no Rich markup)."""
    if not key:
        return "not set"
    return f"****{key[-4:]}"


def get_available_providers(config: Config) -> list[str]:
    """Return provider names that have API keys set or claude_max enabled."""
    available: list[str] = []
    providers = {
        "anthropic": config.providers.anthropic.api_key,
        "openai": config.providers.openai.api_key,
        "gemini": config.providers.gemini.api_key,
        "openrouter": config.providers.openrouter.api_key,
        "groq": config.providers.groq.api_key,
        "zhipu": config.providers.zhipu.api_key,
        "opencode": config.providers.opencode.api_key,
        "vllm": config.providers.vllm.api_key,
    }
    for name, key in providers.items():
        if key:
            available.append(name)
    if config.providers.claude_max.enabled:
        available.append("claude_max")
    if config.providers.codex.enabled:
        available.append("codex")
    return available


def recommend_models(available: list[str]) -> dict[str, str]:
    """Return ``{role: model_id}`` recommendations based on available providers."""
    cheap = [
        ("gemini", "gemini/gemini-2.5-flash-preview"),
        ("anthropic", "anthropic/claude-haiku-4-5"),
        ("openai", "openai/gpt-4o-mini"),
        ("codex", "codex/o4-mini"),
        ("openrouter", "openrouter/google/gemini-2.5-flash-preview"),
        ("groq", "groq/llama-3.3-70b-versatile"),
    ]
    mid = [
        ("claude_max", "claude_max/claude-sonnet-4-5"),
        ("anthropic", "anthropic/claude-sonnet-4-5"),
        ("gemini", "gemini/gemini-2.5-pro-preview"),
        ("openai", "openai/gpt-4o"),
        ("codex", "codex/gpt-4o"),
        ("openrouter", "openrouter/anthropic/claude-sonnet-4-5"),
    ]
    power = [
        ("claude_max", "claude_max/claude-opus-4-6"),
        ("anthropic", "anthropic/claude-opus-4-6"),
        ("gemini", "gemini/gemini-2.5-pro-preview"),
        ("openai", "openai/o1"),
        ("codex", "codex/o3"),
        ("openrouter", "openrouter/anthropic/claude-opus-4-6"),
    ]

    def _pick(tier: list[tuple[str, str]]) -> str | None:
        for provider, model in tier:
            if provider in available:
                return model
        return None

    cheap_model = _pick(cheap)
    mid_model = _pick(mid)
    power_model = _pick(power)

    result: dict[str, str] = {}
    if cheap_model:
        result["orchestrator"] = cheap_model
        result["explorer"] = cheap_model
        result["researcher"] = cheap_model
    if mid_model:
        result["coder"] = mid_model
    if power_model:
        result["thinker"] = power_model
    return result


# ---------------------------------------------------------------------------
# Questionary wrappers (return None on Ctrl-C to go back to menu)
# ---------------------------------------------------------------------------


def _restore_terminal() -> None:
    """Restore terminal to cooked mode after questionary/prompt_toolkit."""
    if sys.platform == "win32":
        # On Windows, reset the console mode so the shell accepts input again.
        os.system("")  # noqa: S605 — triggers Windows to reset console mode
    else:
        # On Unix, `stty sane` restores terminal settings.
        os.system("stty sane 2>/dev/null")  # noqa: S605


def _ask_confirm(message: str, default: bool = False) -> bool | None:
    return questionary.confirm(message, default=default, style=_STYLE).ask()


def _ask_text(message: str, default: str = "") -> str | None:
    return questionary.text(message, default=default, style=_STYLE).ask()


def _ask_password(message: str) -> str | None:
    return questionary.password(message, style=_STYLE).ask()


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

_PROVIDER_NAMES = [
    "anthropic",
    "openai",
    "gemini",
    "openrouter",
    "groq",
    "zhipu",
    "opencode",
    "vllm",
]


def _get_provider_config(config: Config, name: str) -> ProviderConfig:
    """Get a provider config by name."""
    return getattr(config.providers, name)


def _section_providers(console: Console, config: Config) -> None:
    """Section 1 — configure provider API keys."""
    console.print(Rule("Provider API Keys"))

    table = Table(show_header=True)
    table.add_column("Provider", style="cyan")
    table.add_column("API Key")
    table.add_column("API Base")

    for name in _PROVIDER_NAMES:
        prov = _get_provider_config(config, name)
        table.add_row(name, mask_key(prov.api_key), prov.api_base or "[dim]-[/dim]")
    console.print(table)
    console.print()

    # Let user pick which providers to configure
    choices = []
    for name in _PROVIDER_NAMES:
        prov = _get_provider_config(config, name)
        status = _mask_key_plain(prov.api_key)
        choices.append(questionary.Choice(f"{name}  ({status})", value=name))

    selected = questionary.checkbox(
        "Select providers to configure (Space to toggle, Enter to confirm):",
        choices=choices,
        style=_STYLE,
    ).ask()

    if not selected:
        console.print("[dim]No providers selected.[/dim]\n")
        return

    for name in selected:
        prov = _get_provider_config(config, name)
        console.print(f"\n[cyan]{name}[/cyan]:")

        api_key = _ask_text(f"  {name} API key", default=prov.api_key)
        if api_key is None:
            continue
        api_key = api_key.strip()
        if api_key:
            prov.api_key = api_key

        api_base = _ask_text(f"  {name} API base URL (Enter to skip)", default=prov.api_base or "")
        if api_base is not None:
            api_base = api_base.strip()
            if api_base:
                prov.api_base = api_base

    console.print("[green]Provider keys updated.[/green]\n")


def _section_claude_sub(console: Console, config: Config) -> None:
    """Section 2 — configure Claude subscription (CLI)."""
    console.print(Rule("Claude Subscription"))
    console.print(
        "Claude subscription uses the [cyan]claude[/cyan] CLI with your "
        "Claude Pro/Max subscription - no API key needed."
    )
    status = "[green]enabled[/green]" if config.providers.claude_max.enabled else "[dim]disabled[/dim]"
    console.print(f"Current status: {status}\n")

    enable = _ask_confirm("Enable Claude subscription?", default=config.providers.claude_max.enabled)
    if enable is None:
        return

    if not enable:
        if config.providers.claude_max.enabled:
            config.providers.claude_max.enabled = False
            console.print("[yellow]Claude subscription disabled.[/yellow]\n")
        return

    # Check if claude CLI is on PATH
    cli_path = config.providers.claude_max.cli_path or "claude"
    found = shutil.which(cli_path)
    if found:
        console.print(f"[green]Found claude CLI at {found}[/green]")
    else:
        console.print("[yellow]claude CLI not found on PATH.[/yellow]")
        if platform.system() == "Windows":
            console.print("Install with: [cyan]irm https://claude.ai/install.ps1 | iex[/cyan]")
        else:
            console.print(
                "Install with: [cyan]curl -fsSL https://claude.ai/install.sh | bash[/cyan]"
            )
        console.print()

    custom_path = _ask_text("CLI path", default=cli_path)
    if custom_path is not None:
        custom_path = custom_path.strip()
        if custom_path:
            cli_path = custom_path

    # Offer to authenticate via claude login (interactive browser OAuth)
    oauth_token = ""
    auth = _ask_confirm(
        "Authenticate now? (will open browser for Claude OAuth)",
        default=False,
    )
    if auth:
        console.print("[dim]Running 'claude login'...[/dim]")
        try:
            result = subprocess.run(
                [cli_path, "login"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                console.print("[green]Authentication successful![/green]")
            else:
                console.print(
                    f"[yellow]claude login returned exit {result.returncode}[/yellow]"
                )
                if result.stderr:
                    console.print(f"[dim]{result.stderr[:300]}[/dim]")
        except FileNotFoundError:
            console.print(
                f"[red]Could not run '{cli_path}'. Make sure it's installed.[/red]"
            )
        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")
    else:
        # Offer headless token for environments without a browser
        use_token = _ask_confirm(
            "Set CLAUDE_CODE_OAUTH_TOKEN for headless use? "
            "(run 'claude setup-token' in another terminal first)",
            default=False,
        )
        if use_token:
            token = _ask_text("Paste your OAuth token")
            if token and token.strip():
                oauth_token = token.strip()

    # Reasoning effort level
    effort = _ask_text(
        "Effort level (low/medium/high, blank for default)",
        default=config.providers.claude_max.effort_level or "",
    )
    effort_level = effort.strip().lower() if effort else ""
    if effort_level and effort_level not in ("low", "medium", "high"):
        console.print(f"[yellow]Unknown effort '{effort_level}', skipping.[/yellow]")
        effort_level = ""

    config.providers.claude_max = ClaudeMaxConfig(
        enabled=True, cli_path=cli_path, oauth_token=oauth_token,
        effort_level=effort_level,
    )
    console.print("[green]Claude subscription enabled.[/green]\n")


def _section_codex_sub(console: Console, config: Config) -> None:
    """Section — configure Codex subscription (CLI)."""
    console.print(Rule("Codex Subscription"))
    console.print(
        "Codex subscription uses the [cyan]codex[/cyan] CLI with your "
        "ChatGPT Plus/Pro subscription - no API key needed."
    )
    status = "[green]enabled[/green]" if config.providers.codex.enabled else "[dim]disabled[/dim]"
    console.print(f"Current status: {status}\n")

    enable = _ask_confirm("Enable Codex subscription?", default=config.providers.codex.enabled)
    if enable is None:
        return

    if not enable:
        if config.providers.codex.enabled:
            config.providers.codex.enabled = False
            console.print("[yellow]Codex subscription disabled.[/yellow]\n")
        return

    # Check if codex CLI is on PATH
    cli_path = config.providers.codex.cli_path or "codex"
    found = shutil.which(cli_path)
    if found:
        console.print(f"[green]Found codex CLI at {found}[/green]")
    else:
        console.print("[yellow]codex CLI not found on PATH.[/yellow]")
        console.print("Install with: [cyan]npm install -g @openai/codex[/cyan]")
        console.print()

    custom_path = _ask_text("CLI path", default=cli_path)
    if custom_path is not None:
        custom_path = custom_path.strip()
        if custom_path:
            cli_path = custom_path

    # Offer to authenticate via codex login
    auth = _ask_confirm(
        "Authenticate now? (will open browser for ChatGPT OAuth)",
        default=False,
    )
    if auth:
        console.print("[dim]Running 'codex login'...[/dim]")
        try:
            result = subprocess.run(
                [cli_path, "login"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                console.print("[green]Authentication successful![/green]")
            else:
                console.print(
                    f"[yellow]codex login returned exit {result.returncode}[/yellow]"
                )
                if result.stderr:
                    console.print(f"[dim]{result.stderr[:300]}[/dim]")
        except FileNotFoundError:
            console.print(
                f"[red]Could not run '{cli_path}'. Make sure it's installed.[/red]"
            )
        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")

    config.providers.codex = CodexConfig(enabled=True, cli_path=cli_path)
    console.print("[green]Codex subscription enabled.[/green]\n")


def _section_models(console: Console, config: Config) -> None:
    """Section 3 — model selection with smart recommendations."""
    console.print(Rule("Model Selection"))

    available = get_available_providers(config)
    if not available:
        console.print(
            "[yellow]No providers configured yet. Set up provider keys or "
            "a CLI subscription first.[/yellow]\n"
        )
        return

    console.print(f"Available providers: {', '.join(available)}\n")

    recommendations = recommend_models(available)

    if not recommendations:
        console.print("[yellow]Could not generate model recommendations.[/yellow]\n")
        return

    # Show recommendation table
    table = Table(title="Recommended Models", show_header=True)
    table.add_column("Role", style="cyan")
    table.add_column("Recommended Model", style="green")
    table.add_column("Temp", justify="right")
    table.add_column("Iterations", justify="right")

    role_order = ["orchestrator", "explorer", "researcher", "coder", "thinker"]
    for role in role_order:
        model = recommendations.get(role)
        if model:
            rd = _ROLE_DEFAULTS.get(role, {})
            temp = str(rd.get("temperature", "-"))
            iters = str(rd.get("max_tool_iterations", "-"))
            table.add_row(role, model, temp, iters)
    console.print(table)
    console.print()

    # Orchestrator = default model
    orch_rec = recommendations.get("orchestrator")
    if orch_rec:
        current = config.agents.defaults.model.model
        choices = [
            questionary.Choice(f"{orch_rec}  (recommended)", value=orch_rec),
            questionary.Choice(f"{current}  (current)", value=current),
            questionary.Choice("Enter custom model...", value="__custom__"),
        ]
        # Deduplicate if current == recommended
        if current == orch_rec:
            choices = [
                questionary.Choice(f"{orch_rec}  (current, recommended)", value=orch_rec),
                questionary.Choice("Enter custom model...", value="__custom__"),
            ]
        answer = questionary.select(
            "Orchestrator (default model):",
            choices=choices,
            style=_STYLE,
        ).ask()
        if answer == "__custom__":
            custom = _ask_text("  Custom model (provider/model)")
            if custom and custom.strip():
                config.agents.defaults.model.model = custom.strip()
        elif answer is not None:
            config.agents.defaults.model.model = answer

    # Agent roles
    for role in ["explorer", "researcher", "coder", "thinker"]:
        rec = recommendations.get(role)
        if not rec:
            continue

        existing_role = config.agents.roles.get(role)
        current_model = existing_role.model.model if existing_role and existing_role.model else None

        choices = [
            questionary.Choice(f"{rec}  (recommended)", value=rec),
        ]
        if current_model and current_model != rec:
            choices.append(questionary.Choice(f"{current_model}  (current)", value=current_model))
        elif current_model == rec:
            choices[0] = questionary.Choice(
                f"{rec}  (current, recommended)", value=rec
            )
        choices.append(questionary.Choice("Enter custom model...", value="__custom__"))
        choices.append(questionary.Choice("Skip", value="__skip__"))

        answer = questionary.select(
            f"{role.title()} model:",
            choices=choices,
            style=_STYLE,
        ).ask()

        if answer is None or answer == "__skip__":
            continue
        if answer == "__custom__":
            custom = _ask_text("  Custom model (provider/model)")
            if not custom or not custom.strip():
                continue
            chosen_model = custom.strip()
        else:
            chosen_model = answer

        defaults = _ROLE_DEFAULTS.get(role, {})
        if role not in config.agents.roles:
            config.agents.roles[role] = AgentRole(model=ModelSpec(model=chosen_model, **defaults))
        else:
            existing = config.agents.roles[role]
            if existing.model is None:
                existing.model = ModelSpec(model=chosen_model, **defaults)
            else:
                existing.model.model = chosen_model

    console.print("[green]Models updated.[/green]\n")


def _section_channels(console: Console, config: Config) -> None:
    """Section 4 — Telegram and WhatsApp channel setup."""
    console.print(Rule("Channels"))

    # --- Telegram ---
    tg = config.channels.telegram
    tg_status = "[green]enabled[/green]" if tg.enabled else "[dim]disabled[/dim]"
    console.print(f"Telegram: {tg_status}")

    if _ask_confirm("Set up Telegram?", default=tg.enabled):
        token = _ask_text("  Bot token (from @BotFather)", default=tg.token)
        if token is not None:
            token = token.strip()
            if token:
                tg.token = token
        allowed = _ask_text(
            "  Allowed user IDs (comma-separated, Enter for all)",
            default=",".join(tg.allow_from) if tg.allow_from else "",
        )
        if allowed is not None:
            allowed = allowed.strip()
            tg.allow_from = [x.strip() for x in allowed.split(",") if x.strip()] if allowed else []
        tg.enabled = True
        console.print("[green]Telegram configured.[/green]")
    console.print()

    # --- WhatsApp ---
    wa = config.channels.whatsapp
    wa_status = "[green]enabled[/green]" if wa.enabled else "[dim]disabled[/dim]"
    console.print(f"WhatsApp: {wa_status}")

    if _ask_confirm("Set up WhatsApp?", default=wa.enabled):
        bridge_url = _ask_text("  Bridge URL", default=wa.bridge_url)
        if bridge_url is not None:
            bridge_url = bridge_url.strip()
            if bridge_url:
                wa.bridge_url = bridge_url
        allowed = _ask_text(
            "  Allowed phone numbers (comma-separated, Enter for all)",
            default=",".join(wa.allow_from) if wa.allow_from else "",
        )
        if allowed is not None:
            allowed = allowed.strip()
            wa.allow_from = [x.strip() for x in allowed.split(",") if x.strip()] if allowed else []
        wa.enabled = True
        console.print("[green]WhatsApp configured.[/green]")
        console.print("[dim]Run `nanobot channels login` to scan QR code after setup.[/dim]")
    console.print()


def _section_web_search(console: Console, config: Config) -> None:
    """Section 5 — Brave Search API key."""
    console.print(Rule("Web Search"))

    current = config.tools.web.search.api_key
    status = mask_key(current) if current else "[dim]not set[/dim]"
    console.print(f"Brave Search API: {status}")
    console.print("[dim]Get a key at https://brave.com/search/api/[/dim]\n")

    if _ask_confirm("Configure Brave Search API?", default=bool(current)):
        api_key = _ask_text("  Brave API key", default=current)
        if api_key is not None:
            api_key = api_key.strip()
            if api_key:
                config.tools.web.search.api_key = api_key
                console.print("[green]Web search configured.[/green]")
    console.print()


def _section_review_save(
    console: Console,
    config: Config,
    config_path: Path,
    workspace: Path,
) -> bool:
    """Section 6 — review the full config and save. Returns True if saved."""
    console.print(Rule("Review & Save"))

    # Providers summary
    table = Table(title="Configuration Summary", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    # Providers
    for name in _PROVIDER_NAMES:
        prov = _get_provider_config(config, name)
        if prov.api_key:
            table.add_row(f"Provider: {name}", mask_key(prov.api_key))

    # CLI subscriptions
    cm = config.providers.claude_max
    table.add_row("Claude sub", "[green]enabled[/green]" if cm.enabled else "[dim]disabled[/dim]")
    cx = config.providers.codex
    table.add_row("Codex sub", "[green]enabled[/green]" if cx.enabled else "[dim]disabled[/dim]")

    # Default model
    table.add_row("Default model", config.agents.defaults.model.model)

    # Role models
    for role_name, role_cfg in config.agents.roles.items():
        if role_cfg.model:
            table.add_row(f"Role: {role_name}", role_cfg.model.model)

    # Channels
    tg = config.channels.telegram
    wa = config.channels.whatsapp
    table.add_row("Telegram", "[green]enabled[/green]" if tg.enabled else "[dim]disabled[/dim]")
    table.add_row("WhatsApp", "[green]enabled[/green]" if wa.enabled else "[dim]disabled[/dim]")

    # Web search
    ws = config.tools.web.search
    table.add_row(
        "Web search",
        mask_key(ws.api_key) if ws.api_key else "[dim]not set[/dim]",
    )

    console.print(table)
    console.print()

    if not _ask_confirm("Save configuration?", default=True):
        return False

    # Import here to avoid circular imports at module level
    from nanobot.cli.commands import _create_workspace_templates
    from nanobot.config.loader import save_config

    # Create workspace + templates
    workspace.mkdir(parents=True, exist_ok=True)
    _create_workspace_templates(workspace)

    # Save config
    save_config(config, config_path)
    console.print(f"\n[green]Configuration saved to {config_path}[/green]")
    console.print(f"[green]Workspace at {workspace}[/green]")
    console.print(f"\n{__logo__} nanobot is ready!")
    console.print("\nNext steps:")
    console.print('  Chat: [cyan]nanobot agent -m "Hello!"[/cyan]')
    if tg.enabled:
        console.print("  Telegram bot is configured and ready to use.")
    if wa.enabled:
        console.print("  WhatsApp: run [cyan]nanobot channels login[/cyan] to scan QR code.")
    return True


# ---------------------------------------------------------------------------
# Main wizard
# ---------------------------------------------------------------------------

_MENU_CHOICES = [
    questionary.Choice("1. Provider API Keys", value="providers"),
    questionary.Choice("2. Claude Subscription (CLI)", value="claude_sub"),
    questionary.Choice("3. Codex Subscription (CLI)", value="codex_sub"),
    questionary.Choice("4. Model Selection", value="models"),
    questionary.Choice("5. Channels (Telegram, WhatsApp)", value="channels"),
    questionary.Choice("6. Web Search (Brave API)", value="web_search"),
    questionary.Separator(),
    questionary.Choice("7. Review & Save", value="save"),
    questionary.Choice("0. Exit without saving", value="exit"),
]


def run_wizard(
    console: Console,
    config: Config,
    config_path: Path,
    workspace: Path,
) -> None:
    """Run the interactive onboarding wizard."""
    # Welcome banner
    console.print(
        Panel(
            f"[bold]{__logo__}  nanobot Setup Wizard[/bold]\n\n"
            "Configure providers, models, channels, and more.\n"
            "Use arrow keys to navigate, Enter to select.",
            expand=False,
        )
    )

    # Show current config summary if anything is configured
    available = get_available_providers(config)
    if available:
        console.print(f"[dim]Existing providers: {', '.join(available)}[/dim]")
        console.print(f"[dim]Current model: {config.agents.defaults.model.model}[/dim]\n")

    dispatch = {
        "providers": lambda: _section_providers(console, config),
        "claude_sub": lambda: _section_claude_sub(console, config),
        "codex_sub": lambda: _section_codex_sub(console, config),
        "models": lambda: _section_models(console, config),
        "channels": lambda: _section_channels(console, config),
        "web_search": lambda: _section_web_search(console, config),
    }

    try:
        while True:
            choice = questionary.select(
                "Main Menu:",
                choices=_MENU_CHOICES,
                style=_STYLE,
                use_arrow_keys=True,
                use_jk_keys=False,
            ).ask()

            if choice is None or choice == "exit":
                console.print("[yellow]Exiting without saving.[/yellow]")
                return
            elif choice == "save":
                if _section_review_save(console, config, config_path, workspace):
                    return
            else:
                dispatch[choice]()
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting without saving.[/yellow]")
    finally:
        _restore_terminal()

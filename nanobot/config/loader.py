"""Configuration loading utilities."""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from nanobot.config.schema import Config


def get_config_path() -> Path:
    """Get the default configuration file path."""
    return Path.home() / ".nanobot" / "config.json"


def get_data_dir() -> Path:
    """Get the nanobot data directory."""
    from nanobot.utils.helpers import get_data_path

    return get_data_path()


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from file or create default.

    Args:
        config_path: Optional path to config file. Uses default if not provided.

    Returns:
        Loaded configuration object.
    """
    path = config_path or get_config_path()

    if path.exists():
        data: Any = {}
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
            raw = json.loads(_strip_json_comments(content))
            data = convert_keys(raw)
            data = _coerce_model_spec(data)
            return Config.model_validate(data)
        except ValidationError as e:
            data = _coerce_model_spec(data)
            data = _coerce_allow_from(data)
            try:
                print(f"Warning: Config validation error in {path}: {e}")
                print("Attempting to coerce allow_from values to strings.")
                return Config.model_validate(data)
            except ValidationError as e2:
                print(f"Warning: Failed to load config from {path}: {e2}")
                print("Using default configuration.")
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to load config from {path}: {e}")
            print("Using default configuration.")

    return Config()


def _coerce_allow_from(data: Any) -> Any:
    """Coerce allow_from entries to strings when possible."""
    if not isinstance(data, dict):
        return data

    channels = data.get("channels")
    if not isinstance(channels, dict):
        return data

    updated_channels = dict(channels)
    for channel_name in ("telegram", "whatsapp"):
        channel = updated_channels.get(channel_name)
        if not isinstance(channel, dict):
            continue
        allow_from = channel.get("allow_from")
        if isinstance(allow_from, list):
            channel = dict(channel)
            channel["allow_from"] = [str(value) for value in allow_from]
            updated_channels[channel_name] = channel

    updated_data = dict(data)
    updated_data["channels"] = updated_channels
    return updated_data


def _coerce_model_spec(data: Any) -> Any:
    """Coerce legacy model fields into the new model spec."""
    if not isinstance(data, dict):
        return data

    agents = data.get("agents")
    if not isinstance(agents, dict):
        return data

    defaults = agents.get("defaults")
    if isinstance(defaults, dict):
        model_value = defaults.get("model")
        model_spec: dict[str, Any] = {}

        if isinstance(model_value, str):
            model_spec["model"] = model_value
        elif isinstance(model_value, dict):
            model_spec.update(model_value)

        # Handle both snake_case and camelCase variants
        if "max_tokens" in defaults:
            model_spec.setdefault("max_tokens", defaults.pop("max_tokens"))
        if "maxTokens" in defaults:
            model_spec.setdefault("max_tokens", defaults.pop("maxTokens"))
        if "temperature" in defaults:
            model_spec.setdefault("temperature", defaults.pop("temperature"))
        if "max_tool_iterations" in defaults:
            model_spec.setdefault("max_tool_iterations", defaults.pop("max_tool_iterations"))
        if "maxToolIterations" in defaults:
            model_spec.setdefault("max_tool_iterations", defaults.pop("maxToolIterations"))
        if "fallbacks" in defaults:
            model_spec.setdefault("fallbacks", defaults.pop("fallbacks"))

        if model_spec:
            defaults = dict(defaults)
            defaults["model"] = model_spec
            agents = dict(agents)
            agents["defaults"] = defaults

    roles = agents.get("roles") if isinstance(agents, dict) else None
    if isinstance(roles, dict):
        updated_roles = dict(roles)
        for role_name, role_value in roles.items():
            if not isinstance(role_value, dict):
                continue
            role_model = role_value.get("model")
            if isinstance(role_model, str):
                role_value = dict(role_value)
                role_value["model"] = {"model": role_model}
                updated_roles[role_name] = role_value
        agents = dict(agents)
        agents["roles"] = updated_roles

    updated_data = dict(data)
    updated_data["agents"] = agents
    return updated_data


def save_config(config: Config, config_path: Path | None = None) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration to save.
        config_path: Optional path to save to. Uses default if not provided.
    """
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to camelCase format
    data = config.model_dump()
    data = convert_to_camel(data)

    content = render_config_with_comments(data)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _strip_json_comments(text: str) -> str:
    """Strip // and /* */ comments from JSON-like text."""
    result: list[str] = []
    in_string = False
    escape = False
    i = 0
    while i < len(text):
        char = text[i]
        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            i += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            i += 1
            continue

        if char == "/" and i + 1 < len(text):
            next_char = text[i + 1]
            if next_char == "/":
                i += 2
                while i < len(text) and text[i] not in "\r\n":
                    i += 1
                continue
            if next_char == "*":
                i += 2
                while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                    i += 1
                i += 2
                continue

        result.append(char)
        i += 1

    return "".join(result)


def _clean_roles(roles: dict[str, Any]) -> dict[str, Any]:
    """Recursively strip keys with ``None`` or empty-list ``[]`` values from role dicts."""

    def _clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if v is not None and v != []}
        if isinstance(obj, list):
            return [_clean(item) for item in obj]
        return obj

    return {name: _clean(role) for name, role in roles.items()}


def render_config_with_comments(data: dict[str, Any]) -> str:
    """Render a JSON-with-comments config template."""

    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=True)

    def _json_indented(value: Any, indent_level: int = 2) -> str:
        """Pretty-print JSON with each line indented to *indent_level* spaces."""
        raw = json.dumps(value, ensure_ascii=True, indent=2)
        prefix = " " * indent_level
        lines = raw.splitlines()
        # First line sits at the insertion point; subsequent lines need the prefix.
        return lines[0] + "\n".join([""] + [prefix + line for line in lines[1:]]) if len(lines) > 1 else raw

    agents = data["agents"]["defaults"]
    model_spec = agents["model"]
    channels = data["channels"]
    providers = data["providers"]
    gateway = data["gateway"]
    tools = data["tools"]["web"]["search"]
    roles = _clean_roles(data["agents"].get("roles", {}))

    return """{
  // Default agent behavior and model selection.
  "agents": {
    "defaults": {
      // Filesystem path used by the agent for workspace files.
      "workspace": %s,
      // Default model spec for the agent (provider/model + settings).
      "model": {
        // Default model (provider/model).
        "model": %s,
        // Maximum tokens per response.
        "maxTokens": %s,
        // Sampling temperature (0.0-1.0).
        "temperature": %s,
        // Max tool calls per turn.
        "maxToolIterations": %s,
        // Optional fallback models (same shape as above, no nested fallbacks).
        "fallbacks": %s
      }
    },
    // Optional role overrides for subagents (explorer, researcher, coder, thinker).
    "roles": %s
  },

  // Chat channel configuration.
  "channels": {
    "whatsapp": {
      // Enable the WhatsApp bridge channel.
      "enabled": %s,
      // WebSocket URL for the local bridge.
      "bridgeUrl": %s,
      // Allowed phone numbers (strings). Empty = allow all.
      "allowFrom": %s
    },
    "telegram": {
      // Enable the Telegram channel.
      "enabled": %s,
      // Bot token from @BotFather.
      "token": %s,
      // Allowed Telegram user IDs/usernames (strings). Empty = allow all.
      "allowFrom": %s
    }
  },

  // Provider API keys and base URLs.
  "providers": {
    "anthropic": {
      // Anthropic API key.
      "apiKey": %s,
      // Optional API base URL.
      "apiBase": %s
    },
    "openai": {
      // OpenAI API key.
      "apiKey": %s,
      // Optional API base URL.
      "apiBase": %s
    },
    "opencode": {
      // OpenCode Zen API key.
      "apiKey": %s,
      // Optional API base URL (default is https://opencode.ai/zen/v1).
      "apiBase": %s
    },
    "openrouter": {
      // OpenRouter API key.
      "apiKey": %s,
      // Optional API base URL (default is https://openrouter.ai/api/v1).
      "apiBase": %s
    },
    "groq": {
      // Groq API key.
      "apiKey": %s,
      // Optional API base URL.
      "apiBase": %s
    },
    "zhipu": {
      // Zhipu API key.
      "apiKey": %s,
      // Optional API base URL.
      "apiBase": %s
    },
    "vllm": {
      // vLLM API key (if required).
      "apiKey": %s,
      // vLLM API base URL.
      "apiBase": %s
    },
    "gemini": {
      // Gemini API key.
      "apiKey": %s,
      // Optional API base URL.
      "apiBase": %s
    },
    "claudeMax": {
      // Enable Claude CLI provider (Claude subscription, no API key needed).
      "enabled": %s,
      // Path to the claude CLI binary.
      "cliPath": %s,
      // OAuth token for headless use (CLAUDE_CODE_OAUTH_TOKEN).
      "oauthToken": %s,
      // Reasoning effort level: "low", "medium", or "high" (default: high).
      "effortLevel": %s
    },
    "codex": {
      // Enable Codex CLI provider (ChatGPT/Codex subscription, no API key needed).
      "enabled": %s,
      // Path to the codex CLI binary.
      "cliPath": %s
    }
  },

  // Gateway/server binding.
  "gateway": {
    // Host to bind (use 127.0.0.1 for local-only).
    "host": %s,
    // Port to bind the gateway.
    "port": %s
  },

  // Tool configuration.
  "tools": {
    "web": {
      "search": {
        // Brave Search API key.
        "apiKey": %s,
        // Max results per web search.
        "maxResults": %s
      }
    }
  }
}
""" % (
        _json(agents["workspace"]),
        _json(model_spec["model"]),
        _json(model_spec["maxTokens"]),
        _json(model_spec["temperature"]),
        _json(model_spec["maxToolIterations"]),
        _json(model_spec.get("fallbacks", [])),
        _json_indented(roles, 4),
        _json(channels["whatsapp"]["enabled"]),
        _json(channels["whatsapp"]["bridgeUrl"]),
        _json(channels["whatsapp"]["allowFrom"]),
        _json(channels["telegram"]["enabled"]),
        _json(channels["telegram"]["token"]),
        _json(channels["telegram"]["allowFrom"]),
        _json(providers["anthropic"]["apiKey"]),
        _json(providers["anthropic"]["apiBase"]),
        _json(providers["openai"]["apiKey"]),
        _json(providers["openai"]["apiBase"]),
        _json(providers["opencode"]["apiKey"]),
        _json(providers["opencode"]["apiBase"]),
        _json(providers["openrouter"]["apiKey"]),
        _json(providers["openrouter"]["apiBase"]),
        _json(providers["groq"]["apiKey"]),
        _json(providers["groq"]["apiBase"]),
        _json(providers["zhipu"]["apiKey"]),
        _json(providers["zhipu"]["apiBase"]),
        _json(providers["vllm"]["apiKey"]),
        _json(providers["vllm"]["apiBase"]),
        _json(providers["gemini"]["apiKey"]),
        _json(providers["gemini"]["apiBase"]),
        _json(providers.get("claudeMax", {}).get("enabled", False)),
        _json(providers.get("claudeMax", {}).get("cliPath", "claude")),
        _json(providers.get("claudeMax", {}).get("oauthToken", "")),
        _json(providers.get("claudeMax", {}).get("effortLevel", "")),
        _json(providers.get("codex", {}).get("enabled", False)),
        _json(providers.get("codex", {}).get("cliPath", "codex")),
        _json(gateway["host"]),
        _json(gateway["port"]),
        _json(tools["apiKey"]),
        _json(tools["maxResults"]),
    )


def convert_keys(data: Any) -> Any:
    """Convert camelCase keys to snake_case for Pydantic."""
    if isinstance(data, dict):
        return {camel_to_snake(k): convert_keys(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_keys(item) for item in data]
    return data


def convert_to_camel(data: Any) -> Any:
    """Convert snake_case keys to camelCase."""
    if isinstance(data, dict):
        return {snake_to_camel(k): convert_to_camel(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_to_camel(item) for item in data]
    return data


def camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    result = []
    for i, char in enumerate(name):
        if char.isupper() and i > 0:
            result.append("_")
        result.append(char.lower())
    return "".join(result)


def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])

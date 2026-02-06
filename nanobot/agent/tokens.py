"""Token counting abstraction with multi-provider support."""

import json
from typing import Any

from loguru import logger

# Well-known context window sizes (in tokens)
CONTEXT_WINDOWS: dict[str, int] = {
    # Anthropic
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "claude-3.5-sonnet": 200_000,
    "claude-3.5-haiku": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
    "claude-haiku-4": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-opus-4-5": 200_000,
    "claude-opus-4-6": 200_000,
    # OpenAI
    "gpt-4": 8_192,
    "gpt-4-32k": 32_768,
    "gpt-4-turbo": 128_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 1_047_576,
    "gpt-4.1-mini": 1_047_576,
    "gpt-4.1-nano": 1_047_576,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o1-preview": 128_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    # Google Gemini
    "gemini-1.5-pro": 1_000_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-2.0-flash": 1_000_000,
    "gemini-2.5-pro": 1_000_000,
    "gemini-2.5-flash": 1_000_000,
    # DeepSeek
    "deepseek-chat": 64_000,
    "deepseek-reasoner": 64_000,
    "deepseek-coder": 128_000,
}

# Default fallback
DEFAULT_CONTEXT_WINDOW = 128_000


def _get_encoding(model: str) -> Any:
    """Get the tiktoken encoding for a model. Returns None on import failure."""
    try:
        import tiktoken
    except ImportError:
        return None

    model_lower = model.lower()

    # o200k_base for gpt-4o family and newer
    if any(tag in model_lower for tag in ["gpt-4o", "gpt-4.1", "o1", "o3", "o4"]):
        return tiktoken.get_encoding("o200k_base")

    # cl100k_base for everything else (gpt-4, claude approximation, default)
    return tiktoken.get_encoding("cl100k_base")


def _strip_provider(model: str) -> str:
    """Remove provider prefix from model string (e.g. 'anthropic/claude-3' -> 'claude-3')."""
    if "/" in model:
        return model.split("/", 1)[1]
    return model


class TokenCounter:
    """Lightweight token counter with multi-provider support.

    Routes to the appropriate counting method based on model prefix:
    - openai/, deepseek/ -> tiktoken (cl100k_base or o200k_base)
    - anthropic/ -> tiktoken cl100k_base as approximation
    - gemini/ -> character-based estimate (~4 chars/token)
    - Default -> tiktoken cl100k_base
    """

    def count(self, text: str, model: str) -> int:
        """Count tokens for a text string.

        Args:
            text: The text to count tokens for.
            model: Model identifier (e.g. 'anthropic/claude-sonnet-4-5').

        Returns:
            Estimated token count.
        """
        if not text:
            return 0

        model_lower = model.lower()

        # Gemini: character-based estimate
        if model_lower.startswith("gemini/") or "gemini" in model_lower:
            return len(text) // 4 + 1

        # Everything else: tiktoken
        enc = _get_encoding(model)
        if enc is None:
            # Fallback: rough estimate if tiktoken not installed
            return len(text) // 4 + 1
        return len(enc.encode(text))

    def count_messages(self, messages: list[dict[str, Any]], model: str) -> int:
        """Count tokens for a message array.

        Sums up token counts for each message's content plus a small overhead
        per message for role/formatting tokens.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Model identifier.

        Returns:
            Estimated total token count.
        """
        total = 0
        for msg in messages:
            # ~4 tokens overhead per message (role, separators)
            total += 4
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count(content, model)
            elif isinstance(content, list):
                # Multi-part content (text + media blocks)
                for part in content:
                    if isinstance(part, dict):
                        text = part.get("text", "")
                        if text:
                            total += self.count(text, model)
                        # Images/audio: rough fixed estimate per block
                        if part.get("type") in ("image_url", "input_audio"):
                            total += 1000
                    elif isinstance(part, str):
                        total += self.count(part, model)

            # Tool calls in assistant messages
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                total += 4  # overhead for tool call structure
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", "")
                total += self.count(name, model)
                if isinstance(args, str):
                    total += self.count(args, model)
                elif isinstance(args, dict):
                    total += self.count(json.dumps(args), model)

            # Tool result content
            if msg.get("role") == "tool":
                tool_content = msg.get("content", "")
                if isinstance(tool_content, str):
                    total += self.count(tool_content, model)

        return total

    def get_context_window(self, model: str, override: int | None = None) -> int:
        """Get the context window size for a model.

        Args:
            model: Model identifier (e.g. 'anthropic/claude-sonnet-4-5').
            override: Optional explicit context window size from config.

        Returns:
            Context window size in tokens.
        """
        if override is not None and override > 0:
            return override

        bare_model = _strip_provider(model).lower()

        # Try exact match first
        if bare_model in CONTEXT_WINDOWS:
            return CONTEXT_WINDOWS[bare_model]

        # Try prefix match (e.g. "claude-3.5-sonnet-20240620" matches "claude-3.5-sonnet")
        for known, size in CONTEXT_WINDOWS.items():
            if bare_model.startswith(known):
                return size

        return DEFAULT_CONTEXT_WINDOW

    def get_context_usage(
        self, messages: list[dict[str, Any]], model: str, context_window: int | None = None
    ) -> tuple[int, int]:
        """Get current token usage vs context limit.

        Args:
            messages: Current message array.
            model: Model identifier.
            context_window: Optional explicit context window override.

        Returns:
            Tuple of (tokens_used, context_limit).
        """
        used = self.count_messages(messages, model)
        limit = self.get_context_window(model, context_window)
        return used, limit

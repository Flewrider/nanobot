"""Automatic context compaction to stay within token limits."""

from typing import Any

from loguru import logger

from nanobot.agent.tokens import TokenCounter
from nanobot.providers.base import LLMProvider


COMPACTION_PROMPT = (
    "Summarize this conversation concisely. Preserve:\n"
    "- Key decisions made\n"
    "- Important facts and file paths\n"
    "- Current task state and goals\n"
    "- Any errors or issues encountered\n"
    "Format as a brief narrative, not a list."
)

# Map provider prefixes to cheaper models for compaction
_COMPACTION_MODELS: dict[str, str] = {
    "openai": "openai/gpt-4o-mini",
    "anthropic": "anthropic/claude-3-haiku-20240307",
    "gemini": "gemini/gemini-2.0-flash-lite",
    "deepseek": "deepseek/deepseek-chat",
    "kimi": "kimi/kimi-k2",
}
_DEFAULT_COMPACTION_MODEL = "openai/gpt-4o-mini"


class Compactor:
    """Automatically compacts conversation context when approaching token limits."""

    def __init__(
        self,
        provider: LLMProvider,
        token_counter: TokenCounter,
        threshold: float = 0.75,
        keep_recent: int = 6,
    ):
        self.provider = provider
        self.token_counter = token_counter
        self.threshold = threshold  # Trigger compaction at 75% of context window
        self.keep_recent = keep_recent  # Keep last N turns verbatim

    async def should_compact(self, messages: list[dict[str, Any]], model: str) -> bool:
        """Check if context exceeds threshold."""
        used = self.token_counter.count_messages(messages, model)
        limit = self.token_counter.get_context_window(model)
        return used >= limit * self.threshold

    async def compact(
        self,
        messages: list[dict[str, Any]],
        model: str,
        memory_manager: Any = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Compact messages by summarizing older turns with a cheaper model.

        Args:
            messages: Full message list (system + conversation).
            model: The main model identifier (used to pick compaction model).
            memory_manager: Optional MemoryManager; if provided, key facts from
                older turns are appended to today's daily notes before compacting.

        Returns:
            Tuple of (compacted_messages, metadata) where metadata contains
            tokens_before, tokens_after, and turns_removed.
        """
        tokens_before = self.token_counter.count_messages(messages, model)

        system_msgs, older, recent = self._split_messages(messages)

        if not older:
            return messages, {
                "tokens_before": tokens_before,
                "tokens_after": tokens_before,
                "turns_removed": 0,
            }

        # Optionally flush key facts to memory before discarding older turns
        if memory_manager is not None:
            try:
                self._flush_to_memory(older, memory_manager)
            except Exception as exc:
                logger.warning(f"Compactor: failed to flush memory: {exc}")

        # Build a temporary conversation for the compaction model
        summary_messages = [
            {"role": "system", "content": COMPACTION_PROMPT},
            {"role": "user", "content": self._render_turns(older)},
        ]

        compaction_model = self._get_compaction_model(model)

        try:
            response = await self.provider.chat(
                messages=summary_messages,
                model=compaction_model,
                max_tokens=1024,
                temperature=0.3,
            )
            summary_text = response.content or "[Compaction failed - no summary returned]"
        except Exception as exc:
            logger.error(f"Compactor: LLM call failed: {exc}")
            summary_text = self._fallback_summary(older)

        # Reassemble: system + summary + recent
        compacted: list[dict[str, Any]] = list(system_msgs)
        compacted.append({
            "role": "assistant",
            "content": f"[Context compacted - summary of earlier conversation]\n\n{summary_text}",
        })
        compacted.extend(recent)

        tokens_after = self.token_counter.count_messages(compacted, model)
        turns_removed = len(older)

        logger.info(
            f"Compacted context: {tokens_before} -> {tokens_after} tokens "
            f"({turns_removed} messages summarized)"
        )

        return compacted, {
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "turns_removed": turns_removed,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_compaction_model(self, model: str) -> str:
        """Select cheaper model for compaction based on current provider prefix."""
        if "/" in model:
            prefix = model.split("/", 1)[0].lower()
        else:
            prefix = ""
        return _COMPACTION_MODELS.get(prefix, _DEFAULT_COMPACTION_MODEL)

    def _split_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Split into (system_messages, older_turns, recent_turns).

        System messages are always kept. The last ``keep_recent`` non-system
        messages are preserved verbatim; everything in between is the "older"
        portion that will be summarized.
        """
        system_msgs: list[dict[str, Any]] = []
        conversation: list[dict[str, Any]] = []

        for msg in messages:
            if msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                conversation.append(msg)

        if len(conversation) <= self.keep_recent:
            return system_msgs, [], conversation

        split_idx = len(conversation) - self.keep_recent
        older = conversation[:split_idx]
        recent = conversation[split_idx:]
        return system_msgs, older, recent

    @staticmethod
    def _render_turns(messages: list[dict[str, Any]]) -> str:
        """Render a list of messages into a readable text block for summarization."""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Multi-part content: extract text parts only
                text_bits = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_bits.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_bits.append(part)
                content = "\n".join(text_bits)
            # Truncate very long individual messages for the summary request
            if len(content) > 2000:
                content = content[:1000] + "\n[...truncated...]\n" + content[-500:]
            parts.append(f"[{role}]: {content}")
        return "\n\n".join(parts)

    @staticmethod
    def _fallback_summary(messages: list[dict[str, Any]]) -> str:
        """Create a basic summary without LLM when the compaction call fails."""
        roles = {"user": 0, "assistant": 0, "tool": 0}
        for msg in messages:
            role = msg.get("role", "other")
            roles[role] = roles.get(role, 0) + 1
        return (
            f"[Earlier conversation with {roles['user']} user messages, "
            f"{roles['assistant']} assistant messages, and "
            f"{roles['tool']} tool results was compacted but summarization failed.]"
        )

    @staticmethod
    def _flush_to_memory(messages: list[dict[str, Any]], memory_manager: Any) -> None:
        """Extract key facts from older turns and append to daily notes."""
        facts: list[str] = []
        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            role = msg.get("role", "")
            # Only extract from assistant messages that contain substance
            if role == "assistant" and len(content) > 50:
                # Take first 200 chars as a fact snippet
                snippet = content[:200].strip()
                if snippet:
                    facts.append(f"- {snippet}")

        if facts:
            entry = "\n## Compacted Context\n" + "\n".join(facts[:10])
            memory_manager.append_daily(entry)

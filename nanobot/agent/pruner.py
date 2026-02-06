"""Dynamic Context Pruner: prunes stale tool outputs to save tokens."""

import re
from typing import Any

from loguru import logger


class ContextPruner:
    """Prunes stale tool outputs from conversation context to save tokens."""

    # Tool names mapped to their staleness threshold (turns before pruning)
    STALE_THRESHOLDS: dict[str, int] = {
        "read_file": 3,
        "web_search": 2,
        "web_fetch": 2,
        "exec": 3,
        "exec_command": 3,
        "list_dir": 2,
        "list_files": 2,
    }

    # Max size before truncation (even for non-stale outputs)
    LARGE_OUTPUT_THRESHOLD = 2000
    TRUNCATE_HEAD = 500
    TRUNCATE_TAIL = 200

    def __init__(self, keep_recent: int = 3):
        self.keep_recent = keep_recent  # Keep last N results per tool type verbatim

    def prune(self, messages: list[dict[str, Any]], current_turn: int) -> list[dict[str, Any]]:
        """Prune stale tool outputs from message list. Returns pruned copy."""
        # Count tool results per tool type (from end) to know which are "recent"
        recent_counts: dict[str, int] = {}
        # Track which tool_call_ids are recent (scanning from end)
        recent_ids: set[str] = set()

        for msg in reversed(messages):
            if msg.get("role") != "tool":
                continue
            tool_name = msg.get("name", "")
            canonical = self._canonical_name(tool_name)
            if canonical not in self.STALE_THRESHOLDS:
                continue
            count = recent_counts.get(canonical, 0)
            if count < self.keep_recent:
                recent_ids.add(msg.get("tool_call_id", ""))
                recent_counts[canonical] = count + 1

        # Now build pruned message list
        pruned = []
        turn = 0
        messages_pruned = 0
        chars_saved = 0

        for msg in messages:
            if msg.get("role") == "user":
                turn += 1

            if msg.get("role") != "tool":
                pruned.append(msg)
                continue

            tool_name = msg.get("name", "")
            tool_call_id = msg.get("tool_call_id", "")
            content = msg.get("content", "")
            canonical = self._canonical_name(tool_name)

            # Not a prunable tool type - keep as-is
            if canonical not in self.STALE_THRESHOLDS:
                pruned.append(msg)
                continue

            # Recent results are always kept fully verbatim - the agent
            # needs full tool output for at least the next few turns.
            if tool_call_id in recent_ids:
                pruned.append(msg)
                continue

            # Check staleness
            turn_age = current_turn - turn
            threshold = self.STALE_THRESHOLDS[canonical]

            if turn_age >= threshold:
                summary = self._create_summary(canonical, tool_call_id, content)
                chars_saved += len(content) - len(summary)
                messages_pruned += 1
                pruned.append({**msg, "content": summary})
            elif turn_age >= 1 and len(content) > self.LARGE_OUTPUT_THRESHOLD:
                # Only truncate large outputs after at least 1 turn has passed
                truncated = self._truncate(content)
                chars_saved += len(content) - len(truncated)
                pruned.append({**msg, "content": truncated})
            else:
                pruned.append(msg)

        if messages_pruned > 0 or chars_saved > 0:
            est_tokens = chars_saved // 4
            logger.debug(
                f"Context pruner: {messages_pruned} messages pruned, "
                f"~{est_tokens} tokens saved ({chars_saved} chars)"
            )

        return pruned

    def _canonical_name(self, tool_name: str) -> str:
        """Map tool name aliases to canonical names."""
        aliases = {
            "exec_command": "exec",
            "list_files": "list_dir",
        }
        return aliases.get(tool_name, tool_name)

    def _should_prune(self, tool_name: str, turn_age: int) -> bool:
        """Whether a tool result is stale enough to prune."""
        canonical = self._canonical_name(tool_name)
        threshold = self.STALE_THRESHOLDS.get(canonical)
        if threshold is None:
            return False
        return turn_age >= threshold

    def _create_summary(self, tool_name: str, tool_call_id: str, content: str) -> str:
        """Create a brief summary of pruned tool output."""
        canonical = self._canonical_name(tool_name)

        if canonical == "read_file":
            path = self._extract_path(content)
            line_count = content.count("\n") + 1
            return f"[Previously read: {path} - {line_count} lines]"

        if canonical in ("web_search", "web_fetch"):
            query = self._extract_query(content)
            result_count = self._count_results(content)
            label = "Search" if canonical == "web_search" else "Fetch"
            return f"[{label}: {query} - {result_count} results]"

        if canonical == "exec":
            command = self._extract_command(content)
            return f"[Shell: {command} - {len(content)} chars output]"

        if canonical == "list_dir":
            path = self._extract_list_path(content)
            file_count = self._count_items(content)
            return f"[Listed: {path} - {file_count} items]"

        return f"[Pruned {tool_name} output - {len(content)} chars]"

    def _truncate(self, content: str) -> str:
        """Truncate large content keeping head and tail."""
        head = content[: self.TRUNCATE_HEAD]
        tail = content[-self.TRUNCATE_TAIL :]
        omitted = len(content) - self.TRUNCATE_HEAD - self.TRUNCATE_TAIL
        return f"{head}\n[...truncated {omitted} chars...]\n{tail}"

    def _extract_path(self, content: str) -> str:
        """Try to extract a file path from read_file output."""
        # read_file just returns raw content, so we can't extract the path from it.
        # But tool results are often prefixed - check first line for path-like strings
        first_line = content.split("\n", 1)[0] if content else ""
        # Look for path patterns
        match = re.search(r'[/\\][\w./\\-]+\.\w+', first_line)
        if match:
            return match.group(0)
        # Truncate first line as fallback
        return first_line[:80] if first_line else "unknown file"

    def _extract_query(self, content: str) -> str:
        """Extract search query or URL from web tool output."""
        # web_search format: "Results for: {query}\n..."
        match = re.search(r'Results for:\s*(.+)', content)
        if match:
            return match.group(1).strip()[:80]
        # web_fetch returns JSON with "url" key
        match = re.search(r'"url"\s*:\s*"([^"]+)"', content)
        if match:
            return match.group(1)[:80]
        return content[:60].strip()

    def _count_results(self, content: str) -> int:
        """Count search results or estimate content sections."""
        # web_search numbers results as "1. ...", "2. ...", etc.
        numbered = re.findall(r'^\d+\.', content, re.MULTILINE)
        if numbered:
            return len(numbered)
        return 1

    def _extract_command(self, content: str) -> str:
        """Extract shell command from exec output. Best-effort."""
        # exec tool returns raw stdout/stderr, command is not in output.
        # Use first line as a hint
        first_line = content.split("\n", 1)[0] if content else ""
        return first_line[:80] if first_line else "command"

    def _extract_list_path(self, content: str) -> str:
        """Extract directory path from list_dir output."""
        # list_dir format: lines of "icon name"
        # The path isn't in the output itself, so we note item count
        match = re.search(r'Directory (.+) is empty', content)
        if match:
            return match.group(1)
        return "directory"

    def _count_items(self, content: str) -> int:
        """Count items in a directory listing."""
        if not content:
            return 0
        return len([line for line in content.split("\n") if line.strip()])

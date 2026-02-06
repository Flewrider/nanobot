"""Memory system for persistent agent memory.

Supports three layers:
- OVERVIEW.md: Auto-generated topic index (always loaded in context)
- topics/: Individual topic files for organized long-term memory
- daily/: Daily notes in YYYY-MM-DD.md format
"""

import logging
import re
from pathlib import Path
from datetime import datetime, timedelta

from nanobot.utils.helpers import ensure_dir, today_date

log = logging.getLogger(__name__)


class MemoryManager:
    """
    Memory system with OVERVIEW, topics, and daily notes.

    Directory structure:
        workspace/memory/
        ├── OVERVIEW.md
        ├── topics/
        │   ├── project-nanobot.md
        │   └── user-preferences.md
        └── daily/
            ├── 2026-02-05.md
            └── 2026-02-04.md
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory_dir = ensure_dir(workspace / "memory")
        self.topics_dir = ensure_dir(self.memory_dir / "topics")
        self.daily_dir = ensure_dir(self.memory_dir / "daily")
        self.overview_file = self.memory_dir / "OVERVIEW.md"
        self._migrate_if_needed()

    # --- Overview ---

    def read_overview(self) -> str:
        """Read OVERVIEW.md content."""
        if self.overview_file.exists():
            return self.overview_file.read_text(encoding="utf-8")
        return ""

    def update_overview(self) -> None:
        """Auto-regenerate OVERVIEW.md by scanning topic files."""
        topics = self.list_topics()
        if not topics:
            self.overview_file.write_text(
                "# Memory Overview\n\nNo topics yet.\n", encoding="utf-8"
            )
            return

        lines = ["# Memory Overview\n"]
        for name in sorted(topics):
            content = self.read_topic(name)
            summary = self._extract_summary(content)
            display_name = name.replace("-", " ").title()
            lines.append(f"- **[{display_name}](topics/{name}.md)**: {summary}")

        lines.append("")  # trailing newline
        self.overview_file.write_text("\n".join(lines), encoding="utf-8")

    # --- Topics ---

    def create_topic(self, name: str, content: str) -> None:
        """Create a new topic file at topics/{name}.md."""
        name = self._sanitize_topic_name(name)
        path = self.topics_dir / f"{name}.md"
        path.write_text(content, encoding="utf-8")
        self.update_overview()

    def read_topic(self, name: str) -> str:
        """Read a topic file. Returns empty string if not found."""
        name = self._sanitize_topic_name(name)
        path = self.topics_dir / f"{name}.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def update_topic(self, name: str, content: str) -> None:
        """Update an existing topic file (or create it)."""
        name = self._sanitize_topic_name(name)
        path = self.topics_dir / f"{name}.md"
        path.write_text(content, encoding="utf-8")
        self.update_overview()

    def list_topics(self) -> list[str]:
        """List all topic names (without .md extension)."""
        if not self.topics_dir.exists():
            return []
        return sorted(p.stem for p in self.topics_dir.glob("*.md"))

    # --- Daily Notes ---

    def append_daily(self, content: str) -> None:
        """Append content to today's daily note."""
        today_file = self.daily_dir / f"{today_date()}.md"

        if today_file.exists():
            existing = today_file.read_text(encoding="utf-8")
            content = existing + "\n" + content
        else:
            header = f"# {today_date()}\n\n"
            content = header + content

        today_file.write_text(content, encoding="utf-8")

    def read_daily(self, date: str | None = None) -> str:
        """Read a specific day's notes, or today if no date given."""
        date = date or today_date()
        path = self.daily_dir / f"{date}.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def get_recent_daily(self, days: int = 7) -> list[tuple[str, str]]:
        """Get recent daily notes as list of (date, content) tuples."""
        results = []
        today = datetime.now().date()

        for i in range(days):
            date = today - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            path = self.daily_dir / f"{date_str}.md"
            if path.exists():
                content = path.read_text(encoding="utf-8")
                results.append((date_str, content))

        return results

    # --- Context for System Prompt ---

    def get_memory_context(self) -> str:
        """Return OVERVIEW.md content for inclusion in system prompt."""
        overview = self.read_overview()
        parts = []

        if overview:
            parts.append("## Memory Overview\n" + overview)

        # Include today's notes if any
        today = self.read_daily()
        if today:
            parts.append("## Today's Notes\n" + today)

        return "\n\n".join(parts) if parts else ""

    # --- Migration ---

    def _migrate_if_needed(self) -> None:
        """Migrate from old MemoryStore format if needed."""
        old_memory_file = self.memory_dir / "MEMORY.md"

        # Migrate MEMORY.md -> topics/general.md
        if old_memory_file.exists():
            content = old_memory_file.read_text(encoding="utf-8")
            if content.strip():
                general_path = self.topics_dir / "general.md"
                if not general_path.exists():
                    log.info("Migrating MEMORY.md -> topics/general.md")
                    general_path.write_text(content, encoding="utf-8")
                old_memory_file.rename(
                    old_memory_file.with_suffix(".md.bak")
                )

        # Migrate daily files from memory/*.md to memory/daily/*.md
        for f in self.memory_dir.glob("????-??-??.md"):
            if f.parent == self.memory_dir:  # only top-level dated files
                dest = self.daily_dir / f.name
                if not dest.exists():
                    log.info("Migrating %s -> daily/%s", f.name, f.name)
                    f.rename(dest)
                else:
                    # Already exists in daily/, remove the old one
                    f.unlink()

        # Regenerate overview after migration
        if self.list_topics():
            self.update_overview()

    # --- Backward Compatibility ---

    def read_long_term(self) -> str:
        """Backward compat: returns overview content."""
        return self.read_overview()

    def write_long_term(self, content: str) -> None:
        """Backward compat: writes to topics/general.md and updates overview."""
        self.update_topic("general", content)

    def append_today(self, content: str) -> None:
        """Backward compat: alias for append_daily."""
        self.append_daily(content)

    def read_today(self) -> str:
        """Backward compat: alias for read_daily (today)."""
        return self.read_daily()

    def get_recent_memories(self, days: int = 7) -> str:
        """Backward compat: get recent daily notes as combined string."""
        entries = self.get_recent_daily(days)
        return "\n\n---\n\n".join(content for _, content in entries)

    def list_memory_files(self) -> list[Path]:
        """Backward compat: list daily note files sorted newest first."""
        if not self.daily_dir.exists():
            return []
        files = list(self.daily_dir.glob("????-??-??.md"))
        return sorted(files, reverse=True)

    # --- Helpers ---

    @staticmethod
    def _sanitize_topic_name(name: str) -> str:
        """Sanitize a topic name into a safe filename slug."""
        name = name.lower().strip()
        name = re.sub(r"[^a-z0-9\-]", "-", name)
        name = re.sub(r"-+", "-", name)
        return name.strip("-")

    @staticmethod
    def _extract_summary(content: str) -> str:
        """Extract summary from first non-heading, non-empty line(s)."""
        if not content:
            return "(empty)"
        lines = content.strip().splitlines()
        for line in lines:
            stripped = line.strip()
            # Skip headings and empty lines
            if not stripped or stripped.startswith("#"):
                continue
            # Return first meaningful line, truncated
            if len(stripped) > 120:
                return stripped[:117] + "..."
            return stripped
        return "(empty)"


# Backward compatibility alias
MemoryStore = MemoryManager

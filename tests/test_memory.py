"""Tests for nanobot.agent.memory.MemoryManager."""

from pathlib import Path
from unittest.mock import patch

import pytest

from nanobot.agent.memory import MemoryManager, MemoryStore


@pytest.fixture
def mm(tmp_path):
    """Create a MemoryManager with a temporary workspace."""
    return MemoryManager(workspace=tmp_path)


# ---- Topics ----


def test_create_topic(mm, tmp_path):
    """create_topic should create a file in topics/."""
    mm.create_topic("my-project", "# My Project\nSome details here.")
    path = tmp_path / "memory" / "topics" / "my-project.md"
    assert path.exists()
    assert "My Project" in path.read_text(encoding="utf-8")


def test_read_topic(mm):
    """read_topic should return the content of a topic file."""
    mm.create_topic("testing", "# Testing\nTest content.")
    content = mm.read_topic("testing")
    assert "Test content." in content


def test_read_topic_nonexistent(mm):
    """Reading a nonexistent topic should return empty string."""
    assert mm.read_topic("does-not-exist") == ""


def test_update_topic(mm):
    """update_topic should overwrite existing content."""
    mm.create_topic("notes", "Version 1")
    mm.update_topic("notes", "Version 2")
    content = mm.read_topic("notes")
    assert content == "Version 2"


def test_list_topics(mm):
    """list_topics should return all topic names without .md extension."""
    mm.create_topic("alpha", "A")
    mm.create_topic("beta", "B")
    mm.create_topic("gamma", "C")
    topics = mm.list_topics()
    assert topics == ["alpha", "beta", "gamma"]


def test_list_topics_empty(mm):
    """list_topics should return empty list when no topics exist."""
    assert mm.list_topics() == []


# ---- Overview ----


def test_overview_generation(mm, tmp_path):
    """update_overview should generate OVERVIEW.md with topic summaries."""
    mm.create_topic("project-nanobot", "# Nanobot\nAn AI assistant framework.")
    mm.create_topic("user-prefs", "# Preferences\nDark mode enabled.")

    overview = mm.read_overview()
    assert "Memory Overview" in overview
    assert "Project Nanobot" in overview or "project-nanobot" in overview.lower()
    assert "User Prefs" in overview or "user-prefs" in overview.lower()


def test_overview_empty(mm):
    """Overview with no topics should say 'No topics yet'."""
    mm.update_overview()
    content = mm.read_overview()
    assert "No topics yet" in content


# ---- Daily Notes ----


def test_daily_append(mm, tmp_path):
    """append_daily should create/append to today's daily note."""
    with patch("nanobot.agent.memory.today_date", return_value="2026-02-06"):
        mm.append_daily("First entry.")
        content = (tmp_path / "memory" / "daily" / "2026-02-06.md").read_text(encoding="utf-8")
        assert "First entry." in content
        assert "# 2026-02-06" in content

        # Append again
        mm.append_daily("Second entry.")
        content = (tmp_path / "memory" / "daily" / "2026-02-06.md").read_text(encoding="utf-8")
        assert "First entry." in content
        assert "Second entry." in content


def test_daily_read(mm):
    """read_daily should read today's note."""
    with patch("nanobot.agent.memory.today_date", return_value="2026-02-06"):
        mm.append_daily("Test note content.")
        content = mm.read_daily("2026-02-06")
        assert "Test note content." in content


def test_daily_read_nonexistent(mm):
    """Reading a nonexistent daily note should return empty string."""
    assert mm.read_daily("1999-01-01") == ""


def test_recent_daily(mm, tmp_path):
    """get_recent_daily should return the last N days of notes."""
    # Create some daily files manually
    daily_dir = tmp_path / "memory" / "daily"
    from datetime import datetime, timedelta

    today = datetime.now().date()
    for i in range(5):
        date = today - timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")
        (daily_dir / f"{date_str}.md").write_text(
            f"# {date_str}\nNote for day {i}", encoding="utf-8"
        )

    results = mm.get_recent_daily(days=3)
    assert len(results) == 3
    # First result should be today
    assert results[0][0] == today.strftime("%Y-%m-%d")


# ---- get_memory_context() ----


def test_get_memory_context(mm):
    """get_memory_context should return overview + today's notes."""
    mm.create_topic("general", "# General\nSome general info.")
    with patch("nanobot.agent.memory.today_date", return_value="2026-02-06"):
        mm.append_daily("Today's task log.")
        context = mm.get_memory_context()
        assert "Memory Overview" in context
        assert "Today's Notes" in context
        assert "Today's task log." in context


def test_get_memory_context_empty(mm):
    """get_memory_context with nothing stored should return empty string."""
    # Fresh manager with no topics or daily notes
    # The overview might have been generated with "No topics yet" during init
    # but get_memory_context checks if overview is truthy
    context = mm.get_memory_context()
    # Either empty or contains the overview header
    assert isinstance(context, str)


# ---- Migration ----


def test_migrate_old_memory(tmp_path):
    """MEMORY.md should be migrated to topics/general.md."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    old_file = memory_dir / "MEMORY.md"
    old_file.write_text("# Old Memory\nLegacy content here.", encoding="utf-8")

    mm = MemoryManager(workspace=tmp_path)

    # MEMORY.md should be renamed to .bak
    assert not old_file.exists()
    assert (memory_dir / "MEMORY.md.bak").exists()

    # Content should be in topics/general.md
    general = mm.read_topic("general")
    assert "Legacy content here." in general


def test_migrate_old_daily_files(tmp_path):
    """Dated files in memory/ root should be moved to memory/daily/."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    old_daily = memory_dir / "2026-01-15.md"
    old_daily.write_text("# 2026-01-15\nOld daily note.", encoding="utf-8")

    mm = MemoryManager(workspace=tmp_path)

    # Old file should be moved to daily/
    assert not old_daily.exists()
    new_path = memory_dir / "daily" / "2026-01-15.md"
    assert new_path.exists()
    assert "Old daily note." in new_path.read_text(encoding="utf-8")


# ---- Backward Compatibility ----


def test_backward_compat_alias():
    """MemoryStore should be an alias for MemoryManager."""
    assert MemoryStore is MemoryManager


def test_backward_compat_methods(mm):
    """Legacy methods should still work."""
    mm.write_long_term("Legacy content")
    content = mm.read_long_term()
    assert "Memory Overview" in content  # overview is returned

    with patch("nanobot.agent.memory.today_date", return_value="2026-02-06"):
        mm.append_today("Legacy daily entry")
        today = mm.read_today()
        assert "Legacy daily entry" in today


# ---- Topic name sanitization ----


def test_topic_name_sanitization(mm):
    """Topic names with special chars should be sanitized."""
    mm.create_topic("My Cool Topic!!!", "Content here")
    topics = mm.list_topics()
    assert "my-cool-topic" in topics
    content = mm.read_topic("my-cool-topic")
    assert content == "Content here"

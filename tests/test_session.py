"""Tests for nanobot.session.manager.SessionManager and Session."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from nanobot.session.manager import Session, SessionManager


@pytest.fixture
def manager(tmp_path):
    """Create a SessionManager with a temporary workspace and sessions dir."""
    mgr = SessionManager(workspace=tmp_path)
    # Override sessions_dir to use tmp_path instead of ~/.nanobot/sessions
    mgr.sessions_dir = tmp_path / "sessions"
    mgr.sessions_dir.mkdir(parents=True, exist_ok=True)
    return mgr


# ---- Session dataclass ----


def test_session_fields():
    """Session should have the expected default fields."""
    session = Session(key="test:123")
    assert session.key == "test:123"
    assert session.messages == []
    assert session.token_count == 0
    assert session.compaction_count == 0
    assert isinstance(session.created_at, datetime)
    assert isinstance(session.last_accessed, datetime)


def test_session_add_message():
    """add_message should append and update timestamps."""
    session = Session(key="test:1")
    before = session.updated_at
    session.add_message("user", "Hello")
    assert len(session.messages) == 1
    assert session.messages[0]["role"] == "user"
    assert session.messages[0]["content"] == "Hello"
    assert session.updated_at >= before


def test_session_processing_metadata():
    """mark_processing / is_processing / clear_processing / get_interrupted_task."""
    session = Session(key="test:1")
    assert session.is_processing is False
    assert session.get_interrupted_task() is None

    session.mark_processing("working on something", "tool: read_file")
    assert session.is_processing is True
    assert session.get_interrupted_task() == "working on something"

    session.clear_processing()
    assert session.is_processing is False
    assert session.get_interrupted_task() is None


# ---- SessionManager.get_or_create() ----


def test_get_or_create_new(manager):
    """get_or_create should create a new session when none exists."""
    session = manager.get_or_create("telegram:123")
    assert session.key == "telegram:123"
    assert session.messages == []


def test_get_or_create_cached(manager):
    """Second call should return the same cached session."""
    session1 = manager.get_or_create("telegram:456")
    session1.add_message("user", "test")
    session2 = manager.get_or_create("telegram:456")
    assert session2 is session1
    assert len(session2.messages) == 1


# ---- save() and reload ----


def test_save_and_reload(manager):
    """Save to JSONL, evict from memory, reload from disk."""
    session = manager.get_or_create("telegram:789")
    session.add_message("user", "Hello!")
    session.add_message("assistant", "Hi there!")
    session.token_count = 42
    session.compaction_count = 1

    manager.save(session)

    # Verify JSONL file was created
    path = manager._get_session_path("telegram:789")
    assert path.exists()

    # Evict from memory
    manager._sessions.clear()
    assert "telegram:789" not in manager._sessions

    # Reload via get_or_create
    reloaded = manager.get_or_create("telegram:789")
    assert reloaded.key == "telegram:789"
    assert len(reloaded.messages) == 2
    assert reloaded.messages[0]["role"] == "user"
    assert reloaded.messages[0]["content"] == "Hello!"
    assert reloaded.token_count == 42
    assert reloaded.compaction_count == 1


# ---- evict_stale() ----


def test_evict_stale(manager):
    """Old sessions should be removed from memory."""
    session = manager.get_or_create("old:1")
    # Make session appear stale
    session.last_accessed = datetime.now() - timedelta(seconds=3600)
    manager._sessions["old:1"] = session

    evicted = manager.evict_stale(ttl_seconds=1800)
    assert "old:1" in evicted
    assert "old:1" not in manager._sessions


def test_evict_preserves_recent(manager):
    """Recently accessed sessions should not be evicted."""
    session = manager.get_or_create("recent:1")
    session.last_accessed = datetime.now()  # just accessed
    manager._sessions["recent:1"] = session

    evicted = manager.evict_stale(ttl_seconds=1800)
    assert "recent:1" not in evicted
    assert "recent:1" in manager._sessions


# ---- get_all_active() ----


def test_get_all_active(manager):
    """get_all_active should return all in-memory sessions."""
    manager.get_or_create("a:1")
    manager.get_or_create("b:2")
    active = manager.get_all_active()
    keys = {s.key for s in active}
    assert "a:1" in keys
    assert "b:2" in keys


# ---- delete() ----


def test_delete_session(manager):
    """delete should remove session from memory and disk."""
    session = manager.get_or_create("del:1")
    manager.save(session)

    path = manager._get_session_path("del:1")
    assert path.exists()

    result = manager.delete("del:1")
    assert result is True
    assert "del:1" not in manager._sessions
    assert not path.exists()


def test_delete_nonexistent(manager):
    """Deleting a nonexistent session should return False."""
    result = manager.delete("nonexistent:999")
    assert result is False


# ---- interrupted sessions ----


def test_interrupted_sessions(manager):
    """Sessions with processing metadata should be detected."""
    session = manager.get_or_create("intr:1")
    session.mark_processing("doing task X", "tool: exec")
    manager.save(session)

    interrupted = manager.get_interrupted_sessions()
    # Should find the session with processing metadata
    keys = [k for k, _, _ in interrupted]
    assert any("intr" in k for k in keys)


# ---- _cache backward compatibility ----


def test_cache_alias(manager):
    """_cache property should be an alias for _sessions."""
    manager.get_or_create("alias:1")
    assert "alias:1" in manager._cache
    assert manager._cache is manager._sessions

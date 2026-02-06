"""Session management for conversation history."""

import asyncio
import json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger

from nanobot.utils.helpers import ensure_dir, safe_filename


@dataclass
class Session:
    """
    A conversation session.

    Stores messages in JSONL format for easy reading and persistence.
    """

    key: str  # channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    compaction_count: int = 0

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        msg = {"role": role, "content": content, "timestamp": datetime.now().isoformat(), **kwargs}
        self.messages.append(msg)
        now = datetime.now()
        self.updated_at = now
        self.last_accessed = now

    def mark_processing(self, content: str, tool_context: str = "") -> None:
        """Mark that we're processing a user message (for crash recovery)."""
        self.metadata["processing"] = {
            "started_at": datetime.now().isoformat(),
            "content": content[:200],  # Store truncated content for context
            "last_tools": tool_context[:500],  # Recent tool activity for better recovery
        }

    @property
    def is_processing(self) -> bool:
        """Return True if the session is currently processing a message."""
        return "processing" in self.metadata

    def clear_processing(self) -> None:
        """Clear the processing flag (task completed successfully)."""
        self.metadata.pop("processing", None)

    def get_interrupted_task(self) -> str | None:
        """Check if there was an interrupted task. Returns the content or None."""
        processing = self.metadata.get("processing")
        if processing:
            return processing.get("content")
        return None

    def get_history(self, max_messages: int = 15) -> list[dict[str, Any]]:
        """
        Get message history for LLM context.

        Args:
            max_messages: Maximum messages to return.

        Returns:
            List of messages in LLM format.
        """
        # Get recent messages
        recent = (
            self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        )

        # Convert to LLM format (just role and content)
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    def clear(self) -> None:
        """Clear all messages in the session."""
        self.messages = []
        self.updated_at = datetime.now()


class SessionManager:
    """
    Manages conversation sessions with in-memory-first storage.

    The in-memory ``_sessions`` dict is the primary data store.
    Every ``save()`` writes through to a JSONL file on disk as a durable backup.
    On ``get_or_create()``, memory is checked first; disk is only consulted on a
    cache miss.  Idle sessions can be evicted from memory via ``evict_stale()``
    while their JSONL files remain on disk for later reloading.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.sessions_dir = ensure_dir(Path.home() / ".nanobot" / "sessions")
        self._sessions: dict[str, Session] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # Backward-compatible alias so callers using ``._cache`` still work.
    # ------------------------------------------------------------------
    @property
    def _cache(self) -> dict[str, Session]:
        return self._sessions

    @_cache.setter
    def _cache(self, value: dict[str, Session]) -> None:
        self._sessions = value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_session_path(self, key: str) -> Path:
        """Get the file path for a session."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.sessions_dir / f"{safe_key}.jsonl"

    def _get_lock(self, key: str) -> asyncio.Lock:
        """Return (or create) a per-session asyncio.Lock."""
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get_or_create(self, key: str) -> Session:
        """
        Get an existing session or create a new one.

        Memory is checked first.  Only on a cache miss is the JSONL file
        consulted.  A brand-new ``Session`` is created when nothing exists
        on disk either.

        Args:
            key: Session key (usually channel:chat_id).

        Returns:
            The session.
        """
        # Evict stale sessions on each access (cheap scan)
        self.evict_stale()

        # Check in-memory store first
        if key in self._sessions:
            self._sessions[key].last_accessed = datetime.now()
            return self._sessions[key]

        # Cache miss -- try disk
        session = self._load(key)
        if session is None:
            session = Session(key=key)

        session.last_accessed = datetime.now()
        self._sessions[key] = session
        return session

    def _load(self, key: str) -> Session | None:
        """Load a session from disk."""
        path = self._get_session_path(key)

        if not path.exists():
            return None

        try:
            messages = []
            metadata = {}
            created_at = None
            token_count = 0
            compaction_count = 0

            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)

                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        created_at = (
                            datetime.fromisoformat(data["created_at"])
                            if data.get("created_at")
                            else None
                        )
                        token_count = data.get("token_count", 0)
                        compaction_count = data.get("compaction_count", 0)
                    else:
                        messages.append(data)

            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                metadata=metadata,
                token_count=token_count,
                compaction_count=compaction_count,
            )
        except Exception as e:
            logger.warning(f"Failed to load session {key}: {e}")
            return None

    def save(self, session: Session) -> None:
        """
        Save a session.

        The in-memory store is updated *and* the session is written through
        to its JSONL file on disk.
        """
        # Update in-memory store
        session.last_accessed = datetime.now()
        self._sessions[session.key] = session

        # Write-through to disk
        path = self._get_session_path(session.key)

        with open(path, "w") as f:
            # Write metadata first
            metadata_line = {
                "_type": "metadata",
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata,
                "token_count": session.token_count,
                "compaction_count": session.compaction_count,
            }
            f.write(json.dumps(metadata_line) + "\n")

            # Write messages
            for msg in session.messages:
                f.write(json.dumps(msg) + "\n")

    def delete(self, key: str) -> bool:
        """
        Delete a session.

        Args:
            key: Session key.

        Returns:
            True if deleted, False if not found.
        """
        # Remove from in-memory store
        self._sessions.pop(key, None)
        self._locks.pop(key, None)

        # Remove file
        path = self._get_session_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    # ------------------------------------------------------------------
    # Session listing / querying
    # ------------------------------------------------------------------

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions.

        Returns:
            List of session info dicts.
        """
        sessions = []

        for path in self.sessions_dir.glob("*.jsonl"):
            try:
                # Read just the metadata line
                with open(path) as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("_type") == "metadata":
                            sessions.append(
                                {
                                    "key": path.stem.replace("_", ":"),
                                    "created_at": data.get("created_at"),
                                    "updated_at": data.get("updated_at"),
                                    "path": str(path),
                                }
                            )
            except Exception:
                continue

        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)

    def get_interrupted_sessions(self) -> list[tuple[str, str, str]]:
        """
        Find sessions that were interrupted mid-processing (e.g., due to OOM/crash).

        Returns:
            List of (session_key, interrupted_content, last_tools) tuples.
        """
        interrupted = []

        for path in self.sessions_dir.glob("*.jsonl"):
            try:
                with open(path) as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("_type") == "metadata":
                            processing = data.get("metadata", {}).get("processing")
                            if processing:
                                key = path.stem.replace("_", ":", 1)  # telegram_123 -> telegram:123
                                content = processing.get("content", "continue")
                                last_tools = processing.get("last_tools", "")
                                interrupted.append((key, content, last_tools))
            except Exception:
                continue

        return interrupted

    def get_all_active(self) -> list[Session]:
        """Return all sessions currently held in the in-memory store."""
        return list(self._sessions.values())

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def evict_stale(self, ttl_seconds: int = 1800) -> list[str]:
        """
        Evict sessions that have been idle longer than *ttl_seconds* from
        the in-memory store.  Their JSONL files remain on disk and will be
        reloaded on the next ``get_or_create()`` call.

        Args:
            ttl_seconds: Maximum idle time in seconds (default 1800 = 30 min).

        Returns:
            List of evicted session keys.
        """
        now = datetime.now()
        evicted: list[str] = []

        for key in list(self._sessions):
            session = self._sessions[key]
            idle = (now - session.last_accessed).total_seconds()
            if idle > ttl_seconds:
                del self._sessions[key]
                self._locks.pop(key, None)
                evicted.append(key)
                logger.debug(f"Evicted stale session {key} (idle {idle:.0f}s)")

        return evicted

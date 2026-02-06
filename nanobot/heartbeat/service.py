"""Heartbeat service - periodic agent wake-up to check for tasks."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.bus.events import InboundMessage, MSG_SYSTEM_HEARTBEAT

if TYPE_CHECKING:
    from nanobot.bus.queue import MessageBus

# Default interval: 30 minutes
DEFAULT_HEARTBEAT_INTERVAL_S = 30 * 60

# The prompt sent to agent during heartbeat
HEARTBEAT_PROMPT = """Read HEARTBEAT.md in your workspace (if it exists).
Follow any instructions or tasks listed there.
Review the most recent chat context for any unfinished tasks and continue them.
If a task requires notifying the user, use the message tool. It will default to the most
recent chat if you omit channel/chat_id.
If nothing needs attention, reply with just: HEARTBEAT_OK"""

# Token that indicates "nothing to do"
HEARTBEAT_OK_TOKEN = "HEARTBEAT_OK"


def _is_heartbeat_empty(content: str | None) -> bool:
    """Check if HEARTBEAT.md has no actionable content."""
    if not content:
        return True

    # Lines to skip: empty, headers, HTML comments, empty checkboxes
    skip_patterns = {"- [ ]", "* [ ]", "- [x]", "* [x]"}

    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("<!--") or line in skip_patterns:
            continue
        return False  # Found actionable content

    return True


class HeartbeatService:
    """
    Periodic heartbeat service that wakes the agent to check for tasks.

    Publishes heartbeat messages to the MessageBus so they are processed
    through the same unified event loop as user messages.
    """

    def __init__(
        self,
        workspace: Path,
        bus: MessageBus,
        interval_s: int = DEFAULT_HEARTBEAT_INTERVAL_S,
        enabled: bool = True,
    ):
        self.workspace = workspace
        self.bus = bus
        self.interval_s = interval_s
        self.enabled = enabled
        self._running = False
        self._task: asyncio.Task | None = None

    @property
    def heartbeat_file(self) -> Path:
        return self.workspace / "HEARTBEAT.md"

    def _read_heartbeat_file(self) -> str | None:
        """Read HEARTBEAT.md content."""
        if self.heartbeat_file.exists():
            try:
                return self.heartbeat_file.read_text()
            except Exception:
                return None
        return None

    async def start(self) -> None:
        """Start the heartbeat service."""
        if not self.enabled:
            logger.info("Heartbeat disabled")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Heartbeat started (every {self.interval_s}s)")

    def stop(self) -> None:
        """Stop the heartbeat service."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def _run_loop(self) -> None:
        """Main heartbeat loop."""
        while self._running:
            try:
                await asyncio.sleep(self.interval_s)
                if self._running:
                    await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def _tick(self) -> None:
        """Execute a single heartbeat tick by publishing to the MessageBus."""
        content = self._read_heartbeat_file()

        # Skip if HEARTBEAT.md is empty or doesn't exist
        if _is_heartbeat_empty(content):
            logger.debug("Heartbeat: no tasks (HEARTBEAT.md empty)")
            return

        logger.info("Heartbeat: publishing to message bus...")

        msg = InboundMessage(
            channel="heartbeat",
            sender_id="heartbeat",
            chat_id="heartbeat",
            content=HEARTBEAT_PROMPT,
            message_type=MSG_SYSTEM_HEARTBEAT,
        )
        await self.bus.publish_inbound(msg)

    async def trigger_now(self) -> None:
        """Manually trigger a heartbeat."""
        msg = InboundMessage(
            channel="heartbeat",
            sender_id="heartbeat",
            chat_id="heartbeat",
            content=HEARTBEAT_PROMPT,
            message_type=MSG_SYSTEM_HEARTBEAT,
        )
        await self.bus.publish_inbound(msg)

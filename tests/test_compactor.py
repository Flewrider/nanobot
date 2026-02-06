"""Tests for nanobot.agent.compactor.Compactor."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.compactor import Compactor, _COMPACTION_MODELS, _DEFAULT_COMPACTION_MODEL
from nanobot.agent.tokens import TokenCounter
from nanobot.providers.base import LLMProvider, LLMResponse


@pytest.fixture
def token_counter():
    return TokenCounter()


@pytest.fixture
def mock_provider():
    provider = AsyncMock(spec=LLMProvider)
    provider.chat.return_value = LLMResponse(content="Summary of conversation.")
    return provider


@pytest.fixture
def compactor(mock_provider, token_counter):
    return Compactor(
        provider=mock_provider,
        token_counter=token_counter,
        threshold=0.75,
        keep_recent=4,
    )


def _make_messages(n_user: int, system: bool = True):
    """Helper to create a message list with optional system message."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": "You are a helpful assistant."})
    for i in range(n_user):
        msgs.append({"role": "user", "content": f"User message {i}"})
        msgs.append({"role": "assistant", "content": f"Assistant response {i}"})
    return msgs


# ---- should_compact() ----


async def test_should_compact_under_threshold(compactor):
    """Should return False when usage is under threshold."""
    messages = _make_messages(2)
    result = await compactor.should_compact(messages, "openai/gpt-4o")
    # A few short messages are nowhere near 75% of 128K tokens
    assert result is False


async def test_should_compact_over_threshold(compactor, token_counter):
    """Should return True when usage exceeds threshold."""
    # Create a compactor with a very low threshold to trigger easily
    provider = AsyncMock(spec=LLMProvider)
    low_compactor = Compactor(
        provider=provider,
        token_counter=token_counter,
        threshold=0.0001,  # effectively always triggers
        keep_recent=2,
    )
    messages = _make_messages(3)
    result = await low_compactor.should_compact(messages, "openai/gpt-4o")
    assert result is True


# ---- _get_compaction_model() ----


def test_get_compaction_model_openai(compactor):
    """OpenAI models should use gpt-4o-mini for compaction."""
    result = compactor._get_compaction_model("openai/gpt-4o")
    assert result == "openai/gpt-4o-mini"


def test_get_compaction_model_anthropic(compactor):
    """Anthropic models should use claude-3-haiku for compaction."""
    result = compactor._get_compaction_model("anthropic/claude-sonnet-4-5")
    assert result == "anthropic/claude-3-haiku-20240307"


def test_get_compaction_model_gemini(compactor):
    """Gemini models should use gemini-2.0-flash-lite for compaction."""
    result = compactor._get_compaction_model("gemini/gemini-2.5-pro")
    assert result == "gemini/gemini-2.0-flash-lite"


def test_get_compaction_model_unknown(compactor):
    """Unknown providers should use the default compaction model."""
    result = compactor._get_compaction_model("mystery-model-no-prefix")
    assert result == _DEFAULT_COMPACTION_MODEL


# ---- _split_messages() ----


def test_split_messages_all_recent(compactor):
    """When under keep_recent, older should be empty."""
    messages = _make_messages(1, system=True)
    # system + 2 conversation messages; keep_recent=4
    system, older, recent = compactor._split_messages(messages)
    assert len(system) == 1
    assert system[0]["role"] == "system"
    assert len(older) == 0
    assert len(recent) == 2  # user + assistant


def test_split_messages_with_older(compactor):
    """When over keep_recent, should correctly split system/older/recent."""
    messages = _make_messages(5, system=True)
    # 1 system + 10 conversation msgs; keep_recent=4 -> 6 older, 4 recent
    system, older, recent = compactor._split_messages(messages)
    assert len(system) == 1
    assert len(older) == 6
    assert len(recent) == 4


# ---- compact() ----


async def test_compact_preserves_system_messages(compactor, mock_provider):
    """System messages should always be preserved in compacted output."""
    messages = _make_messages(5, system=True)
    compacted, meta = await compactor.compact(messages, "openai/gpt-4o")
    # First message should be the system prompt
    assert compacted[0]["role"] == "system"
    assert compacted[0]["content"] == "You are a helpful assistant."


async def test_compact_calls_cheaper_model(compactor, mock_provider):
    """Compaction should call the provider with the cheaper model."""
    messages = _make_messages(5, system=True)
    await compactor.compact(messages, "openai/gpt-4o")
    # Verify mock was called
    mock_provider.chat.assert_called_once()
    call_kwargs = mock_provider.chat.call_args
    assert call_kwargs.kwargs.get("model") == "openai/gpt-4o-mini"


async def test_compact_metadata(compactor, mock_provider):
    """Compaction should return metadata with token counts."""
    messages = _make_messages(5, system=True)
    compacted, meta = await compactor.compact(messages, "openai/gpt-4o")
    assert "tokens_before" in meta
    assert "tokens_after" in meta
    assert "turns_removed" in meta
    assert meta["turns_removed"] == 6  # 10 conv msgs - 4 keep_recent = 6


async def test_compact_fallback_on_error(compactor, mock_provider):
    """When LLM call fails, should use fallback summary."""
    mock_provider.chat.side_effect = Exception("API error")
    messages = _make_messages(5, system=True)
    compacted, meta = await compactor.compact(messages, "openai/gpt-4o")
    # Should still return a valid message list
    assert len(compacted) > 0
    # The summary message should mention the fallback
    summary_msg = compacted[1]  # after system
    assert "[Context compacted" in summary_msg["content"]
    assert "compacted but summarization failed" in summary_msg["content"]


async def test_compact_no_older_messages(compactor, mock_provider):
    """When no older messages exist, return original messages unchanged."""
    messages = _make_messages(1, system=True)  # only 2 conv msgs, under keep_recent=4
    compacted, meta = await compactor.compact(messages, "openai/gpt-4o")
    assert compacted == messages
    assert meta["turns_removed"] == 0
    mock_provider.chat.assert_not_called()

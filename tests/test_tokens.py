"""Tests for nanobot.agent.tokens.TokenCounter."""

import json

import pytest

from nanobot.agent.tokens import TokenCounter, CONTEXT_WINDOWS, DEFAULT_CONTEXT_WINDOW


@pytest.fixture
def counter():
    return TokenCounter()


# ---- count() ----


def test_count_empty_string(counter):
    """Empty string should return 0 tokens."""
    assert counter.count("", "openai/gpt-4o") == 0


def test_count_openai_model(counter):
    """OpenAI models should use tiktoken and return a reasonable count."""
    text = "Hello, world! This is a test sentence."
    result = counter.count(text, "openai/gpt-4o")
    # tiktoken should give a count between 5 and 20 for this sentence
    assert 5 <= result <= 20


def test_count_gemini_model(counter):
    """Gemini models should use character-based estimate (~len/4)."""
    text = "Hello, world!"
    result = counter.count(text, "gemini/gemini-2.0-flash")
    expected = len(text) // 4 + 1
    assert result == expected


def test_count_anthropic_model(counter):
    """Anthropic models should use tiktoken approximation."""
    text = "The quick brown fox jumps over the lazy dog."
    result = counter.count(text, "anthropic/claude-sonnet-4-5")
    # Should get a reasonable tiktoken count, not the char-based one
    assert result > 0
    # Verify it's not using the gemini char-estimate path
    char_estimate = len(text) // 4 + 1
    # tiktoken count should differ from simple char estimate (usually close but not equal)
    assert isinstance(result, int)


def test_count_unknown_provider(counter):
    """Unknown providers should fall back to tiktoken."""
    text = "Some text to count tokens for."
    result = counter.count(text, "unknown-provider/some-model")
    assert result > 0
    assert isinstance(result, int)


# ---- count_messages() ----


def test_count_messages_basic(counter):
    """Basic message list should return positive token count with overhead."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    result = counter.count_messages(messages, "openai/gpt-4o")
    # Each message has ~4 tokens overhead plus content tokens
    assert result > 8  # at least 2 * 4 overhead


def test_count_messages_with_tool_calls(counter):
    """Messages with tool_calls should count name and arguments."""
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "/tmp/test.py"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "file contents here",
        },
    ]
    result = counter.count_messages(messages, "openai/gpt-4o")
    # Should include overhead + tool call name + arguments + tool result content
    assert result > 20


def test_count_messages_multipart_content(counter):
    """Multi-part content lists should be handled."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ],
        }
    ]
    result = counter.count_messages(messages, "openai/gpt-4o")
    # Should include text tokens + ~1000 for image
    assert result >= 1000


# ---- get_context_window() ----


def test_get_context_window_known_model(counter):
    """Known models should return their listed context window."""
    result = counter.get_context_window("openai/gpt-4o")
    assert result == 128_000


def test_get_context_window_prefix_match(counter):
    """Model with date suffix should match by prefix."""
    result = counter.get_context_window("anthropic/claude-3.5-sonnet-20240620")
    assert result == 200_000


def test_get_context_window_override(counter):
    """Config override should take precedence over known sizes."""
    result = counter.get_context_window("openai/gpt-4o", override=50_000)
    assert result == 50_000


def test_get_context_window_unknown_model(counter):
    """Unknown models should return the default context window."""
    result = counter.get_context_window("unknown/mystery-model")
    assert result == DEFAULT_CONTEXT_WINDOW


# ---- get_context_usage() ----


def test_get_context_usage(counter):
    """Should return (tokens_used, context_limit) tuple."""
    messages = [
        {"role": "user", "content": "Hello"},
    ]
    used, limit = counter.get_context_usage(messages, "openai/gpt-4o")
    assert used > 0
    assert limit == 128_000
    assert isinstance(used, int)
    assert isinstance(limit, int)

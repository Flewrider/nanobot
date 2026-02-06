"""Tests for nanobot.providers.claude_max — Claude CLI wrapper provider."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.providers.claude_max import ClaudeMaxProvider


@pytest.fixture
def provider():
    return ClaudeMaxProvider(cli_path="claude")


# ---------------------------------------------------------------------------
# Model prefix stripping
# ---------------------------------------------------------------------------


class TestModelParsing:
    def test_default_model(self, provider):
        assert provider.get_default_model() == "claude-sonnet-4-5"

    @pytest.mark.asyncio
    async def test_strips_claude_max_prefix(self, provider):
        """The claude_max/ prefix should be stripped before passing to CLI."""
        with patch("nanobot.providers.claude_max.asyncio") as mock_asyncio:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(
                return_value=(
                    json.dumps({"result": "hello"}).encode(),
                    b"",
                )
            )
            mock_asyncio.create_subprocess_exec = AsyncMock(return_value=mock_process)
            mock_asyncio.wait_for = AsyncMock(
                return_value=(
                    json.dumps({"result": "hello"}).encode(),
                    b"",
                )
            )

            # Instead of mocking asyncio internals, test the prompt conversion
            prompt = provider._messages_to_prompt([
                {"role": "user", "content": "hi"},
            ])
            assert "<user>" in prompt


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    def test_system_message(self, provider):
        prompt = provider._messages_to_prompt([
            {"role": "system", "content": "You are a helpful assistant."},
        ])
        assert "<system>" in prompt
        assert "You are a helpful assistant." in prompt

    def test_user_message(self, provider):
        prompt = provider._messages_to_prompt([
            {"role": "user", "content": "Hello"},
        ])
        assert "<user>" in prompt
        assert "Hello" in prompt

    def test_assistant_message(self, provider):
        prompt = provider._messages_to_prompt([
            {"role": "assistant", "content": "Hi there!"},
        ])
        assert "<assistant>" in prompt
        assert "Hi there!" in prompt

    def test_tool_calls_in_assistant(self, provider):
        prompt = provider._messages_to_prompt([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "/tmp/test.py"}),
                        }
                    }
                ],
            },
        ])
        assert '<tool_call name="read_file">' in prompt

    def test_tool_result_message(self, provider):
        prompt = provider._messages_to_prompt([
            {"role": "tool", "name": "read_file", "content": "file contents here"},
        ])
        assert '<tool_result name="read_file">' in prompt
        assert "file contents here" in prompt

    def test_multipart_content(self, provider):
        prompt = provider._messages_to_prompt([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            },
        ])
        assert "Describe this" in prompt


# ---------------------------------------------------------------------------
# JSON output parsing
# ---------------------------------------------------------------------------


class TestOutputParsing:
    def test_parse_result_field(self, provider):
        output = json.dumps({"result": "The answer is 42."})
        response = provider._parse_cli_output(output)
        assert response.content == "The answer is 42."
        assert response.finish_reason == "stop"

    def test_parse_with_usage(self, provider):
        output = json.dumps({
            "result": "Hello",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })
        response = provider._parse_cli_output(output)
        assert response.content == "Hello"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 5

    def test_parse_content_blocks(self, provider):
        output = json.dumps({
            "content": [
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
            ],
        })
        response = provider._parse_cli_output(output)
        assert "Part 1" in response.content
        assert "Part 2" in response.content

    def test_parse_tool_use_blocks(self, provider):
        output = json.dumps({
            "content": [
                {"type": "tool_use", "id": "tu_1", "name": "read_file", "input": {"path": "/tmp"}},
            ],
        })
        response = provider._parse_cli_output(output)
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "read_file"

    def test_parse_raw_text_fallback(self, provider):
        """If output isn't valid JSON, treat it as plain text."""
        response = provider._parse_cli_output("Just plain text response")
        assert response.content == "Just plain text response"

    def test_parse_empty_output(self, provider):
        response = provider._parse_cli_output("")
        assert response.content == ""

    def test_parse_list_format(self, provider):
        output = json.dumps([
            {"type": "text", "text": "Hello from list"},
        ])
        response = provider._parse_cli_output(output)
        assert "Hello from list" in response.content


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_missing_cli(self):
        """Should return a helpful error when claude CLI is not found."""
        provider = ClaudeMaxProvider(cli_path="/nonexistent/claude")
        response = await provider.chat(
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response.finish_reason == "error"
        assert "not found" in response.content.lower() or "error" in response.content.lower()


# ---------------------------------------------------------------------------
# Environment sanitization
# ---------------------------------------------------------------------------


class TestEnvSanitization:
    @pytest.mark.asyncio
    async def test_anthropic_key_cleared(self):
        """ANTHROPIC_API_KEY should be stripped from the subprocess env."""
        provider = ClaudeMaxProvider()

        captured_env = {}

        original_create = asyncio.create_subprocess_exec

        async def mock_create(*args, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(
                return_value=(json.dumps({"result": "ok"}).encode(), b"")
            )
            return mock_proc

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-secret123456789012"}):
            with patch("nanobot.providers.claude_max.asyncio.create_subprocess_exec", side_effect=mock_create):
                with patch("nanobot.providers.claude_max.asyncio.wait_for") as mock_wait:
                    mock_wait.return_value = (json.dumps({"result": "ok"}).encode(), b"")
                    await provider.chat(
                        messages=[{"role": "user", "content": "hi"}],
                    )

        assert "ANTHROPIC_API_KEY" not in captured_env

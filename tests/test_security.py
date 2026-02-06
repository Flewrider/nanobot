"""Tests for security hardening — credential protection."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from nanobot.agent.tools.shell import SENSITIVE_ENV_VARS, _build_clean_env
from nanobot.agent.tools.filesystem import _is_path_denied, DENIED_PATHS
from nanobot.agent.tools.registry import redact_api_keys


# ---------------------------------------------------------------------------
# Exec tool: env var sanitization
# ---------------------------------------------------------------------------


class TestEnvSanitization:
    def test_sensitive_vars_stripped(self):
        """All listed sensitive env vars should be removed from the clean env."""
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "sk-ant-secret",
            "OPENAI_API_KEY": "sk-secret",
            "GEMINI_API_KEY": "AIza-secret",
            "OPENROUTER_API_KEY": "sk-or-secret",
            "GROQ_API_KEY": "gsk_secret",
            "ZHIPUAI_API_KEY": "zhipu-secret",
            "BRAVE_API_KEY": "brave-secret",
            "OPENCLAW_GATEWAY_TOKEN": "gw-secret",
            "PATH": "/usr/bin",
        }):
            env = _build_clean_env()
            for var in SENSITIVE_ENV_VARS:
                assert var not in env, f"{var} should be stripped"
            # Non-sensitive vars should remain
            assert "PATH" in env

    def test_nanobot_providers_stripped(self):
        """Env vars matching NANOBOT_PROVIDERS__* should be stripped."""
        with patch.dict(os.environ, {
            "NANOBOT_PROVIDERS__ANTHROPIC__API_KEY": "secret",
            "NANOBOT_PROVIDERS__OPENAI__API_KEY": "secret",
            "NANOBOT_AGENTS__DEFAULTS__MODEL": "safe-value",
        }):
            env = _build_clean_env()
            assert "NANOBOT_PROVIDERS__ANTHROPIC__API_KEY" not in env
            assert "NANOBOT_PROVIDERS__OPENAI__API_KEY" not in env
            # Non-provider nanobot vars should remain
            assert "NANOBOT_AGENTS__DEFAULTS__MODEL" in env

    def test_clean_env_is_copy(self):
        """Modifying the returned env should not affect os.environ."""
        env = _build_clean_env()
        env["NEW_VAR"] = "test"
        assert "NEW_VAR" not in os.environ


# ---------------------------------------------------------------------------
# Read file: sensitive path denial
# ---------------------------------------------------------------------------


class TestPathDenial:
    def test_nanobot_config_denied(self):
        assert _is_path_denied("~/.nanobot/config.json") is True

    def test_openclaw_dir_denied(self):
        assert _is_path_denied("~/.openclaw/something") is True

    def test_claude_dir_denied(self):
        assert _is_path_denied("~/.claude/credentials") is True

    def test_dot_env_denied(self):
        assert _is_path_denied(".env") is True

    def test_etc_shadow_denied(self):
        assert _is_path_denied("/etc/shadow") is True

    def test_etc_environment_denied(self):
        assert _is_path_denied("/etc/environment") is True

    def test_normal_path_allowed(self):
        assert _is_path_denied("/tmp/test.py") is False

    def test_home_dir_allowed(self):
        assert _is_path_denied("~/Documents/file.txt") is False

    def test_workspace_file_allowed(self):
        assert _is_path_denied("~/.nanobot/workspace/AGENTS.md") is False


@pytest.mark.asyncio
async def test_read_file_tool_denies_config():
    from nanobot.agent.tools.filesystem import ReadFileTool

    tool = ReadFileTool()
    result = await tool.execute(path="~/.nanobot/config.json")
    assert "Access denied" in result


# ---------------------------------------------------------------------------
# Output filtering: API key redaction
# ---------------------------------------------------------------------------


class TestApiKeyRedaction:
    def test_anthropic_key_redacted(self):
        text = "Found key: sk-ant-api03-abcdef1234567890abcdef"
        assert "[REDACTED]" in redact_api_keys(text)
        assert "sk-ant-" not in redact_api_keys(text)

    def test_openai_key_redacted(self):
        text = "Key is sk-proj-abcdefghijklmnopqrstuv"
        assert "[REDACTED]" in redact_api_keys(text)

    def test_openrouter_key_redacted(self):
        text = "sk-or-v1-abcdefghijklmnopqrstuvwxyz"
        assert "[REDACTED]" in redact_api_keys(text)

    def test_groq_key_redacted(self):
        text = "gsk_abcdefghijklmnopqrstuvwxyz1234"
        assert "[REDACTED]" in redact_api_keys(text)

    def test_gemini_key_redacted(self):
        text = "AIzaSyDabcdefghijklmnopqrstuvwxyz"
        assert "[REDACTED]" in redact_api_keys(text)

    def test_normal_text_unchanged(self):
        text = "Hello, this is a normal response with no keys."
        assert redact_api_keys(text) == text

    def test_multiple_keys_redacted(self):
        text = "Keys: sk-ant-api03-aaabbbcccddd123456789012 and gsk_xxxyyyzzz123456789012345"
        result = redact_api_keys(text)
        assert "sk-ant-" not in result
        assert "gsk_" not in result
        assert result.count("[REDACTED]") == 2

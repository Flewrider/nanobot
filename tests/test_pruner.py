"""Tests for nanobot.agent.pruner.ContextPruner."""

import pytest

from nanobot.agent.pruner import ContextPruner


@pytest.fixture
def pruner():
    return ContextPruner(keep_recent=3)


def _tool_msg(name: str, content: str, tool_call_id: str = "tc_1"):
    return {
        "role": "tool",
        "name": name,
        "tool_call_id": tool_call_id,
        "content": content,
    }


# ---- prune() basics ----


def test_prune_no_tool_messages(pruner):
    """Messages without tool outputs should pass through unchanged."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm great!"},
    ]
    result = pruner.prune(messages, current_turn=2)
    assert result == messages


def test_prune_recent_kept_verbatim(pruner):
    """Last keep_recent (3) tool results per tool type should be kept verbatim."""
    messages = [
        {"role": "user", "content": "Read files"},
        _tool_msg("read_file", "content A", "tc_a"),
        _tool_msg("read_file", "content B", "tc_b"),
        _tool_msg("read_file", "content C", "tc_c"),
    ]
    result = pruner.prune(messages, current_turn=1)
    tool_msgs = [m for m in result if m.get("role") == "tool"]
    assert len(tool_msgs) == 3
    # All 3 are in recent_ids so should be kept as-is
    assert tool_msgs[0]["content"] == "content A"
    assert tool_msgs[1]["content"] == "content B"
    assert tool_msgs[2]["content"] == "content C"


def test_prune_stale_read_file(pruner):
    """read_file outputs older than 3 turns should be replaced with summary.

    The pruner keeps the last 3 tool results per type as "recent" (never pruned
    for staleness). To test staleness we need >3 results so the oldest falls
    outside the recent window.
    """
    messages = [
        {"role": "user", "content": "Q1"},
        _tool_msg("read_file", "/path/to/file.py\nline1\nline2\nline3", "tc_old"),
        {"role": "user", "content": "Q2"},
        _tool_msg("read_file", "recent content 1", "tc_r1"),
        {"role": "user", "content": "Q3"},
        _tool_msg("read_file", "recent content 2", "tc_r2"),
        {"role": "user", "content": "Q4"},
        _tool_msg("read_file", "recent content 3", "tc_r3"),
    ]
    # tc_old is at turn 1; current_turn=7; age = 6 >= threshold 3
    # tc_old is NOT in recent_ids (4 results, keep_recent=3, so tc_old is the oldest)
    result = pruner.prune(messages, current_turn=7)
    tool_msgs = [m for m in result if m.get("role") == "tool"]
    # First tool msg (tc_old) should be pruned
    assert "[Previously read:" in tool_msgs[0]["content"]
    # The other 3 should be kept verbatim
    assert tool_msgs[1]["content"] == "recent content 1"
    assert tool_msgs[2]["content"] == "recent content 2"
    assert tool_msgs[3]["content"] == "recent content 3"


def test_prune_stale_web_search(pruner):
    """web_search outputs older than 2 turns should be replaced."""
    messages = [
        {"role": "user", "content": "Search something"},
        _tool_msg("web_search", "Results for: python testing\n1. result one\n2. result two", "tc_old"),
        {"role": "user", "content": "Q2"},
        _tool_msg("web_search", "Results for: q2\n1. r", "tc_r1"),
        {"role": "user", "content": "Q3"},
        _tool_msg("web_search", "Results for: q3\n1. r", "tc_r2"),
        {"role": "user", "content": "Q4"},
        _tool_msg("web_search", "Results for: q4\n1. r", "tc_r3"),
    ]
    result = pruner.prune(messages, current_turn=7)
    tool_msgs = [m for m in result if m.get("role") == "tool"]
    assert "[Search:" in tool_msgs[0]["content"]


def test_prune_stale_exec(pruner):
    """exec outputs older than 3 turns should be replaced."""
    messages = [
        {"role": "user", "content": "Run command"},
        _tool_msg("exec", "ls -la output\nfile1\nfile2", "tc_old"),
        {"role": "user", "content": "Q2"},
        _tool_msg("exec", "echo ok", "tc_r1"),
        {"role": "user", "content": "Q3"},
        _tool_msg("exec", "echo ok2", "tc_r2"),
        {"role": "user", "content": "Q4"},
        _tool_msg("exec", "echo ok3", "tc_r3"),
    ]
    result = pruner.prune(messages, current_turn=7)
    tool_msgs = [m for m in result if m.get("role") == "tool"]
    assert "[Shell:" in tool_msgs[0]["content"]


def test_prune_unknown_tool_not_pruned(pruner):
    """Non-listed tools should be kept as-is regardless of age."""
    messages = [
        {"role": "user", "content": "Use custom tool"},
        _tool_msg("custom_tool", "some output data", "tc_custom"),
        {"role": "user", "content": "Q2"},
        {"role": "user", "content": "Q3"},
        {"role": "user", "content": "Q4"},
        {"role": "user", "content": "Q5"},
    ]
    result = pruner.prune(messages, current_turn=10)
    tool_msg = [m for m in result if m.get("role") == "tool"][0]
    assert tool_msg["content"] == "some output data"


def test_recent_large_output_not_truncated(pruner):
    """Recent large outputs should be kept fully verbatim."""
    large_content = "x" * 5000
    messages = [
        {"role": "user", "content": "Read big file"},
        _tool_msg("read_file", large_content, "tc_big"),
    ]
    result = pruner.prune(messages, current_turn=1)
    tool_msg = [m for m in result if m.get("role") == "tool"][0]
    assert tool_msg["content"] == large_content


def test_truncate_large_output_after_age(pruner):
    """Large outputs should be truncated only after at least 2 turns have passed."""
    large_content = "x" * 5000
    messages = [
        {"role": "user", "content": "Q1"},
        _tool_msg("read_file", large_content, "tc_old"),
        {"role": "user", "content": "Q2"},
        _tool_msg("read_file", "small recent 1", "tc_r1"),
        {"role": "user", "content": "Q3"},
        _tool_msg("read_file", "small recent 2", "tc_r2"),
        _tool_msg("read_file", "small recent 3", "tc_r3"),
    ]
    # tc_old is not in recent_ids (4 results, keep_recent=3)
    # turn_age = 3-1 = 2, which is >= 2 but < threshold 3 for summary, so truncate
    result = pruner.prune(messages, current_turn=3)
    tool_msgs = [m for m in result if m.get("role") == "tool"]
    assert len(tool_msgs[0]["content"]) < len(large_content)
    assert "truncated" in tool_msgs[0]["content"]


def test_no_truncate_within_2_turns(pruner):
    """Large outputs within 2 turns should NOT be truncated."""
    large_content = "x" * 5000
    messages = [
        {"role": "user", "content": "Q1"},
        _tool_msg("read_file", large_content, "tc_old"),
        {"role": "user", "content": "Q2"},
        _tool_msg("read_file", "r1", "tc_r1"),
        _tool_msg("read_file", "r2", "tc_r2"),
        _tool_msg("read_file", "r3", "tc_r3"),
    ]
    # tc_old turn_age = 2-1 = 1, which is < 2, so no truncation
    result = pruner.prune(messages, current_turn=2)
    tool_msgs = [m for m in result if m.get("role") == "tool"]
    assert tool_msgs[0]["content"] == large_content


# ---- _create_summary() ----


def test_create_summary_read_file(pruner):
    """read_file summary should mention the file and line count."""
    content = "/path/to/file.py\nline1\nline2\nline3"
    summary = pruner._create_summary("read_file", "tc_1", content)
    assert "[Previously read:" in summary
    assert "4 lines" in summary


def test_create_summary_exec(pruner):
    """exec summary should mention the command and output size."""
    content = "ls -la\ntotal 42\nfile1\nfile2"
    summary = pruner._create_summary("exec", "tc_1", content)
    assert "[Shell:" in summary
    assert "chars output" in summary


def test_create_summary_web_search(pruner):
    """web_search summary should mention the query and result count."""
    content = "Results for: python pytest\n1. first result\n2. second result"
    summary = pruner._create_summary("web_search", "tc_1", content)
    assert "[Search:" in summary
    assert "2 results" in summary


def test_create_summary_list_dir(pruner):
    """list_dir summary should mention the item count."""
    content = "file1.py\nfile2.py\ndir1"
    summary = pruner._create_summary("list_dir", "tc_1", content)
    assert "[Listed:" in summary
    assert "3 items" in summary


# ---- _canonical_name() ----


def test_canonical_name_alias(pruner):
    """exec_command should map to exec."""
    assert pruner._canonical_name("exec_command") == "exec"
    assert pruner._canonical_name("list_files") == "list_dir"
    assert pruner._canonical_name("read_file") == "read_file"

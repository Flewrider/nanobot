# nanobot Skills

This directory contains built-in skills that extend nanobot's capabilities.

## Skill Format

Each skill is a directory containing a `SKILL.md` file with:
- YAML frontmatter (name, description, metadata)
- Markdown instructions for the agent

## Attribution

These skills are adapted from [OpenClaw](https://github.com/openclaw/openclaw)'s skill system.
The skill format and metadata structure follow OpenClaw's conventions to maintain compatibility.

## Available Skills

| Skill | Description |
|-------|-------------|
| `github` | Interact with GitHub using the `gh` CLI |
| `weather` | Get weather info using wttr.in and Open-Meteo |
| `summarize` | Summarize URLs, files, and YouTube videos |
| `tmux` | Remote-control tmux sessions |
| `skill-creator` | Create new skills |

## Subagents and Roles

nanobot can spawn background subagents to handle focused tasks while the main agent continues. Use the `spawn` tool with a task description, and optionally pick a role defined in config.

Example tool call:

```json
{"task": "Scan the repo for TODOs and summarize", "role": "research"}
```

Roles are configured under `agents.roles` in the config. Each role can override the model, limit tools, and cap tool iterations:

```json
"roles": {
  "research": {
    "model": {"model": "openrouter/sonar"},
    "toolAllow": ["read_file", "list_dir", "web_search", "web_fetch"],
    "maxToolIterations": 8
  }
}
```

Tool names for allow/deny lists include: `read_file`, `write_file`, `edit_file`, `list_dir`, `exec`, `web_search`, `web_fetch`.

Subagents use a separate tool registry from the main agent to enforce role allow/deny filters and to keep `message`/`spawn` unavailable.

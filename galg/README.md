# AI Template

| Director | Status |
|:--------:|:------:|
| MATH | ACTIVE |

Template repository for autonomous AI-driven software development using Claude and Codex.

**Author:** Andrew Yates
**Copyright:** 2026 Dropbox, Inc.
**License:** Apache 2.0

**When using this template:** Replace this README with your project's README.

## What This Is

A starter kit for projects where AI workers autonomously write, test, and commit code. A human or AI manager provides direction via roadmaps and hints; workers execute continuously.

## Quick Start

1. Clone this template for your new project
2. Customize `CLAUDE.md` with your project goals and rules
3. Run the worker: `./run_worker.sh`
4. Monitor: `cat worker_status.json` or `tail -f worker_logs/*.jsonl | ./json_to_text.py`
5. Send hints: `echo "Focus on X" > HINT.txt`

## Files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project rules, goals, commit format. Workers read this. |
| `AGENTS.md` | Points to CLAUDE.md + Codex-specific notes |
| `run_worker.sh` | Autonomous worker loop (Claude + Codex rotation) |
| `json_to_text.py` | Pretty-prints Claude's streaming JSON output |

## How It Works

1. Worker reads `CLAUDE.md` and recent git history
2. Executes tasks, commits with structured messages
3. Next worker picks up from last commit
4. Manager intervenes via `HINT.txt` or `[MANAGER]` commits

## Requirements

- `claude` CLI (required)
- `codex` CLI (optional, `npm i -g @openai/codex`)
- Python 3 (for json_to_text.py)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

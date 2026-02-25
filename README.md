# Langfuse Observability

Agent Zero plugin for LLM observability via [Langfuse](https://langfuse.com).

## Features

- **Tracing** — Automatic tracing of LLM calls, tool executions, and agent interactions
- **Trace Viewer** — Browse and inspect traces from the Agent Zero sidebar
- **Chat Forking** — Fork any conversation at any point to explore alternatives
- **Prompt Lab** — Test and refine prompts with judge-based evaluation
- **Split View** — Side-by-side comparison of chat responses

## Installation

Clone or download this repo into your Agent Zero plugins directory:

```bash
git clone <repo-url> /path/to/agent-zero/usr/plugins/langfuse-observability
```

## Configuration

Configure via the Agent Zero settings UI (Agent tab > Observability section), or set environment variables:

- `LANGFUSE_PUBLIC_KEY` — Your Langfuse project public key
- `LANGFUSE_SECRET_KEY` — Your Langfuse project secret key
- `LANGFUSE_HOST` — Langfuse instance URL (default: `https://cloud.langfuse.com`)

## Requirements

The `langfuse` Python package is auto-installed on first use.

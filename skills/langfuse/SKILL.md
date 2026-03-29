---
name: langfuse
description: Interact with Langfuse and access its documentation. Use when needing to (1) query or modify Langfuse data programmatically via the CLI — traces, prompts, datasets, scores, sessions, and any other API resource, (2) look up Langfuse documentation, concepts, integration guides, or SDK usage, or (3) understand how any Langfuse feature works. This skill covers CLI-based API access (via npx) and multiple documentation retrieval methods.
version: 1.0.0
author: langfuse
tags: [langfuse, observability, tracing, llm-monitoring, prompts, datasets, scores, cli, instrumentation, sdk]
trigger_patterns: ["langfuse", "langfuse trace", "langfuse prompt", "langfuse dataset", "langfuse score", "langfuse session", "langfuse cli", "langfuse sdk", "langfuse api", "instrument application", "llm observability", "llm tracing", "monitor llm", "trace llm", "observability sdk", "migrate prompts langfuse", "user feedback langfuse"]
---

# Langfuse

This skill helps you use Langfuse effectively across all common workflows: instrumenting applications, migrating prompts, debugging traces, and accessing data programmatically.

## When to use

Load this skill when the user needs to:

- **Query or modify Langfuse data** — list/filter traces, sessions, prompts, datasets, scores via CLI or API
- **Instrument an application** — add Langfuse observability to Python/JS/TS code using SDK or integrations (OpenAI, LangChain, LlamaIndex, LiteLLM, Vercel AI SDK)
- **Look up Langfuse documentation** — find integration guides, SDK reference, concept explanations, or changelog
- **Migrate prompts** — move hardcoded prompts from a codebase into Langfuse Prompt Management
- **Capture user feedback** — attach thumbs-up/down ratings or implicit signals as scores to traces
- **Upgrade the Langfuse SDK** — migrate to a newer SDK version or resolve breaking changes
- **Debug traces** — explore trace/span hierarchy, identify missing tokens/model names, fix flat traces
- **Understand how a Langfuse feature works** — concepts like observations, generations, sessions, evals

## Inputs required

| Input | Where to find | Required? |
|---|---|---|
| `LANGFUSE_PUBLIC_KEY` | Langfuse UI → Settings → API Keys | Yes, for CLI/SDK |
| `LANGFUSE_SECRET_KEY` | Langfuse UI → Settings → API Keys | Yes, for CLI/SDK |
| `LANGFUSE_HOST` | Langfuse UI (EU: `https://cloud.langfuse.com`, US: `https://us.cloud.langfuse.com`, self-hosted: custom URL) | Yes — always required |
| Target codebase or file | User provides path or pastes code | For instrumentation tasks |
| Use-case intent | Inferred from context or user message | Determines which reference to load |

If credentials are not set, ask the user for their API keys. Do not proceed with CLI or SDK calls without `LANGFUSE_HOST`.

## Core Principles

Follow these principles for ALL Langfuse work:

1. **Documentation First**: NEVER implement based on memory. Always fetch current docs before writing code (Langfuse updates frequently) See the section below on how to access documentation.
2. **CLI for Data Access**: Use `langfuse-cli` when querying/modifying Langfuse data. See the section below on how to use the CLI.
3. **Best Practices by Use Case**: Check the relevant reference file below for use-case-specific guidelines before implementing
4. **Use latest Langfuse versions**: Unless the user specified otherwise or there's a good reason, always use the latest version of Langfuse SDKs/APIs.

## Use case specific references

- instrumenting an existing function/application: references/instrumentation.md
- migrating prompts from a codebase into Langfuse: references/prompt-migration.md
- capturing user feedback (thumbs, ratings, implicit signals) as scores on traces: references/user-feedback.md
- further tips on using the Langfuse CLI: references/cli.md
- upgrading or migrating Langfuse SDKs to the latest version: references/sdk-upgrade.md
- submitting feedback about this skill: references/skill-feedback.md

## Procedure

### Step 1 — Identify use case and load reference

1. Determine which use case applies (instrumentation, prompt migration, user feedback, CLI query, SDK upgrade, docs lookup).
2. Load the matching reference file listed above.
3. Fetch current documentation using one of the three methods in the **Langfuse Documentation** section below.

### Step 2 — Set credentials (for CLI or SDK work)

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://cloud.langfuse.com
```

Or use `scripts/check_credentials.sh` to verify credentials are set and reachable.

### Step 3 — Discover schema (for CLI work)

```bash
# Discover all available resources
npx langfuse-cli api __schema

# List actions for a resource
npx langfuse-cli api <resource> --help

# Show args/options for a specific action
npx langfuse-cli api <resource> <action> --help
```

For common workflows, tips, and full usage patterns, see [references/cli.md](references/cli.md).

### Step 4 — Fetch documentation

Three methods to access Langfuse docs, in order of preference. **Always prefer your application's native web fetch and search tools** (e.g., `WebFetch`, `WebSearch`, `mcp_fetch`, etc.) over `curl` when available.

#### 4a. Documentation Index (llms.txt)

Fetch the full index of all documentation pages:

```bash
curl -s https://langfuse.com/llms.txt
```

Returns a structured list of every doc page with titles and URLs. Use this to discover the right page for a topic, then fetch that page directly.

Alternatively, start on `https://langfuse.com/docs` and explore the site to find the page you need.

#### 4b. Fetch Individual Pages as Markdown

Any page listed in llms.txt can be fetched as markdown by appending `.md` to its path or by using `Accept: text/markdown` in the request headers.

```bash
curl -s "https://langfuse.com/docs/observability/overview.md"
curl -s "https://langfuse.com/docs/observability/overview" -H "Accept: text/markdown"
```

#### 4c. Search Documentation

When you need to find information across all docs and GitHub issues/discussions:

```bash
curl -s "https://langfuse.com/api/search-docs?query=<url-encoded-query>"
# Example:
curl -s "https://langfuse.com/api/search-docs?query=How+do+I+trace+LangGraph+agents"
```

Returns JSON with `query`, `answer` (array of matching docs with `url`, `title`, `source.content`). Responses can be large — extract only relevant portions.

#### Documentation Workflow

1. Start with **llms.txt** to orient — scan for relevant page titles
2. **Fetch specific pages** when you identify the right one
3. Fall back to **search** when the topic is unclear or you need GitHub Issues context

### Step 5 — Implement with baseline requirements

Every trace should capture:
- Model name and token usage
- Descriptive trace names (not generic)
- Proper span hierarchy (nested, not flat)
- Correct observation types (generations vs spans)
- Masked sensitive data
- Meaningful trace input/output

Prefer framework integrations over manual instrumentation when available (OpenAI, LangChain, LlamaIndex, Vercel AI SDK, LiteLLM).

### Step 6 — Verify in Langfuse UI

After implementation, direct the user to explore:
- **Traces view** — confirm trace/span hierarchy and names
- **Sessions view** — verify session grouping if `session_id` was added
- **Dashboard** — review filtered tag-based views
- **Scores** — check user feedback and eval scores are attached

## Verification

After completing Langfuse work, verify:

- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` are set and correct
- Traces appear in the Langfuse UI with expected names and hierarchy
- Token usage and model name are captured on generation spans
- No sensitive data (PII, API keys) is logged in trace inputs/outputs
- For scripts: `flush()` is called before process exit to prevent data loss
- For integrations: import order is correct (Langfuse before OpenAI client if using OpenAI integration)
- CLI queries return expected resource counts or data shapes
- Scores/feedback are attached to the correct trace IDs

## Failure modes

| Failure | Cause | Fix |
|---|---|---|
| Traces not appearing in UI | Missing `flush()` call in scripts | Add `langfuse.flush()` before process exit |
| Flat trace structure | All spans at root level | Use context managers or parent references to nest spans |
| Missing token counts | Manual instrumentation without model tracking | Switch to framework integration or set `usage` on generation |
| Missing model name | Generation created without `model` field | Always pass `model=` to generation objects |
| Generic trace names | Traces named after function names not intent | Use descriptive names like `chat-with-context` not `run()` |
| OpenAI import order error | Langfuse wrapper applied after OpenAI client created | Import and wrap OpenAI client before first use |
| Env vars not loaded | Langfuse imported before dotenv loaded | Load dotenv before importing Langfuse SDK |
| CLI auth failure | Missing or wrong `LANGFUSE_HOST` | Always set `LANGFUSE_HOST` — no default assumed |
| Sensitive data in traces | PII logged in trace inputs/outputs | Mask before passing to Langfuse; use `metadata` for debug info |
| `npx langfuse-cli` not found | Node.js not installed | Install Node.js; CLI requires npx |

For more detail on common instrumentation mistakes, see [references/instrumentation.md](references/instrumentation.md).

## Escalation

Escalate or seek additional help when:

- The Langfuse API schema has changed and CLI commands no longer work — re-run `npx langfuse-cli api __schema` and check the [changelog](https://langfuse.com/changelog)
- Authentication errors persist after verifying credentials — check Langfuse project permissions and key scope
- SDK behavior does not match documentation — fall back to `2c. Search Documentation` to check GitHub Issues
- This skill's guidance appears incorrect or outdated — submit feedback using the process in [references/skill-feedback.md](references/skill-feedback.md)
- Self-hosted Langfuse deployment issues — consult [self-hosting docs](https://langfuse.com/docs/deployment/self-host) and Langfuse Discord

## Skill Feedback

When the user expresses that something about this skill is not working as expected, gives incorrect guidance, is missing information, or could be improved — offer to submit feedback to the Langfuse skill maintainers. This includes when:

- The skill gave wrong or outdated instructions
- A workflow didn't produce the expected result
- The user wishes the skill covered something it doesn't
- The user explicitly says something like "this should work differently" or "this is wrong"

**Do NOT trigger this** for issues with Langfuse itself (the product) — only for issues with this skill's instructions and behavior.

When triggered, follow the process in [references/skill-feedback.md](references/skill-feedback.md).

import os
import sys
import importlib
import json

_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PLUGIN_ROOT not in sys.path:
    sys.path.append(_PLUGIN_ROOT)

# Force-reload langfuse_helper from disk so any runtime edits are always picked up
# (guards against stale sys.modules cache in long-running processes)
_lf_mod_name = "langfuse_helpers.langfuse_helper"
if _lf_mod_name in sys.modules:
    try:
        importlib.reload(sys.modules[_lf_mod_name])
    except Exception:
        pass

from helpers.extension import Extension
from langfuse_helpers.langfuse_helper import get_langfuse_client, should_sample, get_version_info, get_langfuse_config, set_active_span
from langfuse import LangfuseOtelSpanAttributes
from agent import Agent, LoopData
from helpers import projects


def _resolve_project_name(agent) -> "str | None":
    """Resolve the active A0 project name for the current agent session.

    Tries three tiers in order so that both top-level and subordinate agents
    correctly inherit the project context:

    1. context.data['project']  — set by projects.activate_project() on this
       context (the happy-path; works for top-level sessions).
    2. context.output_data['project']['name']  — also set by activate_project()
       alongside tier-1; provides a fallback when data/output_data diverge.
    3. Superior-agent walk — subordinate agents spawned via call_subordinate
       may have a fresh context; walking the superior chain surfaces the
       root session's project name.

    Returns None gracefully when no active project is found.
    """
    context = getattr(agent, "context", None)
    if context is None:
        return None

    # Tier 1: direct context data key (set by activate_project)
    name = projects.get_context_project_name(context)
    if name:
        return name

    # Tier 2: output_data['project'] (also set by activate_project alongside data)
    try:
        out = context.get_output_data(projects.CONTEXT_DATA_KEY_PROJECT)
        if isinstance(out, dict):
            name = out.get("name") or ""
            if name:
                return name
        elif isinstance(out, str) and out:
            return out
    except Exception:
        pass

    # Tier 3: walk superior-agent chain so subordinates inherit project
    try:
        from agent import Agent as _Agent
        superior = agent.get_data(_Agent.DATA_NAME_SUPERIOR)
        while superior is not None:
            sup_ctx = getattr(superior, "context", None)
            if sup_ctx is not None:
                name = projects.get_context_project_name(sup_ctx)
                if name:
                    return name
                try:
                    out = sup_ctx.get_output_data(projects.CONTEXT_DATA_KEY_PROJECT)
                    if isinstance(out, dict):
                        name = out.get("name") or ""
                        if name:
                            return name
                    elif isinstance(out, str) and out:
                        return out
                except Exception:
                    pass
            # move up another level
            try:
                superior = superior.get_data(_Agent.DATA_NAME_SUPERIOR)
            except Exception:
                break
    except Exception:
        pass

    return None


def _build_trace_name(
    agent,
    user_msg: str,
    superior=None,
    template: str = "",
) -> str:
    """Build trace/span name per ADR-001 tiered strategy.

    Tiers (evaluated in order):
      Override  - plugin config template string (if set)
      L0        - agent_number == 0 and user message present → first 60 chars
      L1        - agent_number == 0, no user message         → 'agent0'
      L2        - agent_number  > 0                          → 'parent > this' (max 2 levels)
    """
    profile = agent.config.profile or f"agent{agent.number}"
    model = "unknown"
    try:
        m = agent.get_chat_model()
        raw = getattr(m, "model_name", "unknown") if m else "unknown"
        # strip provider prefix for template
        model = raw.split("/", 1)[1] if "/" in raw and not raw.startswith("ft:") else raw
    except Exception:
        pass

    parent_profile = ""
    if superior:
        try:
            parent_profile = superior.config.profile or f"agent{superior.number}"
        except Exception:
            pass

    # Override tier — simple {var} interpolation, no external calls
    if template:
        try:
            return template.format(
                profile=profile,
                model=model,
                agent_number=agent.number,
                parent_profile=parent_profile,
                user_msg=(user_msg[:60].strip() if user_msg else ""),
            )
        except Exception:
            pass  # fall through to tiered logic

    # L2 — subordinate agent
    if agent.number > 0:
        if parent_profile:
            return f"{parent_profile} > {profile}"
        return profile

    # L0 — top-level agent with user message
    if user_msg and user_msg.strip():
        truncated = user_msg.strip()[:60]
        return truncated + ("…" if len(user_msg.strip()) > 60 else "")

    # L1 — top-level agent, no message
    return profile


class LangfuseTraceStart(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        client = get_langfuse_client(self.agent)
        if not client:
            return

        if not should_sample(self.agent):
            loop_data.params_persistent["lf_sampled"] = False
            return
        loop_data.params_persistent["lf_sampled"] = True

        agent = self.agent
        context_id = str(agent.context.id) if agent.context else "unknown"
        project_name = _resolve_project_name(agent)
        config = get_langfuse_config(agent)
        template = config.get("trace_name_template", "")

        # Check for parent agent (subordinate nesting)
        superior = agent.get_data(Agent.DATA_NAME_SUPERIOR)
        if superior and hasattr(superior, "loop_data"):
            parent_span = superior.loop_data.params_temporary.get("lf_tool_span")
            if not parent_span:
                parent_span = superior.loop_data.params_temporary.get("lf_iteration_span")
            if not parent_span:
                parent_span = superior.loop_data.params_persistent.get("lf_trace")

            if parent_span:
                span_name = _build_trace_name(agent, "", superior=superior, template=template)
                span = parent_span.start_observation(
                    name=span_name,
                    as_type="agent",
                    metadata={
                        "agent_number": agent.number,
                        "agent_profile": agent.config.profile,
                        **({'project': project_name} if project_name else {}),
                    },
                )
                loop_data.params_persistent["lf_trace"] = span
                loop_data.params_persistent["lf_root_trace"] = (
                    superior.loop_data.params_persistent.get("lf_root_trace")
                    or superior.loop_data.params_persistent.get("lf_trace")
                )
                set_active_span(span)
                return

        # Top-level agent: create a root observation (becomes a new trace in v4)
        user_msg = ""
        if loop_data.user_message:
            # output_text() returns "user: <message>" — strip role prefix to get bare text
            # Using output_text() instead of str(content) avoids dict repr e.g. {'user_message': 'test'}
            _raw = loop_data.user_message.output_text()
            user_msg = _raw.split(": ", 1)[-1].strip() if ": " in _raw else _raw.strip()
            # If the stripped value is still a serialised JSON dict, unwrap the
            # human-readable text field. Handles A0 content objects like
            #   '{"user_message": "hello world"}'
            if user_msg.startswith("{") and user_msg.endswith("}"):
                try:
                    _parsed = json.loads(user_msg)
                    if isinstance(_parsed, dict):
                        user_msg = str(
                            _parsed.get("user_message")
                            or _parsed.get("message")
                            or _parsed.get("text")
                            or user_msg
                        )
                except (json.JSONDecodeError, TypeError):
                    pass

        trace_name = _build_trace_name(agent, user_msg, superior=None, template=template)
        trace_name = f"[{context_id}] {trace_name}"

        root_span = client.start_observation(
            name=trace_name,
            as_type="agent",
            input=user_msg,
            metadata={
                "agent_number": agent.number,
                "agent_profile": agent.config.profile,
                **{f"v_{k}": v for k, v in get_version_info().items() if v},
                **({'v_context_id': context_id} if context_id else {}),
                **({'project': project_name} if project_name else {}),
            },
        )
        # Set trace-level attributes via OTel span
        if hasattr(root_span, "_otel_span"):
            root_span._otel_span.set_attribute(
                LangfuseOtelSpanAttributes.TRACE_SESSION_ID, context_id
            )
            root_span._otel_span.set_attribute(
                LangfuseOtelSpanAttributes.TRACE_NAME, trace_name
            )
            # tags — "agent-zero" for global cross-trace filtering, profile for per-agent
            profile = agent.config.profile or f"agent{agent.number}"
            root_span._otel_span.set_attribute(
                LangfuseOtelSpanAttributes.TRACE_TAGS,
                json.dumps(["agent-zero", profile] + ([project_name] if project_name else [])),
            )
            # agent.profile — surface agent profile in Langfuse trace for per-profile filtering
            root_span._otel_span.set_attribute('agent.profile', str(profile))
            if project_name:
                root_span._otel_span.set_attribute('agent.project', project_name)
        loop_data.params_persistent["lf_trace"] = root_span
        loop_data.params_persistent["lf_root_trace"] = root_span
        loop_data.params_persistent["lf_trace_id"] = root_span.trace_id
        set_active_span(root_span)

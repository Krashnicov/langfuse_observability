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
from langfuse_helpers.langfuse_helper import (
    get_langfuse_client,
    should_sample,
    get_version_info,
    get_langfuse_config,
    set_active_span,
    resolve_project_name,
)
from langfuse import LangfuseOtelSpanAttributes
from agent import Agent, LoopData


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
        return truncated + ("\u2026" if len(user_msg.strip()) > 60 else "")

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

        # project_name may be None on the first monologue of a new-chat started from a
        # project (Bug 1 — activation race): activate_project() may not have fired yet.
        # The monologue_end flush extension re-resolves and retroactively patches the
        # trace tags / agent.project OTel attribute before the span is closed.
        project_name = resolve_project_name(agent)

        config = get_langfuse_config(agent)
        template = config.get("trace_name_template", "")

        # --- Version / release info (computed once, used in multiple places) ---------
        v_info = get_version_info()
        release = config.get("release") or v_info.get("plugin_version") or "1.0.0"

        # OTel resource attributes mirror what get_langfuse_client() bakes into the
        # TracerProvider Resource so observers can correlate traces with the host.
        resource_attributes: dict = {
            "service.name": config.get("service_name", "agent-zero"),
        }
        if config.get("environment"):
            resource_attributes["deployment.environment"] = config["environment"]
        if release:
            resource_attributes["service.version"] = release

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

                # Observation metadata — ALL fields unconditional except project
                obs_metadata: dict = {
                    "agent_number": agent.number,
                    "agent_profile": agent.config.profile,
                    "resourceAttributes": resource_attributes,
                    **{f"v_{k}": v for k, v in v_info.items()},
                    "v_context_id": context_id,
                }
                if project_name:
                    obs_metadata["project"] = project_name

                span = parent_span.start_observation(
                    name=span_name,
                    as_type="agent",
                    metadata=obs_metadata,
                )
                loop_data.params_persistent["lf_trace"] = span
                loop_data.params_persistent["lf_root_trace"] = (
                    superior.loop_data.params_persistent.get("lf_root_trace")
                    or superior.loop_data.params_persistent.get("lf_trace")
                )
                # Store resolved project name (None means unresolved — flush will retry)
                loop_data.params_persistent["lf_project_name"] = project_name
                set_active_span(span)
                return

        # --- Top-level agent: create a root observation (becomes a new trace in v4) --
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

        # Build trace metadata — ALL fields unconditional except project
        # This dict is passed both to start_observation() (observation-level view)
        # and to TRACE_METADATA OTel attr (trace-level view in Langfuse UI).
        trace_metadata: dict = {
            "agent_number": agent.number,
            "agent_profile": agent.config.profile,
            "resourceAttributes": resource_attributes,
            **{f"v_{k}": v for k, v in v_info.items()},
            "v_context_id": context_id,
        }
        if project_name:
            trace_metadata["project"] = project_name

        root_span = client.start_observation(
            name=trace_name,
            as_type="agent",
            input=user_msg,
            metadata=trace_metadata,
        )

        # Set trace-level OTel attributes.
        # ALL attributes set unconditionally — project-related ones are the only exception.
        if hasattr(root_span, "_otel_span"):
            otel = root_span._otel_span

            # Trace identity — always set
            otel.set_attribute(LangfuseOtelSpanAttributes.TRACE_SESSION_ID, context_id)
            otel.set_attribute(LangfuseOtelSpanAttributes.TRACE_NAME, trace_name)

            # Release and version — always set (plugin version or '1.0.0' fallback)
            otel.set_attribute(LangfuseOtelSpanAttributes.RELEASE, release)
            otel.set_attribute(LangfuseOtelSpanAttributes.VERSION, release)

            # Trace-level metadata — mirrors trace_metadata so both trace and
            # observation views in the Langfuse UI show the full field set
            otel.set_attribute(
                LangfuseOtelSpanAttributes.TRACE_METADATA,
                json.dumps(trace_metadata),
            )

            # Tags — base tags always present; project tag appended only when resolved.
            # 'agent-zero' and 'agent0' are stable cross-trace filter handles.
            # Bug 1: if project_name is None here the flush extension will retroactively
            # add the project tag after late resolution at monologue_end time.
            tags = ["agent-zero", "agent0"]
            if project_name:
                tags.append(project_name)
            otel.set_attribute(
                LangfuseOtelSpanAttributes.TRACE_TAGS,
                json.dumps(tags),
            )

            # Custom span attributes for per-agent filtering
            profile = agent.config.profile or f"agent{agent.number}"
            otel.set_attribute("agent.profile", str(profile))
            # agent.project only set when resolved — never set to None/empty
            if project_name:
                otel.set_attribute("agent.project", project_name)

        # Store for downstream extensions
        loop_data.params_persistent["lf_trace"] = root_span
        loop_data.params_persistent["lf_root_trace"] = root_span
        loop_data.params_persistent["lf_trace_id"] = root_span.trace_id
        # lf_project_name: None signals flush to attempt retroactive resolution (Bug 1)
        loop_data.params_persistent["lf_project_name"] = project_name
        set_active_span(root_span)

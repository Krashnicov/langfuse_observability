"""Flush active Langfuse trace when the agent is stopped via kill_process().

Why this exists
---------------
The normal flush path lives in monologue_end/_90_langfuse_flush.py, which runs
from monologue()'s ``finally`` block.  That block is guarded:

    if self.context.task and self.context.task.is_alive():
        await call_extensions_async("monologue_end", ...)

When the user presses the stop button, AgentContext.kill_process() calls
self.task.kill(), which terminates the task *before* the finally block fires.
Consequently monologue_end never runs, the OTel span is never ended/flushed,
and the trace silently disappears from Langfuse.

This extension registers on the ``kill_process/end`` _functions hook (fired
after task.kill() returns) and performs a best-effort flush of any open span.

Calling context
---------------
- self.agent is None — AgentContext is not an Agent instance.
- data['args'][0] is the AgentContext instance.
- We walk context.agents (if available) or fall back to context.agent0 to
  locate the Agent whose loop_data holds the open trace.
- All operations are wrapped in broad except clauses: never raise from here.
"""
import logging
import json
import os
import sys

_PLUGIN_ROOT = None
_here = os.path.abspath(__file__)
for _ in range(8):  # walk up: end/kill_process/AgentContext/agent/_functions/python/extensions/plugin
    _here = os.path.dirname(_here)
    _candidate = os.path.join(_here, "langfuse_helpers")
    if os.path.isdir(_candidate):
        _PLUGIN_ROOT = _here
        break

if _PLUGIN_ROOT and _PLUGIN_ROOT not in sys.path:
    sys.path.append(_PLUGIN_ROOT)

from helpers.extension import Extension

_logger = logging.getLogger(__name__)


class LangfuseKillFlush(Extension):
    """Flush open Langfuse trace when kill_process() is called (stop-agent)."""

    def execute(self, data: dict = {}, **kwargs) -> None:  # sync — kill_process is sync
        try:
            self._flush(data)
        except Exception as exc:
            _logger.debug("[langfuse] kill_process flush failed (non-fatal): %s", exc)

    def _flush(self, data: dict) -> None:
        # Resolve the AgentContext from the extensible data payload
        args = data.get("args", ())
        context = args[0] if args else None
        if context is None:
            return

        # Locate the streaming agent (likely mid-monologue) or fall back to agent0
        streaming = getattr(context, "streaming_agent", None)
        agent0 = getattr(context, "agent0", None)
        candidate_agents = [a for a in [streaming, agent0] if a is not None]

        for agent in candidate_agents:
            loop_data = getattr(agent, "loop_data", None)
            if loop_data is None:
                continue
            if not loop_data.params_persistent.get("lf_sampled"):
                continue
            self._flush_agent(agent, loop_data)
            break  # only flush once for the root agent

    def _flush_agent(self, agent, loop_data) -> None:
        try:
            from langfuse_helpers.langfuse_helper import get_langfuse_client
        except ImportError:
            return

        client = get_langfuse_client(agent)
        if not client:
            return

        trace = loop_data.params_persistent.get("lf_trace")
        if not trace:
            return

        # Add a stop event observation so the trace records the interruption
        try:
            event = trace.start_observation(
                name="agent-stopped",
                as_type="event",
                metadata={
                    "agent_number": agent.number,
                    "agent_profile": agent.config.profile,
                    "reason": "kill_process",
                },
            )
            event.end()
        except Exception:
            pass

        # End and flush the trace
        try:
            trace.update(output="[agent stopped by user]")
            trace.end()
        except Exception:
            pass

        try:
            client.flush()
        except Exception:
            pass

        _logger.debug(
            "[langfuse] flushed trace on kill_process for agent %s (context %s)",
            agent.number,
            getattr(agent.context, "id", "?"),
        )

        # Clean up params so a subsequent monologue_end doesn't double-flush
        loop_data.params_persistent.pop("lf_trace", None)
        loop_data.params_persistent.pop("lf_root_trace", None)
        loop_data.params_persistent.pop("lf_sampled", None)
        loop_data.params_persistent.pop("lf_project_name", None)

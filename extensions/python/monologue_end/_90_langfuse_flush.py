import json
import os
import importlib
import sys

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
from langfuse_helpers.langfuse_helper import get_langfuse_client, resolve_project_name
from langfuse import LangfuseOtelSpanAttributes
from agent import LoopData


class LangfuseFlush(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        if not loop_data.params_persistent.get("lf_sampled"):
            return

        client = get_langfuse_client(self.agent)
        if not client:
            return

        trace = loop_data.params_persistent.get("lf_trace")
        if trace:
            # ---------------------------------------------------------------
            # Bug 1 retroactive fix — activation race on new-chat from project
            #
            # When a user clicks 'new chat' from a project the A0 chat_create
            # handler copies project data via set_data() onto the new context.
            # This happens synchronously in the HTTP handler BEFORE the first
            # user message is sent, so by monologue_start time the data is
            # already present — tier-1 of resolve_project_name() should catch
            # it.  However api_message.py also calls activate_project() on the
            # *first* message, which additionally sets output_data.  If for any
            # reason the context data copy was incomplete or the resolve raced
            # with the set, lf_project_name will be None here.
            #
            # We re-resolve at flush time (after the full monologue has run).
            # By this point activate_project() has almost certainly completed.
            # If we now obtain a project name we retroactively patch the OTel
            # span attributes so the Langfuse trace is properly tagged.
            # ---------------------------------------------------------------
            start_project_name = loop_data.params_persistent.get("lf_project_name")

            # Only attempt re-resolution when monologue_start could not resolve
            if start_project_name is None:
                late_project_name = resolve_project_name(self.agent)
                if late_project_name and hasattr(trace, "_otel_span"):
                    otel = trace._otel_span
                    try:
                        # Retroactively add project tag — preserve any existing
                        # tags that were set at trace-start time
                        tags = ["agent-zero", "agent0", late_project_name]
                        otel.set_attribute(
                            LangfuseOtelSpanAttributes.TRACE_TAGS,
                            json.dumps(tags),
                        )
                        # Set agent.project custom attribute
                        otel.set_attribute("agent.project", late_project_name)
                    except Exception:
                        pass

            try:
                trace.update(
                    # No truncation — pass full response text to Langfuse
                    # (API-side /traces/{id} response is truncated to 2000 chars by server;
                    #  individual /observations responses return full content)
                    output=loop_data.last_response or "",
                )
                trace.end()
            except Exception:
                pass

        try:
            client.flush()
        except Exception:
            pass

        loop_data.params_persistent.pop("lf_trace", None)
        loop_data.params_persistent.pop("lf_root_trace", None)
        loop_data.params_persistent.pop("lf_sampled", None)
        loop_data.params_persistent.pop("lf_project_name", None)

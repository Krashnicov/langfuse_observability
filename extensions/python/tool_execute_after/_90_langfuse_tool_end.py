from helpers.extension import Extension
from helpers.tool import Response


class LangfuseToolSpanEnd(Extension):

    async def execute(self, response: Response | None = None, tool_name: str = "", **kwargs):
        loop_data = self.agent.loop_data
        if not loop_data or not loop_data.params_persistent.get("lf_sampled"):
            return

        span = loop_data.params_temporary.get("lf_tool_span")
        if not span:
            return

        output = ""
        if response:
            output = str(response.message)[:2000] if response.message else ""

        try:
            span.update(output=output)
            span.end()
        except Exception:
            pass

        # Item 7b — tool error event
        is_error = response and getattr(response, "error", False)
        if is_error:
            try:
                parent = loop_data.params_temporary.get("lf_iteration_span")
                if not parent:
                    parent = loop_data.params_persistent.get("lf_trace")
                if parent:
                    error_msg = output[:500] if output else "tool error"
                    event = parent.start_observation(
                        name="tool-error",
                        as_type="event",
                        metadata={
                            "tool_name": tool_name,
                            "error_preview": error_msg,
                            "iteration": loop_data.iteration,
                            "agent_profile": self.agent.config.profile,
                        },
                    )
                    event.end()
            except Exception:
                pass

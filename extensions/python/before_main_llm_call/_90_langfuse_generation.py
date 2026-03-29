import json

from helpers.extension import Extension
from agent import Agent, LoopData


def _strip_provider(model_name: str) -> str:
    if "/" in model_name and not model_name.startswith("ft:"):
        return model_name.split("/", 1)[1]
    return model_name


def _stringify(content) -> str:
    """Convert any MessageContent to a readable string."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "raw_content" in content:
            return content.get("preview") or json.dumps(content["raw_content"], default=str, indent=2)
        return json.dumps(content, default=str, indent=2)
    if isinstance(content, list):
        parts = []
        for item in content:
            parts.append(_stringify(item))
        return "\n".join(parts)
    return str(content)


def _format_prompt(system_parts, history_output) -> str:
    """Build a clean, readable markdown-formatted prompt string."""
    sections = []

    if system_parts:
        system_text = "\n\n".join(str(s) for s in system_parts)
        sections.append(f"# System\n\n{system_text}")

    if history_output:
        for msg in history_output:
            role = "Assistant" if msg.get("ai") else "User"
            content = _stringify(msg.get("content", ""))
            if content.strip():
                sections.append(f"# {role}\n\n{content}")

    return "\n\n---\n\n".join(sections)


def _get_ctx_limit(model_name: str) -> int:
    """Return max_input_tokens for model via litellm.get_model_info(). Returns 0 on failure."""
    try:
        import litellm
        info = litellm.get_model_info(model_name)
        return int(info.get("max_input_tokens") or info.get("max_tokens") or 0)
    except Exception:
        return 0


class LangfuseGenerationStart(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        if not loop_data.params_persistent.get("lf_sampled"):
            return

        parent = loop_data.params_temporary.get("lf_iteration_span")
        if not parent:
            parent = loop_data.params_persistent.get("lf_trace")
        if not parent:
            return

        model = self.agent.get_chat_model()
        model_name = getattr(model, "model_name", "unknown") if model else "unknown"
        model_name_stripped = _strip_provider(model_name)

        # Build readable formatted prompt string
        prompt_text = _format_prompt(loop_data.system, loop_data.history_output)

        # Get pre-computed input token count
        ctx_window = self.agent.get_data(Agent.DATA_NAME_CTX_WINDOW)
        input_tokens = 0
        if isinstance(ctx_window, dict):
            input_tokens = int(ctx_window.get("tokens", 0))

        # Context window utilisation (item 5)
        ctx_limit = _get_ctx_limit(model_name)
        ctx_pct = round(input_tokens / ctx_limit * 100, 1) if ctx_limit and input_tokens else None

        metadata: dict = {
            "agent_number": self.agent.number,
            "agent_profile": self.agent.config.profile,
            "iteration": loop_data.iteration,
            "ctx_tokens_used": input_tokens,
        }
        if ctx_limit:
            metadata["ctx_limit"] = ctx_limit
        if ctx_pct is not None:
            metadata["ctx_pct"] = ctx_pct

        generation = parent.start_observation(
            name="main-llm",
            as_type="generation",
            model=model_name_stripped,
            input=prompt_text or None,
            metadata=metadata,
        )
        loop_data.params_temporary["lf_generation"] = generation
        loop_data.params_temporary["lf_input_tokens"] = input_tokens

        # Register span for LiteLLM usage callback (item 4)
        try:
            from langfuse_helpers.langfuse_helper import register_pending_generation
            register_pending_generation(generation, loop_data)
        except Exception:
            pass

        # Item 7a — context pressure event when >= 80% of window used
        if ctx_pct is not None and ctx_pct >= 80:
            try:
                event = parent.start_observation(
                    name="context-pressure",
                    as_type="event",
                    metadata={
                        "ctx_pct": ctx_pct,
                        "ctx_tokens_used": input_tokens,
                        "ctx_limit": ctx_limit,
                        "iteration": loop_data.iteration,
                        "agent_profile": self.agent.config.profile,
                    },
                )
                event.end()
            except Exception:
                pass

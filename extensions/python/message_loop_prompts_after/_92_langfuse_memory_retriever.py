import json
from helpers.extension import Extension
from agent import LoopData


class LangfuseMemoryRetriever(Extension):
    """Emit a Langfuse 'retriever' observation after memory recall completes.

    Fires at message_loop_prompts_after (number _92) — after _91_recall_wait.py
    has resolved the async recall task and populated extras_persistent.
    """

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        if not loop_data.params_persistent.get("lf_sampled"):
            return

        parent = loop_data.params_temporary.get("lf_iteration_span")
        if not parent:
            parent = loop_data.params_persistent.get("lf_trace")
        if not parent:
            return

        extras = loop_data.extras_persistent
        memories_txt = extras.get("memories", "")
        solutions_txt = extras.get("solutions", "")

        # Only emit if at least one recall ran this iteration
        # (check agent data for the recall task to confirm it fired)
        try:
            from plugins._memory.extensions.python.message_loop_prompts_after._50_recall_memories import (
                DATA_NAME_ITER as DATA_NAME_ITER_MEMORIES,
            )
            recall_iter = self.agent.get_data(DATA_NAME_ITER_MEMORIES)
            if recall_iter != loop_data.iteration:
                return  # recall did not run this iteration
        except Exception:
            # If memory plugin not present, skip gracefully
            return

        # Count results
        mem_count = len([m for m in memories_txt.split("\n\n") if m.strip()]) if memories_txt else 0
        sol_count = len([s for s in solutions_txt.split("\n\n") if s.strip()]) if solutions_txt else 0
        total = mem_count + sol_count

        # Build compact output summary
        output_parts = []
        if mem_count:
            output_parts.append(f"{mem_count} memories")
        if sol_count:
            output_parts.append(f"{sol_count} solutions")
        output_summary = ", ".join(output_parts) if output_parts else "no results"

        # Full recalled text as retriever output (truncated to 4000 chars)
        full_output = ""
        if memories_txt:
            full_output += f"### Memories\n\n{memories_txt}"
        if solutions_txt:
            full_output += f"\n\n### Solutions\n\n{solutions_txt}"
        full_output = full_output[:4000] + ("\n…" if len(full_output) > 4000 else "")

        # Derive recall query from current user message (same logic as _90_langfuse_trace.py)
        query_text = ""
        if loop_data.user_message:
            try:
                _qraw = loop_data.user_message.output_text()
                _q = _qraw.split(": ", 1)[-1].strip() if ": " in _qraw else _qraw.strip()
                if _q.startswith("{") and _q.endswith("}"):
                    _qp = json.loads(_q)
                    if isinstance(_qp, dict):
                        _q = str(_qp.get("user_message") or _qp.get("message") or _qp.get("text") or _q)
                query_text = _q[:500]  # cap query preview at 500 chars
            except Exception:
                pass

        try:
            span = parent.start_observation(
                name="memory-recall",
                as_type="retriever",
                input=query_text or None,
                output=full_output or output_summary,
                metadata={
                    "memories_found": mem_count,
                    "solutions_found": sol_count,
                    "total_found": total,
                    "iteration": loop_data.iteration,
                    "agent_number": self.agent.number,
                    "agent_profile": self.agent.config.profile,
                },
            )
            span.end()
        except Exception:
            pass

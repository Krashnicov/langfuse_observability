import json
import os

from helpers.api import ApiHandler, Input, Output, Request, Response
from helpers import files
from plugins._model_config.helpers.model_config import build_utility_model

_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PromptJudge(ApiHandler):
    async def process(self, input: Input, request: Request) -> Output:
        original_prompt = input.get("original_prompt", "")
        original_response = input.get("original_response", "")
        variants = input.get("variants", [])

        if not original_prompt:
            return {"success": False, "error": "original_prompt is required"}
        if not variants:
            return {"success": False, "error": "variants is required"}

        # Load judge system prompt from plugin's prompts/ directory
        judge_prompt_path = os.path.join(_PLUGIN_ROOT, "prompts", "prompt_judge.md")
        try:
            judge_prompt = files.read_file(judge_prompt_path)
        except Exception:
            return {"success": False, "error": "Judge prompt template not found"}

        # Build the input for the judge
        variants_text = ""
        for i, v in enumerate(variants):
            prompt_text = v.get("prompt", "") if isinstance(v, dict) else str(v)
            explanation = v.get("explanation", "") if isinstance(v, dict) else ""
            variants_text += (
                f"### Variant {i}\n\n"
                f"**Prompt:**\n{prompt_text}\n\n"
                f"**Explanation:** {explanation}\n\n"
            )

        judge_input = (
            f"## Original System Prompt\n\n{original_prompt}\n\n"
            f"## Original Response\n\n{original_response}\n\n"
            f"## Variants to Judge\n\n{variants_text}"
        )

        try:
            llm = build_utility_model()

            content, _reasoning = await llm.unified_call(
                system_message=judge_prompt,
                user_message=judge_input,
            )

            # Parse JSON — strip markdown code fences if present
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            results = json.loads(content)

            return {
                "success": True,
                "results": results,
            }
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Failed to parse judge output: {e}"}
        except Exception as e:
            return {"success": False, "error": f"Judge failed: {e}"}

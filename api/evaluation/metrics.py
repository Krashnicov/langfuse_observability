"""Metric catalog and factory for deepeval evaluation. STORY-009 AC-9.4."""
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# ── sys.path (3 levels up from api/evaluation/) ──────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PLUGIN_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PLUGIN_ROOT not in sys.path:
    sys.path.append(_PLUGIN_ROOT)


@dataclass
class MetricConfig:
    """Configuration descriptor for a named evaluation metric. AC-9.4"""
    name: str
    display_name: str
    description: str
    default_criteria: str
    score_range: tuple = (0, 10)


# AC-9.4: METRIC_CATALOG must contain at minimum: geval, task_completion, tool_correctness
METRIC_CATALOG: Dict[str, MetricConfig] = {
    "geval": MetricConfig(  # AC-9.4
        name="geval",
        display_name="GEval Answer Quality",
        description="LLM-as-judge evaluation using GEval criteria scoring (0-10 scale)",
        default_criteria=(
            "Evaluate whether the actual output is a high-quality answer to the input. "
            "Score 1-10 where 10 is perfect."
        ),
    ),
    "task_completion": MetricConfig(  # AC-9.4
        name="task_completion",
        display_name="Task Completion",
        description="Measures whether the LLM successfully completed the assigned task",
        default_criteria=(
            "Did the model successfully and completely fulfill the user's request? "
            "Score 1-10 where 10 means fully complete."
        ),
    ),
    "tool_correctness": MetricConfig(  # AC-9.4
        name="tool_correctness",
        display_name="Tool Correctness",
        description="Measures whether the model selected and used tools correctly",
        default_criteria=(
            "Were the tools selected and used correctly and appropriately for the task? "
            "Score 1-10 where 10 means perfect tool usage."
        ),
    ),
}


def create_metric(metric_name: str, judge_model: str, criteria: Optional[str] = None) -> Any:
    """Factory: instantiate a deepeval metric by catalog name. AC-9.4

    Satisfies: AC-9.4

    Args:
        metric_name: Key in METRIC_CATALOG.
        judge_model: LLM model identifier for the judge (e.g. 'gpt-4.1').
        criteria: Override default criteria string; uses catalog default if None.

    Returns:
        A configured deepeval GEval metric instance.

    Raises:
        ValueError: If metric_name is not in METRIC_CATALOG. AC-9.4
    """
    # AC-9.4: raise ValueError for unknown metric names
    if metric_name not in METRIC_CATALOG:  # AC-9.4
        raise ValueError(
            f"Unknown metric '{metric_name}'. "
            f"Available metrics: {sorted(METRIC_CATALOG.keys())}"
        )

    config = METRIC_CATALOG[metric_name]
    effective_criteria = criteria or config.default_criteria

    # Lazy import — OPENAI_API_KEY must already be set by EvaluationPipeline.__init__
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams

    return GEval(
        name=config.display_name,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        criteria=effective_criteria,
        model=judge_model,
        threshold=0.5,
    )

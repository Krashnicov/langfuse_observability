"""Tests for EvaluationPipeline — STORY-009 TDD anchors."""
import json
import os
import sys

import pytest
from unittest.mock import MagicMock, patch

# ─── Path setup ───────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PLUGIN_ROOT = os.path.dirname(_HERE)
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)


# ─── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_api():
    api = MagicMock()
    api.get_trace.return_value = {
        "id": "trace-1",
        "input": "What is 2+2?",
        "output": "4",
    }
    api.create_score.return_value = {"id": "score-1"}
    return api


@pytest.fixture
def mock_belief_store():
    return MagicMock()


@pytest.fixture
def mock_updater():
    updater = MagicMock()
    updater.update.return_value = MagicMock()
    return updater


@pytest.fixture
def pipeline(mock_api, mock_belief_store, mock_updater, tmp_path):
    """Construct a ready EvaluationPipeline with temp config.json."""
    config = {"openai_api_key": "sk-test-fixture"}
    (tmp_path / "config.json").write_text(json.dumps(config))

    from api.evaluation.pipeline import EvaluationPipeline
    return EvaluationPipeline(
        api=mock_api,
        belief_store=mock_belief_store,
        updater=mock_updater,
        judge_model="gpt-4.1",
        config_path=str(tmp_path / "config.json"),
    )


# ─── TDD Anchor 1 ─────────────────────────────────────────────────────────

def test_pipeline_sets_telemetry_opt_out_on_init(mock_api, tmp_path):
    """EvaluationPipeline.__init__ sets DEEPEVAL_TELEMETRY_OPT_OUT=True. AC-9.2"""
    config = {}
    (tmp_path / "config.json").write_text(json.dumps(config))

    from api.evaluation.pipeline import EvaluationPipeline
    os.environ.pop("DEEPEVAL_TELEMETRY_OPT_OUT", None)
    EvaluationPipeline(
        api=mock_api,
        config_path=str(tmp_path / "config.json"),
    )
    assert os.environ.get("DEEPEVAL_TELEMETRY_OPT_OUT") == "True"  # AC-9.2


# ─── TDD Anchor 2 ─────────────────────────────────────────────────────────

def test_pipeline_does_not_set_confident_api_key(mock_api, tmp_path):
    """EvaluationPipeline.__init__ does NOT set CONFIDENT_API_KEY. AC-9.2"""
    config = {}
    (tmp_path / "config.json").write_text(json.dumps(config))

    from api.evaluation.pipeline import EvaluationPipeline
    os.environ.pop("CONFIDENT_API_KEY", None)
    EvaluationPipeline(
        api=mock_api,
        config_path=str(tmp_path / "config.json"),
    )
    assert "CONFIDENT_API_KEY" not in os.environ  # AC-9.2


# ─── TDD Anchor 3 ─────────────────────────────────────────────────────────

def test_pipeline_raises_import_error_when_deepeval_missing(mock_api, tmp_path):
    """ImportError raised with install hint when deepeval not available. AC-9.9"""
    config = {}
    (tmp_path / "config.json").write_text(json.dumps(config))

    import api.evaluation.pipeline as pipeline_module
    original = pipeline_module.DEEPEVAL_AVAILABLE
    try:
        pipeline_module.DEEPEVAL_AVAILABLE = False
        with pytest.raises(ImportError, match="deepeval is required"):  # AC-9.9
            pipeline_module.EvaluationPipeline(
                api=mock_api,
                config_path=str(tmp_path / "config.json"),
            )
    finally:
        pipeline_module.DEEPEVAL_AVAILABLE = original


# ─── TDD Anchor 4 ─────────────────────────────────────────────────────────

def test_inject_openai_key_reads_from_config(pipeline, tmp_path):
    """_inject_openai_key reads openai_api_key from config.json. AC-9.3"""
    config = {"openai_api_key": "sk-test-specific-key"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    pipeline._config_path = str(tmp_path / "config.json")

    os.environ.pop("OPENAI_API_KEY", None)
    pipeline._inject_openai_key()
    assert os.environ.get("OPENAI_API_KEY") == "sk-test-specific-key"  # AC-9.3


# ─── TDD Anchor 5 ─────────────────────────────────────────────────────────

def test_get_geval_distribution_returns_dict_when_logprobs_available(pipeline):
    """_get_geval_distribution returns logprob dict when attribute present. AC-9.7"""
    mock_metric = MagicMock()
    mock_metric.logprob_distribution = {10: 0.8, 9: 0.2}
    result = pipeline._get_geval_distribution(mock_metric)
    assert result == {10: 0.8, 9: 0.2}  # AC-9.7


# ─── TDD Anchor 6 ─────────────────────────────────────────────────────────

def test_get_geval_distribution_returns_point_estimate_fallback_when_none(pipeline):
    """_get_geval_distribution returns {score: 1.0} when logprobs=None. AC-9.7"""
    mock_metric = MagicMock()
    mock_metric.logprob_distribution = None
    mock_metric.score = 8
    result = pipeline._get_geval_distribution(mock_metric)
    assert result == {8: 1.0}  # AC-9.7: point-estimate fallback


# ─── TDD Anchor 7 ─────────────────────────────────────────────────────────

def test_pipe_score_to_langfuse_calls_create_score_with_api_source(pipeline):
    """_pipe_score_to_langfuse calls create_score with source='API'. AC-9.6"""
    pipeline._pipe_score_to_langfuse("trace-1", "geval", 0.9, "Good")
    pipeline._api.create_score.assert_called_once()
    call_kwargs = pipeline._api.create_score.call_args[1]
    assert call_kwargs["source"] == "API"  # AC-9.6


# ─── TDD Anchor 8 ─────────────────────────────────────────────────────────

def test_pipe_score_to_langfuse_passes_all_fields(pipeline):
    """_pipe_score_to_langfuse forwards all required fields. AC-9.6"""
    pipeline._pipe_score_to_langfuse("trace-1", "geval", 0.9, "Good")
    call_kwargs = pipeline._api.create_score.call_args[1]
    assert call_kwargs["name"] == "geval"        # AC-9.6
    assert call_kwargs["value"] == 0.9           # AC-9.6
    assert call_kwargs["trace_id"] == "trace-1"  # AC-9.6
    assert call_kwargs["data_type"] == "NUMERIC"  # AC-9.6
    assert call_kwargs["comment"] == "Good"       # AC-9.6


# ─── TDD Anchor 9 ─────────────────────────────────────────────────────────

def test_evaluate_traces_returns_required_keys(pipeline):
    """evaluate_traces result dict has required keys. AC-9.8"""
    with patch.object(pipeline, "_evaluate_single_trace") as mock_eval:
        mock_eval.return_value = {
            "trace_id": "trace-1",
            "scores": [{"metric": "geval", "score": 0.9}],
            "scores_written": 1,
            "belief_updates": 1,
        }
        result = pipeline.evaluate_traces(["trace-1"])

    assert "results" in result        # AC-9.8
    assert "scores_written" in result  # AC-9.8
    assert "belief_updates" in result  # AC-9.8
    assert "errors" in result          # AC-9.8


# ─── TDD Anchor 10 ────────────────────────────────────────────────────────

def test_evaluate_traces_isolates_individual_errors(pipeline):
    """evaluate_traces catches per-trace errors; batch continues. AC-9.8"""
    call_count = 0

    def side_effect(trace_id, metrics, criteria):
        nonlocal call_count
        call_count += 1
        if trace_id == "bad-trace":
            raise RuntimeError("Simulated trace failure")
        return {"trace_id": trace_id, "scores": [], "scores_written": 0, "belief_updates": 0}

    with patch.object(pipeline, "_evaluate_single_trace", side_effect=side_effect):
        result = pipeline.evaluate_traces(["good-trace", "bad-trace", "good-trace-2"])

    assert len(result["errors"]) == 1          # AC-9.8: error isolated
    assert "bad-trace" in result["errors"][0]  # AC-9.8: error contains trace id
    assert call_count == 3                     # AC-9.8: all traces attempted


# ─── TDD Anchor 11 ────────────────────────────────────────────────────────

def test_metric_catalog_contains_required_metrics():
    """METRIC_CATALOG contains geval, task_completion, tool_correctness. AC-9.4"""
    from api.evaluation.metrics import METRIC_CATALOG
    assert "geval" in METRIC_CATALOG          # AC-9.4
    assert "task_completion" in METRIC_CATALOG  # AC-9.4
    assert "tool_correctness" in METRIC_CATALOG  # AC-9.4


# ─── TDD Anchor 12 ────────────────────────────────────────────────────────

def test_create_metric_raises_value_error_on_unknown_name():
    """create_metric raises ValueError for unknown metric name. AC-9.4"""
    from api.evaluation.metrics import create_metric
    with pytest.raises(ValueError, match="nonexistent"):  # AC-9.4
        create_metric("nonexistent", "gpt-4.1")

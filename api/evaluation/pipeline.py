"""EvaluationPipeline — deepeval headless evaluation engine. STORY-009."""
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

# ── sys.path (plugin root = 3 levels up from api/evaluation/) ─────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PLUGIN_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PLUGIN_ROOT not in sys.path:
    sys.path.append(_PLUGIN_ROOT)

logger = logging.getLogger(__name__)

# ── Headless env vars BEFORE any deepeval import ──────────────────────────────
# AC-9.2: set telemetry opt-out before the import so deepeval init reads it
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "True"   # AC-9.2
os.environ["DEEPEVAL_DISABLE_DOTENV"] = "1"          # AC-9.2: don't load .env
# AC-9.2: intentionally do NOT set CONFIDENT_API_KEY

# ── Optional deepeval import ─────────────────────────────────────────────────
# AC-9.9: DEEPEVAL_AVAILABLE is a module-level flag; tests can set it to False
try:
    import deepeval  # noqa: F401
    DEEPEVAL_AVAILABLE = True
except ImportError:  # AC-9.9
    DEEPEVAL_AVAILABLE = False


class EvaluationPipeline:
    """Headless deepeval evaluation pipeline that pipes scores to Langfuse.

    Satisfies: AC-9.1, AC-9.2, AC-9.3, AC-9.5, AC-9.6, AC-9.7, AC-9.8, AC-9.9
    """

    def __init__(
        self,
        api: Any,
        belief_store: Optional[Any] = None,
        updater: Optional[Any] = None,
        judge_model: str = "gpt-4.1",
        config_path: str = "",
    ) -> None:
        """Initialise the pipeline and configure headless deepeval mode.

        Satisfies: AC-9.2, AC-9.9

        Args:
            api: LangfuseObservabilityAPI instance (provides create_score).
            belief_store: Optional BeliefStore for persisting beliefs.
            updater: Optional BayesianUpdater for belief updates.
            judge_model: LLM identifier used as judge (default 'gpt-4.1').
            config_path: Path to config.json containing openai_api_key.
        """
        # AC-9.9: raise immediately if deepeval is not installed
        if not DEEPEVAL_AVAILABLE:  # AC-9.9
            raise ImportError(
                "deepeval is required for evaluation. "
                "Install with: pip install -r requirements-eval.txt"
            )

        self._api = api
        self._belief_store = belief_store
        self._updater = updater
        self._judge_model = judge_model
        self._config_path = config_path or os.path.join(_PLUGIN_ROOT, "config.json")

        # AC-9.2: configure headless mode (idempotent — may already be set at module level)
        self._configure_headless()

    def _configure_headless(self) -> None:
        """Set deepeval headless env vars. AC-9.2

        Satisfies: AC-9.2
        """
        os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "True"  # AC-9.2
        os.environ["DEEPEVAL_DISABLE_DOTENV"] = "1"         # AC-9.2
        # AC-9.2: explicitly do NOT set CONFIDENT_API_KEY — no cloud sync

    def _inject_openai_key(self) -> None:
        """Read openai_api_key from config.json and inject into env. AC-9.3

        Satisfies: AC-9.3

        This MUST be called before any deepeval metric class is imported
        at call time, because deepeval reads OPENAI_API_KEY during LLM init.
        Agent Zero sets OPENAI_API_KEY='proxy-a0' which is only valid for
        the local LiteLLM proxy — not for direct OpenAI API calls.
        """
        try:
            with open(self._config_path, "r") as fh:
                cfg = json.load(fh)
            key = cfg.get("openai_api_key", "")
            if key:  # AC-9.3: only overwrite if config provides a real key
                os.environ["OPENAI_API_KEY"] = key  # AC-9.3
                logger.debug("OPENAI_API_KEY injected from config.json (length=%d)", len(key))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read openai_api_key from %s: %s", self._config_path, exc)

    def _get_geval_distribution(self, metric: Any) -> Dict[int, float]:
        """Extract logprob distribution from a GEval metric result. AC-9.7

        Satisfies: AC-9.7

        Returns:
            {score_int: probability} dict when logprobs are available,
            OR {point_score: 1.0} as fallback when logprobs are None.
            Never raises — always returns a valid distribution dict.
        """
        dist = getattr(metric, "logprob_distribution", None)
        if dist is not None:  # AC-9.7: logprobs available
            return dist
        # AC-9.7: point-estimate fallback (e.g. GLM-5.1 returns logprobs=None)
        point_score = getattr(metric, "score", 5)
        return {point_score: 1.0}  # AC-9.7

    def _pipe_score_to_langfuse(
        self,
        trace_id: str,
        metric_name: str,
        score: float,
        reason: str,
        observation_id: Optional[str] = None,
    ) -> Optional[Dict]:
        """POST a score to Langfuse via the HTTP API. AC-9.6

        Satisfies: AC-9.6

        Calls self._api.create_score with all required fields and source='API'.
        """
        # AC-9.6: source='API' (Langfuse public API auth overrides EVAL→API)
        return self._api.create_score(
            name=metric_name,         # AC-9.6
            value=score,               # AC-9.6
            trace_id=trace_id,         # AC-9.6
            source="API",              # AC-9.6
            data_type="NUMERIC",       # AC-9.6
            comment=reason,            # AC-9.6
        )

    def _build_test_case(self, trace: Dict) -> Any:
        """Map a Langfuse trace dict to a deepeval LLMTestCase. AC-9.5

        Satisfies: AC-9.5
        """
        # AC-9.3: inject key before any deepeval metric import
        self._inject_openai_key()
        from deepeval.test_case import LLMTestCase  # lazy import after key injection

        return LLMTestCase(
            input=str(trace.get("input", "")),
            actual_output=str(trace.get("output", "")),
        )

    def _evaluate_single_trace(
        self,
        trace_id: str,
        metrics: Optional[List[str]] = None,
        criteria: Optional[Dict[str, str]] = None,
    ) -> Dict:
        """Evaluate one trace and pipe scores to Langfuse. AC-9.5

        Satisfies: AC-9.5, AC-9.6, AC-9.7

        Args:
            trace_id: Langfuse trace ID.
            metrics: List of metric names from METRIC_CATALOG. Defaults to ['geval'].
            criteria: Optional per-metric criteria overrides.

        Returns:
            {trace_id, scores, scores_written, belief_updates}
        """
        metrics = metrics or ["geval"]
        criteria = criteria or {}

        # AC-9.3: inject OpenAI key before any deepeval import
        self._inject_openai_key()

        from deepeval.test_case import LLMTestCase  # lazy import
        from api.evaluation.metrics import create_metric

        trace = self._api.get_trace(trace_id)
        test_case = LLMTestCase(
            input=str(trace.get("input", "")),
            actual_output=str(trace.get("output", "")),
        )

        scores_written = 0
        belief_updates = 0
        scores: List[Dict] = []

        for metric_name in metrics:
            try:
                metric = create_metric(
                    metric_name,
                    self._judge_model,
                    criteria.get(metric_name),
                )
                metric.measure(test_case)

                raw_score = metric.score or 0.0
                # Normalize GEval 0-10 to [0,1]
                normalized = raw_score / 10.0 if raw_score > 1.0 else raw_score
                reason = getattr(metric, "reason", "") or ""

                # AC-9.6: pipe score to Langfuse
                self._pipe_score_to_langfuse(trace_id, metric_name, normalized, reason)
                scores_written += 1

                score_entry = {"metric": metric_name, "score": normalized}
                scores.append(score_entry)

                # AC-9.7: get distribution for Bayesian update
                distribution = self._get_geval_distribution(metric)

                # AC-9.5: trigger BayesianUpdater if available
                if self._updater is not None:
                    self._updater.update(
                        entity_id=trace_id,
                        metric_name=metric_name,
                        score=normalized,
                        channel="EVAL",
                        distribution=distribution,
                    )
                    belief_updates += 1

            except Exception as exc:
                logger.warning(
                    "Metric '%s' failed for trace '%s': %s",
                    metric_name, trace_id, exc,
                )

        return {
            "trace_id": trace_id,
            "scores": scores,
            "scores_written": scores_written,
            "belief_updates": belief_updates,
        }

    def evaluate_trace(
        self,
        trace_id: str,
        metrics: Optional[List[str]] = None,
        criteria: Optional[Dict[str, str]] = None,
    ) -> Dict:
        """Single-trace convenience wrapper. AC-9.5

        Satisfies: AC-9.5
        """
        return self._evaluate_single_trace(trace_id, metrics, criteria)

    def evaluate_traces(
        self,
        trace_ids: List[str],
        metrics: Optional[List[str]] = None,
        criteria: Optional[Dict[str, str]] = None,
    ) -> Dict:
        """Batch evaluation with per-trace error isolation. AC-9.8

        Satisfies: AC-9.8

        Args:
            trace_ids: List of Langfuse trace IDs to evaluate.
            metrics: Metric names to run (default ['geval']).
            criteria: Per-metric criteria overrides.

        Returns:
            {
                'results': list of per-trace result dicts,
                'scores_written': total scores written to Langfuse,
                'belief_updates': total Bayesian updates triggered,
                'errors': list of error strings (trace_id + message),
            }
        """
        results: List[Dict] = []
        total_scores = 0
        total_updates = 0
        errors: List[str] = []  # AC-9.8: error list for per-trace failures

        for trace_id in trace_ids:  # AC-9.8: iterate ALL traces
            try:
                result = self._evaluate_single_trace(trace_id, metrics, criteria)  # AC-9.8
                results.append(result)
                total_scores += result.get("scores_written", 0)
                total_updates += result.get("belief_updates", 0)
            except Exception as exc:  # AC-9.8: catch per-trace errors
                # AC-9.8: error message contains trace_id; batch continues
                error_msg = f"{trace_id}: {exc}"  # AC-9.8
                errors.append(error_msg)
                logger.warning("evaluate_traces: error for trace '%s': %s", trace_id, exc)

        # AC-9.8: return dict with all required keys
        return {
            "results": results,           # AC-9.8
            "scores_written": total_scores,  # AC-9.8
            "belief_updates": total_updates,  # AC-9.8
            "errors": errors,             # AC-9.8
        }

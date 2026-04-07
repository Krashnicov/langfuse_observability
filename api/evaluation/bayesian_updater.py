"""
Bayesian Updater — conjugate Beta update logic, score normalization, drift detection.

AC coverage: AC-10.9, AC-10.10
"""
import os
import sys
from datetime import datetime, timezone
from typing import Optional

# sys.path: 2 levels up from api/evaluation/ to reach plugin root
_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PLUGIN_ROOT not in sys.path:
    sys.path.append(_PLUGIN_ROOT)

from api.belief_store.models import (
    BeliefState,
    EntityType,
    ScoreDataType,
    ScoreSource,
    TrustLevel,
)
from api.belief_store.store import BeliefStore


# ---------------------------------------------------------------------------
# AC-10.10: normalize_score
# ---------------------------------------------------------------------------

def normalize_score(
    value,
    data_type: ScoreDataType,
    categories: Optional[dict] = None,
) -> float:  # AC-10.10
    """Normalize a raw score value to [0,1].

    Rules (AC-10.10):
      BOOLEAN    → 0.0 if falsy, 1.0 if truthy
      NUMERIC    → clamp to [0,1]
      CATEGORICAL → lookup in categories dict; raises ValueError on unknown key

    Satisfies: AC-10.10
    """
    if data_type == ScoreDataType.BOOLEAN:  # AC-10.10: BOOLEAN → 0.0/1.0
        return 1.0 if value else 0.0

    elif data_type == ScoreDataType.NUMERIC:  # AC-10.10: NUMERIC → clamp [0,1]
        return float(max(0.0, min(1.0, float(value))))

    elif data_type == ScoreDataType.CATEGORICAL:  # AC-10.10: CATEGORICAL → dict lookup
        if categories is None:
            raise ValueError(  # AC-10.10: no category map provided
                "categories dict required for CATEGORICAL score type"
            )
        if value not in categories:  # AC-10.10: raises ValueError on unknown category
            raise ValueError(
                f"Unknown category '{value}'. Known: {list(categories.keys())}"
            )
        return float(categories[value])

    else:
        raise ValueError(f"Unknown ScoreDataType: {data_type}")


# ---------------------------------------------------------------------------
# AC-10.9: detect_drift helper
# ---------------------------------------------------------------------------

def detect_drift(mean_before: float, mean_after: float) -> bool:  # AC-10.9
    """Return True if |mean_after - mean_before| > 0.15 (drift threshold).

    Satisfies: AC-10.9
    """
    return abs(mean_after - mean_before) > 0.15  # AC-10.9: drift threshold


# ---------------------------------------------------------------------------
# AC-10.9: BayesianUpdater
# ---------------------------------------------------------------------------

class BayesianUpdater:  # AC-10.9
    """Apply O(1) conjugate Beta updates to BeliefState objects.

    Satisfies: AC-10.9
    """

    def __init__(self, store: BeliefStore) -> None:
        """Initialise with an existing BeliefStore.

        Args:
            store: BeliefStore instance (may be :memory: for tests).
        """
        self._store = store

    def update(
        self,
        entity_type: EntityType,
        entity_id: str,
        score_name: str,
        score_value: float,
        source: ScoreSource,
        distribution: Optional[dict] = None,
    ) -> BeliefState:  # AC-10.9
        """Apply conjugate Beta update and return updated BeliefState.

        Algorithm (AC-10.9):
          1. Load channel weight from store (configurable, default API=1.0 etc.)
          2. Compute α_inc / β_inc via distribution or point-estimate
          3. Apply conjugate update: α += α_inc; β += β_inc
          4. Recompute posterior (mean, variance, CI)
          5. Recompute trust level
          6. Persist to store
          7. Record score history
          8. Detect and record drift if |mean_after - mean_before| > 0.15

        Returns:
            Updated BeliefState.

        Satisfies: AC-10.9
        """
        # Step 1 — load belief (create if first observation)  # AC-10.9
        belief = self._store.get_or_create_belief(entity_type, entity_id, score_name)

        # Snapshot before-state
        alpha_before = belief.alpha
        beta_before = belief.beta
        mean_before = belief.posterior_mean
        trust_before = belief.trust_level

        # Step 2 — determine channel weight from store  # AC-10.9
        weight = self._store.get_channel_weight(source)

        # Step 3 — compute increments  # AC-10.9
        if distribution is not None:  # AC-10.9: GEval log-prob distribution path
            # distribution = {score_int: probability}, scores 0–10, probs sum to ≈1.0
            alpha_inc = sum(  # AC-10.9: weighted sum over distribution
                p * (s / 10.0) * weight for s, p in distribution.items()
            )
            beta_inc = sum(   # AC-10.9: complementary weighted sum
                p * (1.0 - s / 10.0) * weight for s, p in distribution.items()
            )
        else:  # AC-10.9: point-estimate fallback (API, ANNOTATION, zAI/GLM-5.1 EVAL)
            alpha_inc = score_value * weight          # AC-10.9: point-estimate α_inc
            beta_inc = (1.0 - score_value) * weight  # AC-10.9: point-estimate β_inc

        # Step 4 — conjugate update  # AC-10.9: O(1) Beta update
        belief.alpha += alpha_inc
        belief.beta += beta_inc

        # Step 5 — recompute posterior  # AC-10.9
        belief.compute_posterior()

        # Step 6 — recompute trust level  # AC-10.9
        belief.trust_level = belief.compute_trust_level()

        # Step 7 — update observation counters  # AC-10.9
        belief.total_observations += 1
        belief.last_observation_at = datetime.now(timezone.utc).isoformat()
        if source == ScoreSource.API:
            belief.api_observations += 1
        elif source == ScoreSource.ANNOTATION:
            belief.annotation_observations += 1
        elif source == ScoreSource.EVAL:
            belief.eval_observations += 1

        # Step 8 — persist  # AC-10.9
        self._store.save_belief(belief)

        # Step 9 — record score history  # AC-10.9 / AC-10.8
        self._store.record_score_history(
            belief_key=belief.belief_key,
            source=source,
            score_value=score_value,
            alpha_before=alpha_before,
            beta_before=beta_before,
            alpha_after=belief.alpha,
            beta_after=belief.beta,
            mean_before=mean_before,
            mean_after=belief.posterior_mean,
            trust_before=trust_before,
            trust_after=belief.trust_level,
            has_distribution=(distribution is not None),
        )

        # Step 10 — detect and record drift  # AC-10.9: drift threshold 0.15
        if detect_drift(mean_before, belief.posterior_mean):
            self._store.record_drift_event(
                belief_key=belief.belief_key,
                mean_before=mean_before,
                mean_after=belief.posterior_mean,
            )

        return belief

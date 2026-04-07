"""
STORY-010 -- Bayesian Updater tests
AC coverage: AC-10.2, AC-10.3, AC-10.9

TDD anchors (10):
  test_point_estimate_update_increases_alpha_on_high_score
  test_point_estimate_update_increases_beta_on_low_score
  test_annotation_weight_doubles_alpha_beta_increment
  test_distribution_update_uses_weighted_sum
  test_distribution_fallback_when_none
  test_posterior_mean_converges_toward_score
  test_drift_detected_when_mean_shifts_above_threshold
  test_no_drift_when_shift_below_threshold
  test_trust_level_trusted_at_high_mean_low_ci
  test_trust_level_provisional_insufficient_observations
"""
import unittest

from api.belief_store.models import BeliefState, EntityType, ScoreSource, TrustLevel
from api.belief_store.store import BeliefStore
from api.evaluation.bayesian_updater import BayesianUpdater


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_updater():
    """Return (BayesianUpdater, BeliefStore) backed by in-memory SQLite."""
    store = BeliefStore(db_path=':memory:')
    updater = BayesianUpdater(store=store)
    return updater, store


# ---------------------------------------------------------------------------
# AC-10.9: BayesianUpdater.update — point-estimate path
# ---------------------------------------------------------------------------

class TestPointEstimateUpdate(unittest.TestCase):
    """AC-10.9: conjugate Beta update via point-estimate."""

    def test_point_estimate_update_increases_alpha_on_high_score(self):
        """AC-10.9: update(score=0.9, source=API) → alpha > 1.0 (started at uniform Beta(1,1))."""
        updater, _ = _make_updater()
        belief = updater.update(
            EntityType.SKILL, 'agent-1', 'quality', 0.9, ScoreSource.API
        )
        # AC-10.9: high score increases alpha
        self.assertGreater(belief.alpha, 1.0)

    def test_point_estimate_update_increases_beta_on_low_score(self):
        """AC-10.9: update(score=0.1, source=API) → beta > 1.0 AND beta > alpha."""
        updater, _ = _make_updater()
        belief = updater.update(
            EntityType.SKILL, 'agent-2', 'quality', 0.1, ScoreSource.API
        )
        # AC-10.9: low score increases beta more than alpha
        self.assertGreater(belief.beta, 1.0)
        self.assertGreater(belief.beta, belief.alpha)

    def test_annotation_weight_doubles_alpha_beta_increment(self):
        """AC-10.9: ANNOTATION weight=2.0 produces 2× alpha_inc vs API weight=1.0."""
        updater_api, _ = _make_updater()
        updater_ann, _ = _make_updater()
        b_api = updater_api.update(
            EntityType.SKILL, 'e-api', 'quality', 0.5, ScoreSource.API
        )
        b_ann = updater_ann.update(
            EntityType.SKILL, 'e-ann', 'quality', 0.5, ScoreSource.ANNOTATION
        )
        api_alpha_inc = b_api.alpha - 1.0
        ann_alpha_inc = b_ann.alpha - 1.0
        # AC-10.9: ANNOTATION increment is 2× API increment
        self.assertAlmostEqual(ann_alpha_inc, 2.0 * api_alpha_inc, places=5)

    def test_distribution_update_uses_weighted_sum(self):
        """AC-10.9: distribution={10:0.8, 9:0.2}, EVAL → alpha_inc=(0.8*1.0+0.2*0.9)*0.5=0.49."""
        updater, _ = _make_updater()
        belief = updater.update(
            EntityType.SKILL, 'agent-3', 'quality',
            score_value=0.0,          # ignored when distribution provided
            source=ScoreSource.EVAL,
            distribution={10: 0.8, 9: 0.2},
        )
        # AC-10.9: alpha = 1.0 + 0.49 = 1.49
        self.assertAlmostEqual(belief.alpha, 1.49, places=5)

    def test_distribution_fallback_when_none(self):
        """AC-10.9: distribution=None → point-estimate used, no error raised."""
        updater, _ = _make_updater()
        # AC-10.9: degrade gracefully — no exception
        belief = updater.update(
            EntityType.SKILL, 'agent-4', 'quality', 0.7, ScoreSource.EVAL,
            distribution=None
        )
        self.assertGreater(belief.alpha, 1.0)

    def test_posterior_mean_converges_toward_score(self):
        """AC-10.9: 10 updates with score=0.8 → posterior_mean > 0.6."""
        updater, _ = _make_updater()
        belief = None
        for _ in range(10):
            belief = updater.update(
                EntityType.SKILL, 'agent-5', 'quality', 0.8, ScoreSource.API
            )
        # AC-10.9: posterior converges toward high-quality score
        self.assertGreater(belief.posterior_mean, 0.6)


# ---------------------------------------------------------------------------
# AC-10.9: drift detection
# ---------------------------------------------------------------------------

class TestDriftDetection(unittest.TestCase):
    """AC-10.9: drift recorded when |mean_after - mean_before| > 0.15."""

    def test_drift_detected_when_mean_shifts_above_threshold(self):
        """AC-10.9: build low-quality belief (mean≈0.3 per anchor) then flip with high scores → drift recorded."""
        updater, store = _make_updater()
        # 3 low-score API updates → alpha=1.3, beta=3.7, mean≈0.26 (matches anchor 'mean≈0.3')
        for _ in range(3):
            updater.update(EntityType.SKILL, 'drift-agent', 'quality', 0.1, ScoreSource.API)
        # High-score ANNOTATION update: alpha_inc=1.9 → mean shifts from 0.26 to 0.46 (Δ=0.197>0.15)
        for _ in range(5):
            updater.update(
                EntityType.SKILL, 'drift-agent', 'quality', 0.95, ScoreSource.ANNOTATION
            )
        key = f"{EntityType.SKILL.value}:drift-agent:quality"
        events = store.get_drift_events(key)
        # AC-10.9: at least one drift event captured (first ANNOTATION update triggers Δ>0.15)
        self.assertGreater(len(events), 0)

    def test_no_drift_when_shift_below_threshold(self):
        """AC-10.9: single mid-score update from uniform prior → no drift."""
        updater, store = _make_updater()
        updater.update(EntityType.SKILL, 'stable-agent', 'quality', 0.52, ScoreSource.API)
        key = f"{EntityType.SKILL.value}:stable-agent:quality"
        events = store.get_drift_events(key)
        # AC-10.9: prior mean=0.5, new mean≈0.5 → |delta| < 0.15, no drift
        self.assertEqual(len(events), 0)


# ---------------------------------------------------------------------------
# AC-10.2: BeliefState trust level thresholds
# ---------------------------------------------------------------------------

class TestTrustLevel(unittest.TestCase):
    """AC-10.2: compute_trust_level() thresholds (TRUSTED/PROVISIONAL/SUSPENDED/REJECTED)."""

    def test_trust_level_trusted_at_high_mean_low_ci(self):
        """AC-10.2: Beta(80,20) → mean=0.8, ci_width≈0.13 → TRUSTED."""
        belief = BeliefState(
            entity_type=EntityType.SKILL,
            entity_id='trusted-e',
            score_name='quality',
            alpha=80.0,
            beta=20.0,
        )
        belief.compute_posterior()
        result = belief.compute_trust_level()
        # AC-10.2: mean≥0.8 AND ci_width≤0.2 → TRUSTED
        self.assertEqual(result, TrustLevel.TRUSTED)

    def test_trust_level_provisional_insufficient_observations(self):
        """AC-10.2: Beta(8,2) → mean=0.8 but ci_width≈0.39 → NOT TRUSTED (too few obs)."""
        belief = BeliefState(
            entity_type=EntityType.SKILL,
            entity_id='prov-e',
            score_name='quality',
            alpha=8.0,
            beta=2.0,
        )
        belief.compute_posterior()
        result = belief.compute_trust_level()
        # AC-10.2: CI too wide despite high mean — not yet TRUSTED
        self.assertNotEqual(result, TrustLevel.TRUSTED)


if __name__ == '__main__':
    unittest.main()

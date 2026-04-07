"""
STORY-010 -- Belief Store tests
AC coverage: AC-10.4 through AC-10.8

TDD anchors (8):
  test_store_creates_db_file_on_init
  test_get_or_create_returns_uniform_prior_for_new_entity
  test_get_or_create_returns_existing_belief
  test_save_belief_persists_and_retrieves
  test_list_beliefs_filters_by_trust_level
  test_record_score_history_appends_row
  test_channel_weights_defaults_on_init
  test_set_channel_weight_overrides_default
"""
import os
import unittest
import tempfile

from api.belief_store.models import BeliefState, EntityType, ScoreSource, TrustLevel
from api.belief_store.store import BeliefStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory_store():
    """Return BeliefStore backed by in-memory SQLite — zero file I/O."""
    return BeliefStore(db_path=':memory:')


def _make_trusted_belief(entity_id: str) -> BeliefState:
    """Return a BeliefState with TRUSTED level via Beta(80,20)."""
    belief = BeliefState(
        entity_type=EntityType.SKILL,
        entity_id=entity_id,
        score_name='quality',
        alpha=80.0,
        beta=20.0,
    )
    belief.compute_posterior()
    belief.trust_level = belief.compute_trust_level()
    return belief


def _make_provisional_belief(entity_id: str) -> BeliefState:
    """Return a BeliefState with PROVISIONAL level via Beta(6,4)."""
    belief = BeliefState(
        entity_type=EntityType.SKILL,
        entity_id=entity_id,
        score_name='quality',
        alpha=6.0,
        beta=4.0,
    )
    belief.compute_posterior()
    belief.trust_level = belief.compute_trust_level()
    return belief


# ---------------------------------------------------------------------------
# AC-10.4: BeliefStore.__init__ — schema creation + default weights
# ---------------------------------------------------------------------------

class TestBeliefStoreInit(unittest.TestCase):
    """AC-10.4: BeliefStore creates DB file and schema on init."""

    def test_store_creates_db_file_on_init(self):
        """AC-10.4: BeliefStore(db_path) creates the file on disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'b.db')
            self.assertFalse(os.path.exists(db_path))
            BeliefStore(db_path=db_path)
            # AC-10.4: file created on init
            self.assertTrue(os.path.exists(db_path))

    def test_channel_weights_defaults_on_init(self):
        """AC-10.4: default channel weights inserted — ANNOTATION==2.0 on fresh store."""
        store = _make_memory_store()
        # AC-10.4: default ANNOTATION weight
        self.assertEqual(store.get_channel_weight(ScoreSource.ANNOTATION), 2.0)

    def test_channel_weights_all_defaults_on_init(self):
        """AC-10.4: all three default weights inserted correctly."""
        store = _make_memory_store()
        # AC-10.4: API=1.0, ANNOTATION=2.0, EVAL=0.5
        self.assertEqual(store.get_channel_weight(ScoreSource.API), 1.0)
        self.assertEqual(store.get_channel_weight(ScoreSource.ANNOTATION), 2.0)
        self.assertEqual(store.get_channel_weight(ScoreSource.EVAL), 0.5)


# ---------------------------------------------------------------------------
# AC-10.5: get_or_create_belief — uniform prior for new entity
# ---------------------------------------------------------------------------

class TestGetOrCreate(unittest.TestCase):
    """AC-10.5: get_or_create_belief returns uniform Beta(1,1) for new entity."""

    def test_get_or_create_returns_uniform_prior_for_new_entity(self):
        """AC-10.5: new entity gets alpha=1.0, beta=1.0 (uniform prior)."""
        store = _make_memory_store()
        belief = store.get_or_create_belief(EntityType.SKILL, 'web-search', 'geval')
        # AC-10.5: uniform prior
        self.assertEqual(belief.alpha, 1.0)
        self.assertEqual(belief.beta, 1.0)

    def test_get_or_create_returns_existing_belief(self):
        """AC-10.5: second call returns the persisted belief, not a fresh one."""
        store = _make_memory_store()
        b1 = store.get_or_create_belief(EntityType.SKILL, 'web-search', 'geval')
        b1.alpha = 5.5
        b1.beta = 2.3
        store.save_belief(b1)
        b2 = store.get_or_create_belief(EntityType.SKILL, 'web-search', 'geval')
        # AC-10.5: existing belief returned — values preserved
        self.assertAlmostEqual(b2.alpha, 5.5, places=5)
        self.assertAlmostEqual(b2.beta, 2.3, places=5)


# ---------------------------------------------------------------------------
# AC-10.6: save_belief — idempotent upsert
# ---------------------------------------------------------------------------

class TestSaveBelief(unittest.TestCase):
    """AC-10.6: save_belief upserts correctly."""

    def test_save_belief_persists_and_retrieves(self):
        """AC-10.6: save belief with alpha=5, beta=2; retrieve via get_belief; values match."""
        store = _make_memory_store()
        belief = BeliefState(
            entity_type=EntityType.SKILL,
            entity_id='code-gen',
            score_name='correctness',
            alpha=5.0,
            beta=2.0,
        )
        belief.compute_posterior()
        belief.trust_level = belief.compute_trust_level()
        store.save_belief(belief)
        retrieved = store.get_belief(
            EntityType.SKILL, 'code-gen', 'correctness'
        )
        # AC-10.6: round-trip preserves alpha and beta
        self.assertIsNotNone(retrieved)
        self.assertAlmostEqual(retrieved.alpha, 5.0, places=5)
        self.assertAlmostEqual(retrieved.beta, 2.0, places=5)


# ---------------------------------------------------------------------------
# AC-10.7: list_beliefs — filtered query
# ---------------------------------------------------------------------------

class TestListBeliefs(unittest.TestCase):
    """AC-10.7: list_beliefs returns filtered BeliefState objects."""

    def test_list_beliefs_filters_by_trust_level(self):
        """AC-10.7: save TRUSTED + PROVISIONAL; filter TRUSTED returns only TRUSTED."""
        store = _make_memory_store()
        trusted = _make_trusted_belief('trusted-skill')
        provisional = _make_provisional_belief('prov-skill')
        store.save_belief(trusted)
        store.save_belief(provisional)
        results = store.list_beliefs(trust_level=TrustLevel.TRUSTED)
        # AC-10.7: only TRUSTED beliefs returned
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].trust_level, TrustLevel.TRUSTED)


# ---------------------------------------------------------------------------
# AC-10.8: record_score_history + get_score_history
# ---------------------------------------------------------------------------

class TestScoreHistory(unittest.TestCase):
    """AC-10.8: record_score_history appends rows to score_history table."""

    def test_record_score_history_appends_row(self):
        """AC-10.8: record 3 history entries; get_score_history returns 3 rows."""
        store = _make_memory_store()
        belief = store.get_or_create_belief(EntityType.SKILL, 'hist-agent', 'quality')
        for i in range(3):
            store.record_score_history(
                belief_key=belief.belief_key,
                source=ScoreSource.API,
                score_value=0.7 + i * 0.05,
                alpha_before=belief.alpha,
                beta_before=belief.beta,
                alpha_after=belief.alpha + 0.5,
                beta_after=belief.beta + 0.1,
                mean_before=belief.posterior_mean,
                mean_after=belief.posterior_mean + 0.02,
                trust_before=belief.trust_level,
                trust_after=belief.trust_level,
                has_distribution=False,
            )
        rows = store.get_score_history(belief.belief_key, limit=10)
        # AC-10.8: 3 rows recorded
        self.assertEqual(len(rows), 3)


# ---------------------------------------------------------------------------
# AC-10.4 (channel weights): set_channel_weight
# ---------------------------------------------------------------------------

class TestChannelWeights(unittest.TestCase):
    """AC-10.4: channel weights can be overridden via set_channel_weight."""

    def test_set_channel_weight_overrides_default(self):
        """AC-10.4: set_channel_weight(EVAL, 1.0) → get_channel_weight(EVAL) == 1.0."""
        store = _make_memory_store()
        store.set_channel_weight(ScoreSource.EVAL, 1.0)
        # AC-10.4: override persisted
        self.assertEqual(store.get_channel_weight(ScoreSource.EVAL), 1.0)


if __name__ == '__main__':
    unittest.main()

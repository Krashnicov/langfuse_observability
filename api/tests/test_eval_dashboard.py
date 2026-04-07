"""
STORY-012: Eval Dashboard Data — TDD Anchors
10 tests covering EvalDashboardData methods.
All 10 TDD anchors as specified in story-012-eval-dashboard.md.
"""
import os
import sys
import sqlite3
import pytest
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

# sys.path: 3 levels up from api/tests/ to reach plugin root
_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PLUGIN_ROOT not in sys.path:
    sys.path.append(_PLUGIN_ROOT)

from api.belief_store.store import BeliefStore
from api.belief_store.models import BeliefState, EntityType, TrustLevel, ScoreSource
from api.evaluation.dashboard import EvalDashboardData  # AC-12.1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    """In-memory BeliefStore for full test isolation."""
    s = BeliefStore(':memory:')
    yield s
    s.close()


@pytest.fixture
def mock_api():
    """Minimal mock for the api dependency."""
    return MagicMock()


@pytest.fixture
def dashboard(store, mock_api):
    """EvalDashboardData with real in-memory store."""
    return EvalDashboardData(store, mock_api)


# ---------------------------------------------------------------------------
# Helper: build BeliefState without touching the network
# ---------------------------------------------------------------------------

def _make_belief(
    entity_type: EntityType,
    entity_id: str,
    score_name: str,
    trust_level: TrustLevel = TrustLevel.PROVISIONAL,
    api_obs: int = 0,
    annotation_obs: int = 0,
    eval_obs: int = 0,
    mean: float = 0.5,
) -> BeliefState:
    """Build a BeliefState and save to store fixture. Returns the belief."""
    total = api_obs + annotation_obs + eval_obs
    return BeliefState(
        entity_type=entity_type,
        entity_id=entity_id,
        score_name=score_name,
        alpha=1.0 + mean * max(total, 1),
        beta=1.0 + (1.0 - mean) * max(total, 1),
        total_observations=total,
        api_observations=api_obs,
        annotation_observations=annotation_obs,
        eval_observations=eval_obs,
        posterior_mean=mean,
        posterior_variance=0.01,
        ci_lower=0.1,
        ci_upper=0.9,
        ci_width=0.8,
        trust_level=trust_level,
    )


def _insert_history_row(store: BeliefStore, belief_key: str, source: str,
                        score_value: float, mean_after: float,
                        trust_after: str, recorded_at: str) -> None:
    """Direct SQL insert into score_history for controlled test data."""
    cur = store._conn.cursor()
    cur.execute("""
        INSERT INTO score_history (
            belief_key, source, score_value,
            alpha_before, beta_before, alpha_after, beta_after,
            mean_before, mean_after, trust_before, trust_after,
            has_distribution, recorded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        belief_key, source, score_value,
        1.0, 1.0, 2.0, 1.5,
        0.5, mean_after, 'PROVISIONAL', trust_after,
        0, recorded_at,
    ))
    store._conn.commit()


# ===========================================================================
# TDD Anchor 1: init stores dependencies
# ===========================================================================

def test_dashboard_init_stores_dependencies(store, mock_api):
    """Construct with mock store + mock api; assert ._store and ._api are set.

    TDD Anchor 1 / AC-12.2
    """
    dash = EvalDashboardData(store, mock_api)
    assert dash._store is store    # AC-12.2: belief_store stored
    assert dash._api is mock_api   # AC-12.2: api stored


# ===========================================================================
# TDD Anchor 2: get_entity_overview returns list
# ===========================================================================

def test_get_entity_overview_returns_list(store, mock_api):
    """Store with 2 beliefs for same entity; assert 1 entity entry with both metrics.

    TDD Anchor 2 / AC-12.3
    """
    store.save_belief(_make_belief(EntityType.AGENT, 'agent-1', 'geval', TrustLevel.TRUSTED))
    store.save_belief(_make_belief(EntityType.AGENT, 'agent-1', 'tool_correctness', TrustLevel.PROVISIONAL))

    dash = EvalDashboardData(store, mock_api)
    result = dash.get_entity_overview()

    assert isinstance(result, list)          # AC-12.3: returns list
    assert len(result) == 1                  # 1 entity (same entity_id)
    entry = result[0]
    assert entry['entity_id'] == 'agent-1'
    assert 'geval' in entry['metrics']           # AC-12.3: metrics dict
    assert 'tool_correctness' in entry['metrics'] # AC-12.3: both score_names present
    # Each metric dict has required keys
    m = entry['metrics']['geval']
    assert 'trust' in m
    assert 'mean' in m
    assert 'ci' in m
    assert 'observations' in m


# ===========================================================================
# TDD Anchor 3: get_entity_overview filters by entity_type
# ===========================================================================

def test_get_entity_overview_filters_by_entity_type(store, mock_api):
    """Store SKILL + AGENT beliefs; call with entity_type=SKILL; only skill returned.

    TDD Anchor 3 / AC-12.3
    """
    store.save_belief(_make_belief(EntityType.SKILL, 'skill-1', 'accuracy'))
    store.save_belief(_make_belief(EntityType.AGENT, 'agent-1', 'geval'))

    dash = EvalDashboardData(store, mock_api)
    result = dash.get_entity_overview(entity_type=EntityType.SKILL)  # AC-12.3: filter

    assert len(result) == 1                              # only skill entity
    assert result[0]['entity_type'] == 'skill'           # AC-12.3: EntityType.SKILL.value


# ===========================================================================
# TDD Anchor 4: overall_trust is minimum across metrics
# ===========================================================================

def test_get_entity_overview_overall_trust_is_minimum(store, mock_api):
    """Entity with TRUSTED geval + REJECTED tool_correctness; overall_trust=='REJECTED'.

    TDD Anchor 4 / AC-12.3
    """
    store.save_belief(_make_belief(EntityType.AGENT, 'agent-1', 'geval', TrustLevel.TRUSTED))
    store.save_belief(_make_belief(EntityType.AGENT, 'agent-1', 'tool_correctness', TrustLevel.REJECTED))

    dash = EvalDashboardData(store, mock_api)
    result = dash.get_entity_overview()

    assert len(result) == 1
    assert result[0]['overall_trust'] == 'REJECTED'   # AC-12.3: minimum (worst) trust


# ===========================================================================
# TDD Anchor 5: get_metric_timeline returns chronological list
# ===========================================================================

def test_get_metric_timeline_returns_chronological_list(store, mock_api):
    """Store with 3 history rows on different dates; assert list ordered ascending by date.

    TDD Anchor 5 / AC-12.4
    """
    b = _make_belief(EntityType.AGENT, 'agent-1', 'geval', TrustLevel.TRUSTED, api_obs=3)
    store.save_belief(b)
    key = b.belief_key

    # Insert 3 history rows at 3 different dates within last 30 days
    dates = [
        (datetime.now(timezone.utc) - timedelta(days=20)).isoformat(),
        (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
        (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(),
    ]
    for i, dt in enumerate(dates):
        _insert_history_row(store, key, 'API', 0.6 + i * 0.05,
                            0.6 + i * 0.05, 'TRUSTED', dt)

    dash = EvalDashboardData(store, mock_api)
    result = dash.get_metric_timeline(EntityType.AGENT, 'agent-1', 'geval', days=30)

    assert isinstance(result, list)                   # AC-12.4: returns list
    assert len(result) == 3                           # 3 distinct dates
    date_strs = [r['date'] for r in result]
    assert date_strs == sorted(date_strs)             # AC-12.4: ascending by date
    for r in result:
        assert 'date' in r                            # AC-12.4: YYYY-MM-DD
        assert 'mean' in r                            # AC-12.4: mean
        assert 'ci_lower' in r                        # AC-12.4: ci_lower
        assert 'ci_upper' in r                        # AC-12.4: ci_upper


# ===========================================================================
# TDD Anchor 6: get_metric_timeline returns empty for unknown
# ===========================================================================

def test_get_metric_timeline_returns_empty_for_unknown(store, mock_api):
    """Call with unknown entity/metric; assert returns [] not error.

    TDD Anchor 6 / AC-12.4
    """
    dash = EvalDashboardData(store, mock_api)
    result = dash.get_metric_timeline(EntityType.AGENT, 'nonexistent', 'unknown_metric')

    assert result == []    # AC-12.4: empty list, no exception


# ===========================================================================
# TDD Anchor 7: get_channel_breakdown reflects observations
# ===========================================================================

def test_get_channel_breakdown_reflects_observations(store, mock_api):
    """Belief with api_observations=5, annotation_observations=2, eval_observations=20;
    assert breakdown counts match.

    TDD Anchor 7 / AC-12.5
    """
    b = _make_belief(
        EntityType.AGENT, 'agent-1', 'geval', TrustLevel.TRUSTED,
        api_obs=5, annotation_obs=2, eval_obs=20,
    )
    store.save_belief(b)

    dash = EvalDashboardData(store, mock_api)
    result = dash.get_channel_breakdown(EntityType.AGENT, 'agent-1', 'geval')

    assert result['api']['count'] == 5           # AC-12.5: api count
    assert result['annotation']['count'] == 2    # AC-12.5: annotation count
    assert result['eval']['count'] == 20         # AC-12.5: eval count
    # mean_contribution keys present
    assert 'mean_contribution' in result['api']
    assert 'mean_contribution' in result['annotation']
    assert 'mean_contribution' in result['eval']


# ===========================================================================
# TDD Anchor 8: get_trust_summary counts all levels
# ===========================================================================

def test_get_trust_summary_counts_all_levels(store, mock_api):
    """Store TRUSTED×2 + PROVISIONAL×1 + SUSPENDED×1; assert trust_distribution counts.

    TDD Anchor 8 / AC-12.6
    """
    store.save_belief(_make_belief(EntityType.AGENT, 'a1', 'geval', TrustLevel.TRUSTED))
    store.save_belief(_make_belief(EntityType.AGENT, 'a2', 'geval', TrustLevel.TRUSTED))
    store.save_belief(_make_belief(EntityType.AGENT, 'a3', 'geval', TrustLevel.PROVISIONAL))
    store.save_belief(_make_belief(EntityType.AGENT, 'a4', 'geval', TrustLevel.SUSPENDED))

    dash = EvalDashboardData(store, mock_api)
    result = dash.get_trust_summary()

    assert result['total_entities'] == 4                        # AC-12.6: total count
    dist = result['trust_distribution']
    assert dist.get('TRUSTED', 0) == 2                          # AC-12.6: distribution
    assert dist.get('PROVISIONAL', 0) == 1
    assert dist.get('SUSPENDED', 0) == 1
    assert 'needs_attention' in result                          # AC-12.6: needs_attention


# ===========================================================================
# TDD Anchor 9: needs_attention excludes TRUSTED + PROVISIONAL
# ===========================================================================

def test_get_trust_summary_needs_attention_excludes_trusted_provisional(store, mock_api):
    """needs_attention list only contains SUSPENDED + REJECTED entities.

    TDD Anchor 9 / AC-12.6
    """
    store.save_belief(_make_belief(EntityType.AGENT, 'a1', 'geval', TrustLevel.TRUSTED))
    store.save_belief(_make_belief(EntityType.AGENT, 'a2', 'geval', TrustLevel.PROVISIONAL))
    store.save_belief(_make_belief(EntityType.AGENT, 'a3', 'geval', TrustLevel.SUSPENDED))
    store.save_belief(_make_belief(EntityType.AGENT, 'a4', 'geval', TrustLevel.REJECTED))

    dash = EvalDashboardData(store, mock_api)
    result = dash.get_trust_summary()
    attention = result['needs_attention']

    trust_values = {e['trust'] for e in attention}
    assert 'TRUSTED' not in trust_values       # AC-12.6: excluded
    assert 'PROVISIONAL' not in trust_values   # AC-12.6: excluded
    assert 'SUSPENDED' in trust_values         # AC-12.6: included
    assert 'REJECTED' in trust_values          # AC-12.6: included
    assert len(attention) == 2                 # exactly SUSPENDED + REJECTED
    # Each entry has required keys
    for e in attention:
        assert 'entity' in e
        assert 'trust' in e
        assert 'mean' in e


# ===========================================================================
# TDD Anchor 10: get_recent_evaluations makes no writes to store
# ===========================================================================

def test_get_recent_evaluations_no_writes_to_store(mock_api):
    """Mock store; call get_recent_evaluations(); assert no write methods called.

    TDD Anchor 10 / AC-12.7, AC-12.8
    """
    mock_store = MagicMock()
    mock_store.get_recent_history.return_value = []    # read-only stub

    dash = EvalDashboardData(mock_store, mock_api)
    result = dash.get_recent_evaluations()

    assert isinstance(result, list)                        # AC-12.7: returns list
    # AC-12.8: no write methods called
    mock_store.save_belief.assert_not_called()
    mock_store.record_score_history.assert_not_called()
    mock_store.record_drift_event.assert_not_called()
    mock_store.set_channel_weight.assert_not_called()
    # read method WAS called
    mock_store.get_recent_history.assert_called_once()

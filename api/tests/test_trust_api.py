"""
Tests for TrustLevelAPI — STORY-011 TDD anchors.

AC coverage: AC-11.1 through AC-11.6
"""
import os
import sys

_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PLUGIN_ROOT not in sys.path:
    sys.path.append(_PLUGIN_ROOT)

import pytest
from api.belief_store.store import BeliefStore
from api.belief_store.models import BeliefState, EntityType, TrustLevel
from api.evaluation.trust_api import TrustLevelAPI  # AC-11.1: importable


# ---------------------------------------------------------------------------
# Helpers — build canonical BeliefState fixtures
# ---------------------------------------------------------------------------

def _make_belief(
    store: BeliefStore,
    entity_type: EntityType,
    entity_id: str,
    score_name: str,
    alpha: float,
    beta: float,
    total_observations: int = 10,
) -> BeliefState:
    """Create, compute posterior + trust_level, save and return a BeliefState."""
    b = BeliefState(
        entity_type=entity_type,
        entity_id=entity_id,
        score_name=score_name,
        alpha=alpha,
        beta=beta,
        total_observations=total_observations,
    )
    b.compute_posterior()
    b.trust_level = b.compute_trust_level()
    store.save_belief(b)
    return b


def _make_trusted(store, entity_type, entity_id, score_name) -> BeliefState:
    """Beta(80,20) → mean=0.80, ci_width≈0.13 → TRUSTED."""
    return _make_belief(store, entity_type, entity_id, score_name, 80.0, 20.0, 100)


def _make_provisional(store, entity_type, entity_id, score_name) -> BeliefState:
    """Beta(30,20) → mean=0.60, ci_width≈0.23 → PROVISIONAL (mean>=0.6, ci_width<=0.3, not TRUSTED)."""
    return _make_belief(store, entity_type, entity_id, score_name, 30.0, 20.0, 50)


def _make_suspended(store, entity_type, entity_id, score_name) -> BeliefState:
    """Beta(4,6) → mean=0.40 → SUSPENDED (mean >= 0.4 but < 0.6)."""
    return _make_belief(store, entity_type, entity_id, score_name, 4.0, 6.0, 10)


def _make_rejected(store, entity_type, entity_id, score_name) -> BeliefState:
    """Beta(1,9) → mean=0.10 → REJECTED (mean < 0.4)."""
    return _make_belief(store, entity_type, entity_id, score_name, 1.0, 9.0, 10)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    """In-memory BeliefStore for test isolation."""
    s = BeliefStore(':memory:')
    yield s
    s.close()


@pytest.fixture
def api(store):
    """TrustLevelAPI backed by in-memory store."""
    return TrustLevelAPI(store)  # AC-11.1


# ---------------------------------------------------------------------------
# AC-11.2: get_trust_level
# ---------------------------------------------------------------------------

def test_get_trust_level_returns_provisional_for_unknown_entity(store, api):
    """AC-11.2: unknown entity → safe default PROVISIONAL (not TRUSTED, not REJECTED)."""
    result = api.get_trust_level(EntityType.SKILL, 'unknown-skill', 'geval')
    assert result == TrustLevel.PROVISIONAL  # AC-11.2: safe cold-start default


def test_get_trust_level_returns_stored_trust(store, api):
    """AC-11.2: saved TRUSTED belief → get_trust_level returns TRUSTED."""
    _make_trusted(store, EntityType.SKILL, 'agent-x', 'geval')
    result = api.get_trust_level(EntityType.SKILL, 'agent-x', 'geval')
    assert result == TrustLevel.TRUSTED  # AC-11.2: returns stored trust level


def test_get_trust_level_returns_rejected_for_low_mean(store, api):
    """AC-11.2: Beta(1,9) → mean=0.1 → REJECTED."""
    _make_rejected(store, EntityType.SKILL, 'poor-agent', 'geval')
    result = api.get_trust_level(EntityType.SKILL, 'poor-agent', 'geval')
    assert result == TrustLevel.REJECTED  # AC-11.2: low mean → REJECTED


# ---------------------------------------------------------------------------
# AC-11.3: get_belief_summary
# ---------------------------------------------------------------------------

def test_get_belief_summary_returns_all_metrics(store, api):
    """AC-11.3: 2 beliefs for same entity → metrics dict has 2 keys."""
    _make_trusted(store, EntityType.AGENT, 'agent-alpha', 'geval')
    _make_provisional(store, EntityType.AGENT, 'agent-alpha', 'correctness')

    summary = api.get_belief_summary(EntityType.AGENT, 'agent-alpha')

    assert summary['entity_type'] == 'agent'  # AC-11.3
    assert summary['entity_id'] == 'agent-alpha'  # AC-11.3
    assert 'geval' in summary['metrics']  # AC-11.3: both score_names present
    assert 'correctness' in summary['metrics']  # AC-11.3
    assert len(summary['metrics']) == 2  # AC-11.3: exactly 2 metrics
    assert summary['last_updated'] is not None  # AC-11.3


def test_get_belief_summary_overall_trust_is_minimum(store, api):
    """AC-11.3: TRUSTED + REJECTED for same entity → overall_trust == 'REJECTED' (worst wins)."""
    _make_trusted(store, EntityType.AGENT, 'mixed-agent', 'geval')
    _make_rejected(store, EntityType.AGENT, 'mixed-agent', 'correctness')

    summary = api.get_belief_summary(EntityType.AGENT, 'mixed-agent')

    assert summary['overall_trust'] == 'REJECTED'  # AC-11.3: minimum = worst level


# ---------------------------------------------------------------------------
# AC-11.4: list_entities_by_trust
# ---------------------------------------------------------------------------

def test_list_entities_by_trust_filters_correctly(store, api):
    """AC-11.4: filter by TRUSTED returns only TRUSTED entities."""
    _make_trusted(store, EntityType.SKILL, 'skill-a', 'geval')
    _make_provisional(store, EntityType.SKILL, 'skill-b', 'geval')
    _make_suspended(store, EntityType.SKILL, 'skill-c', 'geval')

    results = api.list_entities_by_trust(TrustLevel.TRUSTED)  # AC-11.4

    entity_ids = [r['entity_id'] for r in results]
    assert 'skill-a' in entity_ids  # AC-11.4: TRUSTED entity present
    assert 'skill-b' not in entity_ids  # AC-11.4: PROVISIONAL excluded
    assert 'skill-c' not in entity_ids  # AC-11.4: SUSPENDED excluded
    # Verify required fields in each result
    for r in results:
        assert 'entity_type' in r  # AC-11.4
        assert 'entity_id' in r  # AC-11.4
        assert 'score_name' in r  # AC-11.4
        assert 'mean' in r  # AC-11.4
        assert 'ci_width' in r  # AC-11.4
        assert 'observations' in r  # AC-11.4


# ---------------------------------------------------------------------------
# AC-11.5: check_deployment_gate
# ---------------------------------------------------------------------------

def test_check_deployment_gate_allows_when_trusted(store, api):
    """AC-11.5: TRUSTED belief + required TRUSTED → allowed=True, deficit=None."""
    _make_trusted(store, EntityType.AGENT, 'deploy-ready', 'overall')

    gate = api.check_deployment_gate(EntityType.AGENT, 'deploy-ready')  # AC-11.5

    assert gate['allowed'] is True  # AC-11.5
    assert gate['deficit'] is None  # AC-11.5: no deficit when allowed
    assert gate['current_level'] == 'TRUSTED'  # AC-11.5
    assert gate['required_level'] == 'TRUSTED'  # AC-11.5


def test_check_deployment_gate_denies_with_deficit_explanation(store, api):
    """AC-11.5: PROVISIONAL belief requesting TRUSTED gate → allowed=False, deficit non-empty."""
    _make_provisional(store, EntityType.AGENT, 'not-ready', 'overall')

    gate = api.check_deployment_gate(
        EntityType.AGENT, 'not-ready',
        required_level=TrustLevel.TRUSTED,
    )  # AC-11.5

    assert gate['allowed'] is False  # AC-11.5
    assert isinstance(gate['deficit'], str)  # AC-11.5: deficit explanation present
    assert len(gate['deficit']) > 0  # AC-11.5: non-empty explanation
    assert gate['current_level'] == 'PROVISIONAL'  # AC-11.5


def test_check_deployment_gate_unknown_entity_denied(store, api):
    """AC-11.5: no belief → defaults to PROVISIONAL → denied for TRUSTED gate."""
    gate = api.check_deployment_gate(
        EntityType.AGENT, 'unknown-entity',
        required_level=TrustLevel.TRUSTED,
    )  # AC-11.5

    assert gate['allowed'] is False  # AC-11.5: PROVISIONAL < TRUSTED
    assert gate['current_level'] == 'PROVISIONAL'  # AC-11.5: safe default


# ---------------------------------------------------------------------------
# AC-11.6: get_drift_alerts
# ---------------------------------------------------------------------------

def test_get_drift_alerts_returns_list(store, api):
    """AC-11.6: after recording drift event, get_drift_alerts returns non-empty list."""
    store.record_drift_event(
        belief_key='skill:test-skill:geval',
        mean_before=0.8,
        mean_after=0.5,
    )  # AC-11.6: degrading drift (delta=-0.3)

    alerts = api.get_drift_alerts()  # AC-11.6: default last 24 hours

    assert isinstance(alerts, list)  # AC-11.6
    assert len(alerts) > 0  # AC-11.6: recorded event present
    # Verify drift event fields
    alert = alerts[0]
    assert 'belief_key' in alert  # AC-11.6
    assert 'delta' in alert  # AC-11.6
    assert alert['delta'] < 0  # AC-11.6: degrading direction

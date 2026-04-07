"""
Eval Dashboard Data — read-only aggregation layer for evaluation results.

AC coverage: AC-12.1, AC-12.2, AC-12.3, AC-12.4, AC-12.5, AC-12.6, AC-12.7, AC-12.8
"""
import os
import sys
from collections import defaultdict
from typing import Optional

# sys.path: 3 levels up from api/evaluation/ to reach plugin root  # AC-12.1
_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PLUGIN_ROOT not in sys.path:
    sys.path.append(_PLUGIN_ROOT)

from api.belief_store.store import BeliefStore
from api.belief_store.models import EntityType, TrustLevel, ScoreSource


# ---------------------------------------------------------------------------
# Trust level ordering — worst (0) to best (3)  # AC-12.3
# ---------------------------------------------------------------------------

_TRUST_ORDER: dict[str, int] = {  # AC-12.3: minimum trust = worst level
    TrustLevel.REJECTED.value: 0,
    TrustLevel.SUSPENDED.value: 1,
    TrustLevel.PROVISIONAL.value: 2,
    TrustLevel.TRUSTED.value: 3,
}


# ---------------------------------------------------------------------------
# AC-12.1: EvalDashboardData — importable from api.evaluation.dashboard
# ---------------------------------------------------------------------------

class EvalDashboardData:  # AC-12.1
    """Read-only aggregation layer for evaluation results and belief states.

    All methods return plain dicts/lists — JSON-serializable.
    No writes, no side effects.

    Satisfies: AC-12.1, AC-12.2, AC-12.3, AC-12.4, AC-12.5,
               AC-12.6, AC-12.7, AC-12.8
    """

    def __init__(self, belief_store: BeliefStore, api) -> None:  # AC-12.2
        """Store dependencies. No network calls on init.

        Args:
            belief_store: BeliefStore instance for SQLite reads.
            api: LangfuseObservabilityAPI instance (reserved for future use).

        Satisfies: AC-12.2
        """
        self._store = belief_store  # AC-12.2: store belief_store
        self._api = api             # AC-12.2: store api reference

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _compute_trend(self, belief_key: str) -> str:  # AC-12.3
        """Determine belief trend from recent score history.

        Returns 'improving', 'degrading', or 'stable'.
        Satisfies: AC-12.3
        """
        history = self._store.get_score_history(belief_key, limit=10)
        if len(history) < 2:
            return 'stable'  # AC-12.3: insufficient data → stable
        first_mean = history[-1]['mean_after']   # oldest in limit window
        last_mean  = history[0]['mean_after']    # most recent
        delta = last_mean - first_mean
        if delta > 0.05:    # AC-12.3: improving threshold
            return 'improving'
        elif delta < -0.05: # AC-12.3: degrading threshold
            return 'degrading'
        return 'stable'     # AC-12.3: within ±0.05 → stable

    def _worst_trust(self, trust_values: list[str]) -> str:  # AC-12.3
        """Return the worst (minimum order) trust level from a list.

        Satisfies: AC-12.3
        """
        if not trust_values:
            return TrustLevel.PROVISIONAL.value
        return min(trust_values, key=lambda t: _TRUST_ORDER.get(t, 2))  # AC-12.3

    # -----------------------------------------------------------------------
    # AC-12.3: get_entity_overview
    # -----------------------------------------------------------------------

    def get_entity_overview(
        self,
        entity_type: Optional[EntityType] = None,
    ) -> list[dict]:  # AC-12.3
        """Return aggregated per-entity overview with all metrics.

        Args:
            entity_type: filter to this EntityType if provided.

        Returns:
            List of dicts with keys: entity_type, entity_id, metrics,
            overall_trust, trend.
            Each metric value is {trust, mean, ci, observations}.

        Satisfies: AC-12.3
        """
        # AC-12.3: list beliefs, optionally filtered by entity_type
        beliefs = self._store.list_beliefs(
            entity_type=entity_type,  # AC-12.3: filter by entity_type
            limit=1000,
        )

        # Group beliefs by (entity_type, entity_id)
        grouped: dict[tuple, list] = defaultdict(list)
        for b in beliefs:
            key = (b.entity_type.value, b.entity_id)
            grouped[key].append(b)  # AC-12.3: group by entity

        result = []
        for (etype, eid), ent_beliefs in grouped.items():
            # Build metrics dict — one entry per score_name
            metrics: dict[str, dict] = {}  # AC-12.3: metrics dict
            trust_values: list[str] = []

            for b in ent_beliefs:
                metrics[b.score_name] = {  # AC-12.3: score_name → metric dict
                    'trust': b.trust_level.value,
                    'mean': b.posterior_mean,
                    'ci': [b.ci_lower, b.ci_upper],   # AC-12.3: ci as [lower, upper]
                    'observations': b.total_observations,
                }
                trust_values.append(b.trust_level.value)

            # overall_trust = minimum (worst) across all metrics  # AC-12.3
            overall_trust = self._worst_trust(trust_values)  # AC-12.3

            # trend based on most-observed metric's history
            most_obs_belief = max(ent_beliefs, key=lambda b: b.total_observations)
            trend = self._compute_trend(most_obs_belief.belief_key)  # AC-12.3

            result.append({  # AC-12.3: entity dict
                'entity_type': etype,
                'entity_id': eid,
                'metrics': metrics,
                'overall_trust': overall_trust,  # AC-12.3: worst trust
                'trend': trend,                  # AC-12.3: improving|degrading|stable
            })

        return result  # AC-12.3: list of entity dicts

    # -----------------------------------------------------------------------
    # AC-12.4: get_metric_timeline
    # -----------------------------------------------------------------------

    def get_metric_timeline(
        self,
        entity_type: EntityType,
        entity_id: str,
        score_name: str,
        days: int = 30,
    ) -> list[dict]:  # AC-12.4
        """Return chronological score timeline for an entity metric.

        Args:
            entity_type: EntityType enum value.
            entity_id: entity identifier string.
            score_name: metric name.
            days: lookback window in days (default 30).

        Returns:
            List of dicts ordered by date ascending: {date, mean, ci_lower, ci_upper}.
            Returns empty list if no history found.

        Satisfies: AC-12.4
        """
        belief_key = f"{entity_type.value}:{entity_id}:{score_name}"  # AC-12.4
        # AC-12.4: reads from score_history using mean_after values
        return self._store.get_score_history_by_date(
            belief_key, days=days
        )  # AC-12.4: empty list if no history

    # -----------------------------------------------------------------------
    # AC-12.5: get_channel_breakdown
    # -----------------------------------------------------------------------

    def get_channel_breakdown(
        self,
        entity_type: EntityType,
        entity_id: str,
        score_name: str,
    ) -> dict:  # AC-12.5
        """Return per-channel observation breakdown for an entity metric.

        Returns:
            Dict with keys: api, annotation, eval.
            Each value: {count, mean_contribution}.
            Returns zeroed breakdown if no belief found.

        Satisfies: AC-12.5
        """
        # AC-12.5: zeroed breakdown default
        zeroed = {
            'api':        {'count': 0, 'mean_contribution': 0.0},
            'annotation': {'count': 0, 'mean_contribution': 0.0},
            'eval':       {'count': 0, 'mean_contribution': 0.0},
        }  # AC-12.5: zeroed if no belief found

        belief = self._store.get_belief(entity_type, entity_id, score_name)
        if belief is None:
            return zeroed  # AC-12.5: return zeroed if no belief

        total_obs = belief.total_observations or 1  # avoid division by zero

        # mean_contribution = channel_obs / total_obs * posterior_mean
        # (proportional contribution to the posterior mean)
        api_contrib = (belief.api_observations / total_obs) * belief.posterior_mean
        ann_contrib = (belief.annotation_observations / total_obs) * belief.posterior_mean
        eval_contrib = (belief.eval_observations / total_obs) * belief.posterior_mean

        return {  # AC-12.5: breakdown dict
            'api': {  # AC-12.5: api channel
                'count': belief.api_observations,
                'mean_contribution': round(api_contrib, 4),
            },
            'annotation': {  # AC-12.5: annotation channel
                'count': belief.annotation_observations,
                'mean_contribution': round(ann_contrib, 4),
            },
            'eval': {  # AC-12.5: eval channel
                'count': belief.eval_observations,
                'mean_contribution': round(eval_contrib, 4),
            },
        }

    # -----------------------------------------------------------------------
    # AC-12.6: get_trust_summary
    # -----------------------------------------------------------------------

    def get_trust_summary(self) -> dict:  # AC-12.6
        """Return aggregate trust distribution across all beliefs.

        Returns:
            Dict with: total_entities (int), trust_distribution
            (TrustLevel.value → count), needs_attention (list of dicts
            for SUSPENDED + REJECTED with entity, trust, mean).

        Satisfies: AC-12.6
        """
        # AC-12.6: aggregate across all beliefs in store
        beliefs = self._store.list_beliefs(limit=10000)

        total = len(beliefs)                               # AC-12.6: total_entities
        distribution: dict[str, int] = defaultdict(int)   # AC-12.6: trust_distribution
        needs_attention: list[dict] = []                  # AC-12.6: SUSPENDED + REJECTED

        for b in beliefs:
            tval = b.trust_level.value
            distribution[tval] += 1  # AC-12.6: count per trust level

            if b.trust_level in (TrustLevel.SUSPENDED, TrustLevel.REJECTED):  # AC-12.6
                needs_attention.append({  # AC-12.6: needs_attention entry
                    'entity': f"{b.entity_type.value}:{b.entity_id}",
                    'trust': tval,
                    'mean': b.posterior_mean,
                })

        # Sort needs_attention by mean ascending (worst first)  # AC-12.6
        needs_attention.sort(key=lambda e: e['mean'])  # AC-12.6

        return {  # AC-12.6: trust summary dict
            'total_entities': total,                        # AC-12.6
            'trust_distribution': dict(distribution),      # AC-12.6
            'needs_attention': needs_attention,             # AC-12.6: SUSPENDED+REJECTED
        }

    # -----------------------------------------------------------------------
    # AC-12.7: get_recent_evaluations
    # -----------------------------------------------------------------------

    def get_recent_evaluations(
        self,
        limit: int = 20,
        source=None,
    ) -> list[dict]:  # AC-12.7
        """Return most recent evaluation history rows.

        Args:
            limit: max rows to return (default 20).
            source: optional ScoreSource enum or string to filter by source.

        Returns:
            List of dicts ordered by created_at DESC:
            {belief_key, score_value, source, mean_before, mean_after,
             trust_before, trust_after, timestamp}.

        Satisfies: AC-12.7, AC-12.8
        """
        # AC-12.7: resolve source filter — handle enum or string
        source_str: Optional[str] = None
        if source is not None:
            if isinstance(source, ScoreSource):  # AC-12.7: enum → .value
                source_str = source.value
            else:
                source_str = str(source)  # AC-12.7: string passthrough

        # AC-12.8: read-only — only calls get_recent_history (no writes)
        raw_rows = self._store.get_recent_history(
            limit=limit,
            source=source_str,
        )  # AC-12.7, AC-12.8: read-only access

        result = []
        for row in raw_rows:
            result.append({  # AC-12.7: shaped dict
                'belief_key':   row['belief_key'],
                'score_value':  row['score_value'],
                'source':       row['source'],
                'mean_before':  row['mean_before'],
                'mean_after':   row['mean_after'],
                'trust_before': row['trust_before'],
                'trust_after':  row['trust_after'],
                'timestamp':    row['recorded_at'],  # AC-12.7: timestamp field
            })

        return result  # AC-12.7: list ordered by created_at DESC

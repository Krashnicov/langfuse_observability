"""
Trust Level API — decision layer over BeliefStore.

Translates raw BeliefState (α, β, posterior mean, CI width) into actionable
TrustLevel decisions with deployment gate support.

AC coverage: AC-11.1, AC-11.2, AC-11.3, AC-11.4, AC-11.5, AC-11.6
"""
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Optional

# sys.path: 3 levels up from api/evaluation/ to reach plugin root
_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PLUGIN_ROOT not in sys.path:
    sys.path.append(_PLUGIN_ROOT)

from api.belief_store.store import BeliefStore
from api.belief_store.models import EntityType, TrustLevel


# ---------------------------------------------------------------------------
# Trust hierarchy — lower index = worse trust
# ---------------------------------------------------------------------------

_TRUST_ORDER = [
    TrustLevel.REJECTED,    # 0 — worst
    TrustLevel.SUSPENDED,   # 1
    TrustLevel.PROVISIONAL, # 2
    TrustLevel.TRUSTED,     # 3 — best
]


class TrustLevelAPI:  # AC-11.1
    """Decision layer over BeliefStore — translates posterior stats to trust decisions.

    Satisfies: AC-11.1, AC-11.2, AC-11.3, AC-11.4, AC-11.5, AC-11.6
    """

    def __init__(self, belief_store: BeliefStore) -> None:
        """Initialise with a BeliefStore instance (inject for testability)."""
        self._store = belief_store  # AC-11.1: pure delegation to BeliefStore

    # -----------------------------------------------------------------------
    # AC-11.2: get_trust_level
    # -----------------------------------------------------------------------

    def get_trust_level(  # AC-11.2
        self,
        entity_type: EntityType,
        entity_id: str,
        score_name: str = "overall",
    ) -> TrustLevel:
        """Return TrustLevel for (entity_type, entity_id, score_name).

        Returns TrustLevel.PROVISIONAL if no belief exists — safe cold-start default.

        Satisfies: AC-11.2
        """
        belief = self._store.get_belief(entity_type, entity_id, score_name)
        if belief is None:
            return TrustLevel.PROVISIONAL  # AC-11.2: safe default for unknown entities
        return belief.trust_level  # AC-11.2: return stored trust level

    # -----------------------------------------------------------------------
    # AC-11.3: get_belief_summary
    # -----------------------------------------------------------------------

    def get_belief_summary(  # AC-11.3
        self,
        entity_type: EntityType,
        entity_id: str,
    ) -> dict:
        """Return summary dict for all metrics of an entity.

        Returns:
            dict with keys: entity_type, entity_id, metrics (score_name → details),
            overall_trust (minimum/worst TrustLevel across all metrics),
            last_updated (ISO timestamp or None if no beliefs).

        Satisfies: AC-11.3
        """
        # AC-11.3: fetch all beliefs for this entity_type, then filter by entity_id
        all_beliefs = self._store.list_beliefs(
            entity_type=entity_type,
            limit=500,
        )
        entity_beliefs = [
            b for b in all_beliefs
            if b.entity_id == entity_id
        ]  # AC-11.3: filter by entity_id in Python (list_beliefs has no entity_id param)

        # AC-11.3: build metrics dict
        metrics: dict = {}
        for b in entity_beliefs:
            metrics[b.score_name] = {  # AC-11.3: score_name → belief details
                "trust_level": b.trust_level.value,
                "mean": b.posterior_mean,
                "ci_lower": b.ci_lower,
                "ci_upper": b.ci_upper,
                "ci_width": b.ci_width,
                "alpha": b.alpha,
                "beta": b.beta,
                "observations": b.total_observations,
                "updated_at": b.updated_at,
            }

        # AC-11.3: overall_trust = minimum (worst) TrustLevel across all metrics
        if entity_beliefs:
            worst = min(
                entity_beliefs,
                key=lambda b: _TRUST_ORDER.index(b.trust_level),
            )
            overall_trust = worst.trust_level.value  # AC-11.3: worst wins
            last_updated = max(b.updated_at for b in entity_beliefs)  # most recent
        else:
            overall_trust = TrustLevel.PROVISIONAL.value  # AC-11.3: no beliefs → PROVISIONAL
            last_updated = None  # AC-11.3: None if no beliefs

        return {  # AC-11.3: required keys
            "entity_type": entity_type.value,
            "entity_id": entity_id,
            "metrics": metrics,
            "overall_trust": overall_trust,
            "last_updated": last_updated,
        }  # AC-11.3

    # -----------------------------------------------------------------------
    # AC-11.4: list_entities_by_trust
    # -----------------------------------------------------------------------

    def list_entities_by_trust(  # AC-11.4
        self,
        trust_level: TrustLevel,
        entity_type: Optional[EntityType] = None,
    ) -> list[dict]:
        """Return list of entities matching the given trust_level.

        Each item: entity_type, entity_id, score_name, mean, ci_width,
        observations, last_updated.

        Satisfies: AC-11.4
        """
        # AC-11.4: delegate to BeliefStore.list_beliefs() with trust_level filter
        beliefs = self._store.list_beliefs(
            entity_type=entity_type,
            trust_level=trust_level,  # AC-11.4: filtered by trust level
            limit=500,
        )
        return [
            {  # AC-11.4: required fields
                "entity_type": b.entity_type.value,
                "entity_id": b.entity_id,
                "score_name": b.score_name,
                "mean": b.posterior_mean,
                "ci_width": b.ci_width,
                "observations": b.total_observations,
                "last_updated": b.updated_at,
            }
            for b in beliefs
        ]  # AC-11.4

    # -----------------------------------------------------------------------
    # AC-11.5: check_deployment_gate
    # -----------------------------------------------------------------------

    def check_deployment_gate(  # AC-11.5
        self,
        entity_type: EntityType,
        entity_id: str,
        required_level: TrustLevel = TrustLevel.TRUSTED,
        score_name: str = "overall",
    ) -> dict:
        """Check if entity meets the required trust level for deployment.

        Returns dict: allowed (bool), current_level (str), required_level (str),
        mean (float), ci_width (float), deficit (str or None).

        Satisfies: AC-11.5
        """
        belief = self._store.get_belief(entity_type, entity_id, score_name)
        # AC-11.5: unknown entity defaults to PROVISIONAL (safe default)
        current = belief.trust_level if belief else TrustLevel.PROVISIONAL

        # AC-11.5: allowed if current rank >= required rank in trust hierarchy
        allowed = _TRUST_ORDER.index(current) >= _TRUST_ORDER.index(required_level)

        # AC-11.5: deficit explanation when denied
        deficit: Optional[str] = None
        if not allowed:
            if belief is not None:
                if required_level == TrustLevel.TRUSTED:
                    deficit = (  # AC-11.5: specific TRUSTED deficit explanation
                        f"Need mean \u2265 0.8 AND ci_width \u2264 0.2. "
                        f"Current: mean={belief.posterior_mean:.3f}, "
                        f"ci_width={belief.ci_width:.3f}"
                    )
                else:
                    deficit = (  # AC-11.5: generic deficit explanation
                        f"Current trust level {current.value} "
                        f"below required {required_level.value}"
                    )
            else:
                deficit = (  # AC-11.5: unknown entity deficit
                    f"No belief record found — entity defaults to PROVISIONAL "
                    f"which is below required {required_level.value}"
                )

        return {  # AC-11.5: required response keys
            "allowed": allowed,
            "current_level": current.value,
            "required_level": required_level.value,
            "mean": belief.posterior_mean if belief else 0.5,
            "ci_width": belief.ci_width if belief else 0.9,
            "deficit": deficit,
        }  # AC-11.5

    # -----------------------------------------------------------------------
    # AC-11.6: get_drift_alerts
    # -----------------------------------------------------------------------

    def get_drift_alerts(  # AC-11.6
        self,
        since: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> list[dict]:
        """Return drift events from the belief store.

        Args:
            since: ISO timestamp string. Defaults to 24 hours ago if not provided.
            direction: 'improving' (delta > 0) or 'degrading' (delta < 0). Optional filter.

        Returns:
            list of drift event dicts.

        Satisfies: AC-11.6
        """
        # AC-11.6: default since = 24 hours ago if not provided
        if since is None:
            since = (
                datetime.now(timezone.utc) - timedelta(hours=24)
            ).isoformat()  # AC-11.6: default last 24 hours

        # AC-11.6: delegate to BeliefStore.get_drift_alerts()
        alerts = self._store.get_drift_alerts(since=since, direction=direction)
        return alerts  # AC-11.6

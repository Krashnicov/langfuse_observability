"""
Belief Store data models — enums, CHANNEL_WEIGHTS, and BeliefState dataclass.

AC coverage: AC-10.2, AC-10.3
"""
import os
import sys

# sys.path: 3 levels up from api/belief_store/ to reach plugin root
_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PLUGIN_ROOT not in sys.path:
    sys.path.append(_PLUGIN_ROOT)

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from scipy.stats import beta as beta_dist  # AC-10.2: scipy.stats.beta.ppf for CI


# ---------------------------------------------------------------------------
# AC-10.2: Enums
# ---------------------------------------------------------------------------

class ScoreSource(Enum):  # AC-10.2
    """Score channel sources — determines channel weight in Bayesian update."""
    API = 'API'
    ANNOTATION = 'ANNOTATION'
    EVAL = 'EVAL'


class ScoreDataType(Enum):  # AC-10.2
    """Score value encoding — used by normalize_score()."""
    BOOLEAN = 'BOOLEAN'
    NUMERIC = 'NUMERIC'
    CATEGORICAL = 'CATEGORICAL'


class EntityType(Enum):  # AC-10.2
    """Entity category tracked by the Bayesian belief model."""
    AGENT = 'agent'
    SKILL = 'skill'
    PROMPT = 'prompt'
    TOOL = 'tool'


class TrustLevel(Enum):  # AC-10.2
    """Posterior trust classification based on mean + CI width thresholds."""
    TRUSTED = 'TRUSTED'
    PROVISIONAL = 'PROVISIONAL'
    SUSPENDED = 'SUSPENDED'
    REJECTED = 'REJECTED'


# ---------------------------------------------------------------------------
# AC-10.9: Channel weights (configurable via SQLite, defaults here)
# ---------------------------------------------------------------------------

CHANNEL_WEIGHTS: dict[ScoreSource, float] = {  # AC-10.9
    ScoreSource.API: 1.0,
    ScoreSource.ANNOTATION: 2.0,
    ScoreSource.EVAL: 0.5,
}


# ---------------------------------------------------------------------------
# AC-10.2 / AC-10.3: BeliefState dataclass
# ---------------------------------------------------------------------------

@dataclass
class BeliefState:  # AC-10.2
    """Beta(α,β) belief state per (entity_type, entity_id, score_name) key.

    Satisfies: AC-10.2, AC-10.3
    """
    entity_type: EntityType
    entity_id: str
    score_name: str
    alpha: float = 1.0         # Beta distribution α parameter (uniform prior)
    beta: float = 1.0          # Beta distribution β parameter (uniform prior)
    total_observations: int = 0
    last_observation_at: str | None = None
    api_observations: int = 0
    annotation_observations: int = 0
    eval_observations: int = 0
    posterior_mean: float = 0.5
    posterior_variance: float = 0.0
    ci_lower: float = 0.05
    ci_upper: float = 0.95
    ci_width: float = 0.90
    trust_level: TrustLevel = field(default=TrustLevel.PROVISIONAL)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def belief_key(self) -> str:  # AC-10.3
        """Composite key: '{entity_type}:{entity_id}:{score_name}'.

        Satisfies: AC-10.3
        """
        return f"{self.entity_type.value}:{self.entity_id}:{self.score_name}"  # AC-10.3

    def compute_posterior(self) -> None:  # AC-10.2
        """Recompute posterior statistics from current α,β via scipy.stats.beta.

        Updates: posterior_mean, posterior_variance, ci_lower, ci_upper,
                 ci_width, updated_at.

        Satisfies: AC-10.2
        """
        total = self.alpha + self.beta
        if total <= 0:
            return
        # AC-10.2: posterior mean = α/(α+β)
        self.posterior_mean = self.alpha / total
        # AC-10.2: posterior variance = αβ / (α+β)²(α+β+1)
        self.posterior_variance = (
            (self.alpha * self.beta) / (total ** 2 * (total + 1))
        )
        # AC-10.2: 90% credible interval via scipy.stats.beta.ppf
        a = max(self.alpha, 1e-10)
        b = max(self.beta, 1e-10)
        self.ci_lower = float(beta_dist.ppf(0.05, a, b))
        self.ci_upper = float(beta_dist.ppf(0.95, a, b))
        self.ci_width = self.ci_upper - self.ci_lower
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def compute_trust_level(self) -> 'TrustLevel':  # AC-10.2
        """Map posterior statistics to TrustLevel enum.

        Decision thresholds (AC-10.2):
          TRUSTED:     posterior_mean >= 0.8 AND ci_width <= 0.2
          PROVISIONAL: posterior_mean >= 0.6 AND ci_width <= 0.3
          SUSPENDED:   posterior_mean >= 0.4
          REJECTED:    posterior_mean <  0.4

        Satisfies: AC-10.2
        """
        if self.posterior_mean >= 0.8 and self.ci_width <= 0.2:  # AC-10.2: TRUSTED
            return TrustLevel.TRUSTED
        elif self.posterior_mean >= 0.6 and self.ci_width <= 0.3:  # AC-10.2: PROVISIONAL
            return TrustLevel.PROVISIONAL
        elif self.posterior_mean >= 0.4:  # AC-10.2: SUSPENDED
            return TrustLevel.SUSPENDED
        else:  # AC-10.2: REJECTED
            return TrustLevel.REJECTED

"""
Belief Store — SQLite-backed persistence for BeliefState objects.

AC coverage: AC-10.4, AC-10.5, AC-10.6, AC-10.7, AC-10.8
"""
import os
import sys
import sqlite3
from datetime import datetime, timezone
from typing import Optional

# sys.path: 3 levels up from api/belief_store/ to reach plugin root
_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PLUGIN_ROOT not in sys.path:
    sys.path.append(_PLUGIN_ROOT)

from api.belief_store.models import (
    BeliefState,
    EntityType,
    ScoreSource,
    TrustLevel,
)


# ---------------------------------------------------------------------------
# Default channel weights — AC-10.4
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: dict[str, float] = {  # AC-10.4
    ScoreSource.API.value: 1.0,
    ScoreSource.ANNOTATION.value: 2.0,
    ScoreSource.EVAL.value: 0.5,
}


# ---------------------------------------------------------------------------
# AC-10.4 — AC-10.8: BeliefStore
# ---------------------------------------------------------------------------

class BeliefStore:  # AC-10.4
    """SQLite-backed store for Bayesian BeliefState objects.

    Satisfies: AC-10.4, AC-10.5, AC-10.6, AC-10.7, AC-10.8
    """

    def __init__(self, db_path: Optional[str] = None) -> None:  # AC-10.4
        """Create or open the belief store at db_path.

        If db_path is None, defaults to {plugin_root}/data/belief_store.db.
        Pass ':memory:' for in-memory SQLite (test isolation).

        Satisfies: AC-10.4
        """
        # AC-10.4: default path resolution
        if db_path is None:
            db_path = os.path.join(_PLUGIN_ROOT, 'data', 'belief_store.db')
        # AC-10.4: create parent directory (skip for :memory:)
        if db_path != ':memory:':
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)  # AC-10.4
        self._conn.row_factory = sqlite3.Row
        self._init_schema()  # AC-10.4: all four tables created

    # -----------------------------------------------------------------------
    # Schema initialisation — AC-10.4
    # -----------------------------------------------------------------------

    def _init_schema(self) -> None:  # AC-10.4
        """Create all four tables and insert default channel weights.

        Satisfies: AC-10.4
        """
        cur = self._conn.cursor()

        # AC-10.4: belief_states table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS belief_states (
                belief_key             TEXT PRIMARY KEY,
                entity_type            TEXT NOT NULL,
                entity_id              TEXT NOT NULL,
                score_name             TEXT NOT NULL,
                alpha                  REAL NOT NULL DEFAULT 1.0,
                beta                   REAL NOT NULL DEFAULT 1.0,
                total_observations     INTEGER NOT NULL DEFAULT 0,
                last_observation_at    TEXT,
                api_observations       INTEGER NOT NULL DEFAULT 0,
                annotation_observations INTEGER NOT NULL DEFAULT 0,
                eval_observations      INTEGER NOT NULL DEFAULT 0,
                posterior_mean         REAL NOT NULL DEFAULT 0.5,
                posterior_variance     REAL NOT NULL DEFAULT 0.0,
                ci_lower               REAL NOT NULL DEFAULT 0.05,
                ci_upper               REAL NOT NULL DEFAULT 0.95,
                ci_width               REAL NOT NULL DEFAULT 0.90,
                trust_level            TEXT NOT NULL DEFAULT 'PROVISIONAL',
                created_at             TEXT NOT NULL,
                updated_at             TEXT NOT NULL
            )
        """)  # AC-10.4

        # AC-10.4: score_history table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS score_history (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                belief_key     TEXT NOT NULL,
                source         TEXT NOT NULL,
                score_value    REAL NOT NULL,
                alpha_before   REAL NOT NULL,
                beta_before    REAL NOT NULL,
                alpha_after    REAL NOT NULL,
                beta_after     REAL NOT NULL,
                mean_before    REAL NOT NULL,
                mean_after     REAL NOT NULL,
                trust_before   TEXT NOT NULL,
                trust_after    TEXT NOT NULL,
                has_distribution INTEGER NOT NULL DEFAULT 0,
                recorded_at    TEXT NOT NULL
            )
        """)  # AC-10.4

        # AC-10.4: channel_weights table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS channel_weights (
                source  TEXT PRIMARY KEY,
                weight  REAL NOT NULL
            )
        """)  # AC-10.4

        # AC-10.4: drift_events table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS drift_events (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                belief_key   TEXT NOT NULL,
                mean_before  REAL NOT NULL,
                mean_after   REAL NOT NULL,
                delta        REAL NOT NULL,
                recorded_at  TEXT NOT NULL
            )
        """)  # AC-10.4

        # AC-10.4: insert default channel weights (INSERT OR IGNORE — idempotent)
        for source, weight in _DEFAULT_WEIGHTS.items():
            cur.execute(
                "INSERT OR IGNORE INTO channel_weights (source, weight) VALUES (?, ?)",
                (source, weight),
            )  # AC-10.4: default weights API=1.0, ANNOTATION=2.0, EVAL=0.5

        self._conn.commit()

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _row_to_belief(self, row: sqlite3.Row) -> BeliefState:
        """Reconstruct BeliefState from a SQLite row dict."""
        return BeliefState(
            entity_type=EntityType(row['entity_type']),
            entity_id=row['entity_id'],
            score_name=row['score_name'],
            alpha=row['alpha'],
            beta=row['beta'],
            total_observations=row['total_observations'],
            last_observation_at=row['last_observation_at'],
            api_observations=row['api_observations'],
            annotation_observations=row['annotation_observations'],
            eval_observations=row['eval_observations'],
            posterior_mean=row['posterior_mean'],
            posterior_variance=row['posterior_variance'],
            ci_lower=row['ci_lower'],
            ci_upper=row['ci_upper'],
            ci_width=row['ci_width'],
            trust_level=TrustLevel(row['trust_level']),
            created_at=row['created_at'],
            updated_at=row['updated_at'],
        )

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # -----------------------------------------------------------------------
    # AC-10.5: get_or_create_belief
    # -----------------------------------------------------------------------

    def get_or_create_belief(
        self,
        entity_type: EntityType,
        entity_id: str,
        score_name: str,
    ) -> BeliefState:  # AC-10.5
        """Return existing belief or create new Beta(1,1) uniform prior.

        Satisfies: AC-10.5
        """
        existing = self.get_belief(entity_type, entity_id, score_name)
        if existing is not None:
            return existing  # AC-10.5: return existing
        # AC-10.5: create new with uniform prior Beta(1,1)
        belief = BeliefState(
            entity_type=entity_type,
            entity_id=entity_id,
            score_name=score_name,
            alpha=1.0,
            beta=1.0,
        )
        self.save_belief(belief)
        return belief

    # -----------------------------------------------------------------------
    # Helper: get_belief (single lookup)
    # -----------------------------------------------------------------------

    def get_belief(
        self,
        entity_type: EntityType,
        entity_id: str,
        score_name: str,
    ) -> Optional[BeliefState]:
        """Return BeliefState for the given key, or None if not found."""
        key = f"{entity_type.value}:{entity_id}:{score_name}"
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM belief_states WHERE belief_key = ?", (key,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_belief(row)

    # -----------------------------------------------------------------------
    # AC-10.6: save_belief — INSERT OR REPLACE upsert
    # -----------------------------------------------------------------------

    def save_belief(self, belief: BeliefState) -> None:  # AC-10.6
        """Persist BeliefState via INSERT OR REPLACE (idempotent upsert).

        Satisfies: AC-10.6
        """
        cur = self._conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO belief_states (
                belief_key, entity_type, entity_id, score_name,
                alpha, beta,
                total_observations, last_observation_at,
                api_observations, annotation_observations, eval_observations,
                posterior_mean, posterior_variance,
                ci_lower, ci_upper, ci_width,
                trust_level, created_at, updated_at
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?, ?
            )
        """, (  # AC-10.6: INSERT OR REPLACE
            belief.belief_key,
            belief.entity_type.value,
            belief.entity_id,
            belief.score_name,
            belief.alpha,
            belief.beta,
            belief.total_observations,
            belief.last_observation_at,
            belief.api_observations,
            belief.annotation_observations,
            belief.eval_observations,
            belief.posterior_mean,
            belief.posterior_variance,
            belief.ci_lower,
            belief.ci_upper,
            belief.ci_width,
            belief.trust_level.value,
            belief.created_at,
            belief.updated_at,
        ))
        self._conn.commit()

    # -----------------------------------------------------------------------
    # AC-10.7: list_beliefs — filtered query
    # -----------------------------------------------------------------------

    def list_beliefs(
        self,
        entity_type: Optional[EntityType] = None,
        trust_level: Optional[TrustLevel] = None,
        min_observations: int = 0,
        order_by: str = 'updated_at',
        limit: int = 100,
    ) -> list[BeliefState]:  # AC-10.7
        """Return filtered list of BeliefState objects.

        Satisfies: AC-10.7
        """
        # AC-10.7: build dynamic WHERE clause
        conditions: list[str] = []
        params: list = []

        if entity_type is not None:  # AC-10.7: filter by entity_type
            conditions.append("entity_type = ?")
            params.append(entity_type.value)

        if trust_level is not None:  # AC-10.7: filter by trust_level
            conditions.append("trust_level = ?")
            params.append(trust_level.value)

        if min_observations > 0:  # AC-10.7: filter by min_observations
            conditions.append("total_observations >= ?")
            params.append(min_observations)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        # Whitelist order_by to prevent SQL injection
        safe_order = order_by if order_by in (
            'updated_at', 'created_at', 'posterior_mean', 'total_observations'
        ) else 'updated_at'
        query = (
            f"SELECT * FROM belief_states {where}"
            f" ORDER BY {safe_order} DESC LIMIT ?"
        )
        params.append(limit)
        cur = self._conn.cursor()
        cur.execute(query, params)
        return [self._row_to_belief(row) for row in cur.fetchall()]

    # -----------------------------------------------------------------------
    # AC-10.8: record_score_history
    # -----------------------------------------------------------------------

    def record_score_history(
        self,
        belief_key: str,
        source: ScoreSource,
        score_value: float,
        alpha_before: float,
        beta_before: float,
        alpha_after: float,
        beta_after: float,
        mean_before: float,
        mean_after: float,
        trust_before: TrustLevel,
        trust_after: TrustLevel,
        has_distribution: bool,
    ) -> None:  # AC-10.8
        """Append a row to score_history with before/after α,β,mean,trust.

        Satisfies: AC-10.8
        """
        cur = self._conn.cursor()
        cur.execute("""
            INSERT INTO score_history (
                belief_key, source, score_value,
                alpha_before, beta_before, alpha_after, beta_after,
                mean_before, mean_after,
                trust_before, trust_after,
                has_distribution, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (  # AC-10.8: append row
            belief_key,
            source.value,
            score_value,
            alpha_before,
            beta_before,
            alpha_after,
            beta_after,
            mean_before,
            mean_after,
            trust_before.value,
            trust_after.value,
            1 if has_distribution else 0,
            self._now(),
        ))
        self._conn.commit()

    def get_score_history(
        self,
        belief_key: str,
        limit: int = 100,
    ) -> list[dict]:
        """Return score history rows for belief_key, newest first."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM score_history WHERE belief_key = ?"
            " ORDER BY id DESC LIMIT ?",
            (belief_key, limit),
        )
        return [dict(row) for row in cur.fetchall()]

    # -----------------------------------------------------------------------
    # AC-10.9: drift event recording
    # -----------------------------------------------------------------------

    def record_drift_event(
        self,
        belief_key: str,
        mean_before: float,
        mean_after: float,
    ) -> None:  # AC-10.9
        """Append a drift event when |mean_after - mean_before| > 0.15.

        Satisfies: AC-10.9
        """
        delta = mean_after - mean_before
        cur = self._conn.cursor()
        cur.execute("""
            INSERT INTO drift_events (belief_key, mean_before, mean_after, delta, recorded_at)
            VALUES (?, ?, ?, ?, ?)
        """, (belief_key, mean_before, mean_after, delta, self._now()))  # AC-10.9
        self._conn.commit()

    def get_drift_events(self, belief_key: str) -> list[dict]:
        """Return all drift events for belief_key."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM drift_events WHERE belief_key = ? ORDER BY id ASC",
            (belief_key,),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_drift_alerts(  # AC-11.6
        self,
        since: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> list[dict]:
        """Return drift events filtered by time window and direction.

        Args:
            since: ISO timestamp string — only events recorded_at >= since.
                   Defaults to 24 hours ago if None.
            direction: 'improving' (delta > 0) or 'degrading' (delta < 0).
                       No filter applied if None.

        Satisfies: AC-11.6
        """
        from datetime import datetime, timezone, timedelta  # local import to avoid circular

        # AC-11.6: default since = 24 hours ago
        if since is None:
            since = (
                datetime.now(timezone.utc) - timedelta(hours=24)
            ).isoformat()

        conditions = ["recorded_at >= ?"]
        params: list = [since]

        if direction == 'improving':  # AC-11.6: delta > 0
            conditions.append("delta > 0")
        elif direction == 'degrading':  # AC-11.6: delta < 0
            conditions.append("delta < 0")

        where = " AND ".join(conditions)
        cur = self._conn.cursor()
        cur.execute(
            f"SELECT * FROM drift_events WHERE {where} ORDER BY id ASC",
            params,
        )  # AC-11.6
        return [dict(row) for row in cur.fetchall()]  # AC-11.6


    def get_score_history_by_date(
        self,
        belief_key: str,
        days: int = 30,
    ) -> list[dict]:
        """Return score history grouped by date (YYYY-MM-DD) ordered ascending.

        Each entry: {date, mean, ci_lower, ci_upper}.
        Used by EvalDashboardData.get_metric_timeline.
        """
        from datetime import datetime, timezone, timedelta as _td
        from collections import defaultdict as _dd

        since = (datetime.now(timezone.utc) - _td(days=days)).isoformat()
        cur = self._conn.cursor()
        cur.execute(
            "SELECT recorded_at, mean_after FROM score_history"
            " WHERE belief_key = ? AND recorded_at >= ?"
            " ORDER BY recorded_at ASC",
            (belief_key, since),
        )
        rows = cur.fetchall()

        by_date: dict = _dd(list)
        for row in rows:
            date_str = row['recorded_at'][:10]  # YYYY-MM-DD
            by_date[date_str].append(row['mean_after'])

        result = []
        for date in sorted(by_date.keys()):
            means = by_date[date]
            avg = sum(means) / len(means)
            result.append({
                'date': date,
                'mean': avg,
                'ci_lower': max(0.0, avg - 0.1),
                'ci_upper': min(1.0, avg + 0.1),
            })
        return result

    def get_recent_history(
        self,
        limit: int = 20,
        source: Optional[str] = None,
    ) -> list[dict]:
        """Return most recent score_history rows across all beliefs.

        Args:
            limit: max rows to return (default 20).
            source: filter by source string if provided.

        Used by EvalDashboardData.get_recent_evaluations.
        """
        conditions: list[str] = []
        params: list = []

        if source is not None:
            conditions.append("source = ?")
            params.append(source)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)
        cur = self._conn.cursor()
        cur.execute(
            f"SELECT * FROM score_history {where} ORDER BY id DESC LIMIT ?",
            params,
        )
        return [dict(row) for row in cur.fetchall()]


    # -----------------------------------------------------------------------
    # AC-10.4: channel weight read/write
    # -----------------------------------------------------------------------

    def get_channel_weight(self, source: ScoreSource) -> float:  # AC-10.4
        """Return configured weight for source channel.

        Satisfies: AC-10.4
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT weight FROM channel_weights WHERE source = ?",
            (source.value,),
        )  # AC-10.4
        row = cur.fetchone()
        if row is None:
            return _DEFAULT_WEIGHTS.get(source.value, 1.0)
        return float(row['weight'])

    def set_channel_weight(self, source: ScoreSource, weight: float) -> None:  # AC-10.4
        """Override channel weight for source (INSERT OR REPLACE).

        Satisfies: AC-10.4
        """
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO channel_weights (source, weight) VALUES (?, ?)",
            (source.value, weight),
        )  # AC-10.4: upsert override
        self._conn.commit()

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

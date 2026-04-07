# AC-10.1: api/belief_store package — makes directory importable as Python package
# AC-10.6: exports BeliefStore for downstream consumers (STORY-011, STORY-012)
from api.belief_store.store import BeliefStore  # AC-10.1

__all__ = ['BeliefStore']

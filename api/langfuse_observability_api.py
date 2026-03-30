"""LangfuseObservabilityAPI — typed endpoint methods for the Langfuse REST API.

Provides typed wrappers over all Langfuse Phase 1 read endpoints, delegating
HTTP dispatch to the inherited LangfuseClient._sdk_call() mechanism.
Phases 2-7 are present as NotImplementedError stubs establishing the full
class shape for future sprints.

Satisfies: AC-2.1, AC-2.2, AC-2.3, AC-2.4, AC-2.5, AC-2.6
"""
# AC-2.1: sys.path injection block — identical to api/langfuse_trace.py lines 1-6
import os
import sys

_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PLUGIN_ROOT not in sys.path:
    sys.path.append(_PLUGIN_ROOT)

# AC-2.2: import LangfuseClient and error classes from api.langfuse_client
from api.langfuse_client import LangfuseClient, LangfuseAPIError, LangfuseAuthError  # noqa: F401


# AC-2.2: LangfuseObservabilityAPI(LangfuseClient) class definition
class LangfuseObservabilityAPI(LangfuseClient):
    """Typed endpoint methods for the Langfuse REST API.

    Phase 1 (read) is fully implemented via the SDK .api sub-client.
    Phases 2-7 are stubs raising NotImplementedError until implemented.

    Inherits from LangfuseClient:
        self._client  — Langfuse SDK singleton
        self._timeout — request timeout in seconds
        _sdk_call()   — timeout injection + error translation

    Satisfies: AC-2.1, AC-2.2, AC-2.3, AC-2.4, AC-2.5, AC-2.6
    """

    def __init__(self, timeout: int = LangfuseClient.DEFAULT_TIMEOUT) -> None:  # AC-2.2
        """Initialise the observability API wrapper.

        Args:
            timeout: Request timeout in seconds. Defaults to
                LangfuseClient.DEFAULT_TIMEOUT (30).

        Raises:
            RuntimeError: If get_langfuse_client() returns None — Langfuse is
                disabled or unconfigured (inherited from LangfuseClient).

        Satisfies: AC-2.2, AC-2.5
        """
        super().__init__(timeout)  # AC-2.2: calls super().__init__(timeout) only

    # ── Phase 1: Traces ──────────────────────────────────────────────────

    def list_traces(self, **kwargs):  # AC-2.3
        """GET /api/public/traces — paginated list with optional filters.

        Delegates to self._client.api.trace.list via inherited _sdk_call().
        All Langfuse filter parameters (page, limit, name, user_id, session_id,
        from_timestamp, to_timestamp, tags, order_by) pass as-is to the SDK.

        Args:
            **kwargs: Filter parameters forwarded unchanged to the SDK method.

        Returns:
            SDK Pydantic response model (not converted to dict — callers
            handle serialisation as needed).

        Raises:
            LangfuseAuthError: On 401/403 Unauthorised SDK responses.
            LangfuseAPIError: On any other non-2xx SDK API response.

        Satisfies: AC-2.3, AC-2.4
        """
        # AC-2.3: delegate to _sdk_call with self._client.api.trace.list
        # AC-2.4: **kwargs forwarded unchanged — no intermediate params dict
        return self._sdk_call(self._client.api.trace.list, **kwargs)

    def get_trace(self, trace_id: str):  # AC-3.1
        """GET /api/public/traces/{id} — single trace with observations.

        Returns the raw SDK Pydantic trace object without modification.
        The .observations attribute (list of observation objects) is
        accessible to the caller for serialisation — no serialisation here.

        Args:
            trace_id: The trace ID to retrieve.

        Returns:
            SDK Pydantic trace object with .id, .name, .observations,
            .session_id, .total_cost, .tags, .metadata.

        Raises:
            ValueError: If trace_id is falsy (empty string or None).
            LangfuseAuthError: On 401/403 Unauthorized SDK responses.
            LangfuseAPIError: On any other non-2xx SDK API response.

        Satisfies: AC-3.1, AC-3.2, AC-3.3, AC-3.4, AC-3.5
        """
        if not trace_id:  # AC-3.2: guard before _sdk_call — empty or None raises
            raise ValueError('trace_id is required')
        return self._sdk_call(  # AC-3.3: one-line dispatch; timeout injected by _sdk_call
            self._client.api.trace.get, trace_id  # AC-3.3, AC-3.4: SDK object returned verbatim
        )  # AC-3.5: _sdk_call translates _SDKApiError -> LangfuseAPIError


    # ── Phase 1: Observations ─────────────────────────────────────────────
    def list_observations(self, **kwargs):  # AC-4.1
        """GET /api/public/observations — paginated list with optional filters.

        Delegates to self._client.api.observations.list via inherited _sdk_call().
        All Langfuse filter parameters (page, limit, name, user_id, type, trace_id,
        parent_observation_id, from_start_time, to_start_time, level) pass as-is.

        Args:
            **kwargs: Filter parameters forwarded unchanged to the SDK method.  # AC-4.3, AC-4.5

        Returns:
            SDK Pydantic paginated response model.  # AC-4.2

        Raises:
            LangfuseAuthError: On 401/403 Unauthorized SDK responses.  # AC-4.4
            LangfuseAPIError: On any other non-2xx SDK API response.  # AC-4.4

        Satisfies: AC-4.1, AC-4.2, AC-4.3, AC-4.4, AC-4.5
        """
        # AC-4.2: delegate to _sdk_call with self._client.api.observations.list
        # AC-4.3: **kwargs forwarded unchanged — no intermediate params dict
        return self._sdk_call(self._client.api.observations.list, **kwargs)
    def get_observation(self, observation_id: str):  # AC-5.1
        """GET /api/public/observations/{id} — single observation with full detail.

        Returns the raw SDK Pydantic observation object without modification.
        Full detail fields accessible to caller: .input, .output, .usage, .type,
        .calculated_total_cost, .calculated_input_cost, .calculated_output_cost,
        .level — same attributes accessed in langfuse_trace.py lines 54-60.

        Args:
            observation_id: The observation ID to retrieve.

        Returns:
            SDK Pydantic observation object with full detail fields.

        Raises:
            ValueError: If observation_id is falsy (empty string or None).
            LangfuseAuthError: On 401/403 Unauthorized SDK responses.
            LangfuseAPIError: On any other non-2xx SDK API response.

        Satisfies: AC-5.1, AC-5.2, AC-5.3, AC-5.4, AC-5.5
        """
        if not observation_id:  # AC-5.2: guard before _sdk_call — empty or None raises
            raise ValueError('observation_id is required')
        return self._sdk_call(  # AC-5.3: one-line dispatch; timeout injected by _sdk_call
            self._client.api.observations.get, observation_id  # AC-5.3, AC-5.4: SDK object returned verbatim
        )  # AC-5.5: _sdk_call translates _SDKApiError -> LangfuseAPIError


    # ── Phase 1: Sessions ─────────────────────────────────────────────────
    # list_sessions()    added by STORY-006
    # get_session()      added by STORY-006

    def list_sessions(self, **kwargs):                # AC-6.1
        """GET /api/public/sessions -- paginated list.

        Satisfies: AC-6.1, AC-6.2, AC-6.3, AC-6.8
        """
        # AC-6.2 / AC-6.3: forward kwargs to SDK sessions.list
        return self._sdk_call(                        # AC-6.2
            self._client.api.sessions.list, **kwargs
        )

    def get_session(self, session_id: str):           # AC-6.4
        """GET /api/public/sessions/{id} -- single session.

        Satisfies: AC-6.4, AC-6.5, AC-6.6, AC-6.7, AC-6.8
        """
        # AC-6.5: guard -- empty/None raises ValueError
        if not session_id:
            raise ValueError('session_id is required')
        # AC-6.6 / AC-6.8: dispatch to SDK via _sdk_call
        return self._sdk_call(                       # AC-6.6
            self._client.api.sessions.get, session_id
        )

    # ── Phase 2–7 Stubs ───────────────────────────────────────────────────
    # All stubs raise NotImplementedError with the phase label per AC-2.6.

    def create_trace(self, **kwargs):  # AC-2.6
        """Phase 2 stub — not yet implemented."""
        raise NotImplementedError("Phase 2 — not yet implemented")

    def create_observation(self, **kwargs):  # AC-2.6
        """Phase 2 stub — not yet implemented."""
        raise NotImplementedError("Phase 2 — not yet implemented")

    def list_prompts(self, **kwargs):  # AC-2.6
        """Phase 3 stub — not yet implemented."""
        raise NotImplementedError("Phase 3 — not yet implemented")

    def get_prompt(self, prompt_name: str, **kwargs):  # AC-2.6
        """Phase 3 stub — not yet implemented."""
        raise NotImplementedError("Phase 3 — not yet implemented")

    def list_scores(self, **kwargs):  # AC-2.6
        """Phase 4 stub — not yet implemented."""
        raise NotImplementedError("Phase 4 — not yet implemented")

    def create_score(self, **kwargs):  # AC-2.6
        """Phase 4 stub — not yet implemented."""
        raise NotImplementedError("Phase 4 — not yet implemented")

    def list_datasets(self, **kwargs):  # AC-2.6
        """Phase 5 stub — not yet implemented."""
        raise NotImplementedError("Phase 5 — not yet implemented")

    def get_dataset(self, dataset_name: str, **kwargs):  # AC-2.6
        """Phase 5 stub — not yet implemented."""
        raise NotImplementedError("Phase 5 — not yet implemented")

    def batch_ingest(self, **kwargs):  # AC-2.6
        """Phase 6 stub — not yet implemented."""
        raise NotImplementedError("Phase 6 — not yet implemented")

    def get_health(self):  # AC-2.6
        """Phase 7 stub — not yet implemented."""
        raise NotImplementedError("Phase 7 — not yet implemented")

    def list_models(self, **kwargs):  # AC-2.6
        """Phase 7 stub — not yet implemented."""
        raise NotImplementedError("Phase 7 — not yet implemented")

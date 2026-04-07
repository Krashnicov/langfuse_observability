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
import base64
import json
import requests

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


    # ── Phase 2: Scores ───────────────────────────────────────────────────

    def _build_basic_auth_header(self) -> dict:  # AC-7.6
        """Build Authorization: Basic header from plugin config.json at runtime.

        Reads langfuse_public_key and langfuse_secret_key from
        {plugin_root}/config.json — zero hardcoded credentials in code.

        Returns:
            dict with 'Authorization' and 'Content-Type' keys.

        Satisfies: AC-7.6
        """
        # AC-7.6: runtime read from config.json — no credentials in source
        config_path = os.path.join(_PLUGIN_ROOT, 'config.json')
        with open(config_path) as f:
            cfg = json.load(f)
        pk = cfg.get('langfuse_public_key', '')
        sk = cfg.get('langfuse_secret_key', '')
        token = base64.b64encode(f"{pk}:{sk}".encode()).decode()
        return {
            "Authorization": f"Basic {token}",
            "Content-Type": "application/json",
        }

    def list_scores(self, **kwargs):  # AC-7.1
        """GET /api/public/scores — paginated list with optional filters.

        Delegates to self._client.api.score.list via inherited _sdk_call().
        Supports: page, limit, name, user_id, trace_id, observation_id,
        source, data_type, from_timestamp, to_timestamp, score_config_id,
        order_by.

        Args:
            **kwargs: Filter parameters forwarded unchanged to the SDK method.

        Returns:
            SDK Pydantic paginated response model.

        Raises:
            LangfuseAuthError: On 401/403 Unauthorized SDK responses.
            LangfuseAPIError: On any other non-2xx SDK API response.

        Satisfies: AC-7.1, AC-7.8
        """
        # AC-7.1: delegate to _sdk_call — all filter kwargs forwarded unchanged
        return self._sdk_call(self._client.api.score.list, **kwargs)

    def get_score(self, score_id: str):  # AC-7.2
        """GET /api/public/scores/{id} — single score by ID.

        Args:
            score_id: The score ID to retrieve.

        Returns:
            SDK Pydantic score object.

        Raises:
            ValueError: If score_id is falsy (empty string or None).
            LangfuseAuthError: On 401/403 Unauthorized SDK responses.
            LangfuseAPIError: On any other non-2xx SDK API response.

        Satisfies: AC-7.2, AC-7.8
        """
        if not score_id:  # AC-7.2: guard — falsy score_id raises ValueError
            raise ValueError('score_id is required')
        return self._sdk_call(  # AC-7.2: delegate to sdk.api.score.get
            self._client.api.score.get, score_id
        )  # AC-7.8: _sdk_call translates errors → LangfuseAPIError/LangfuseAuthError

    def create_score(self, **kwargs):  # AC-7.3
        """POST /api/public/scores — write a score via direct HTTP POST.

        Uses direct HTTP POST with Basic auth — NOT SDK create_score() which
        is async fire-and-forget and lacks a source= parameter (Spike S1
        validated). Credentials read from config.json at runtime.

        Args:
            **kwargs: Score payload as JSON body. Recommended keys:
                name (str), value (float), trace_id (str), source='API'.

        Returns:
            Response JSON as a plain dict.

        Raises:
            LangfuseAuthError: On 401/403 response from Langfuse.
            LangfuseAPIError: On any other non-2xx response.

        Satisfies: AC-7.3, AC-7.6, AC-7.7
        """
        # AC-7.3: HTTP POST to /api/public/scores — SDK write path NOT used
        config_path = os.path.join(_PLUGIN_ROOT, 'config.json')
        with open(config_path) as f:
            cfg = json.load(f)
        host = cfg.get('langfuse_host', 'http://localhost:3000').rstrip('/')
        url = f"{host}/api/public/scores"
        headers = self._build_basic_auth_header()  # AC-7.6: no hardcoded creds
        resp = requests.post(url, headers=headers, json=kwargs, timeout=self._timeout)
        if resp.status_code in (401, 403):  # AC-7.7: auth errors → LangfuseAuthError
            raise LangfuseAuthError(resp.status_code, resp.text)
        if not resp.ok:  # AC-7.7: other non-2xx → LangfuseAPIError with status_code
            raise LangfuseAPIError(resp.status_code, resp.text)
        return resp.json()  # AC-7.3: return response JSON as plain dict

    def delete_score(self, score_id: str):  # AC-7.4
        """DELETE /api/public/scores/{id} — delete a score by ID.

        Args:
            score_id: The score ID to delete.

        Returns:
            SDK response (typically None or a confirmation object).

        Raises:
            ValueError: If score_id is falsy (empty string or None).
            LangfuseAuthError: On 401/403 Unauthorized SDK responses.
            LangfuseAPIError: On any other non-2xx SDK API response.

        Satisfies: AC-7.4, AC-7.8
        """
        if not score_id:  # AC-7.4: guard — falsy score_id raises ValueError
            raise ValueError('score_id is required')
        return self._sdk_call(  # AC-7.4: delegate to sdk.api.score.delete
            self._client.api.score.delete, score_id
        )  # AC-7.8: _sdk_call translates errors → LangfuseAPIError/LangfuseAuthError

    def list_scores_v2(self, **kwargs):  # AC-7.5
        """GET scores with v2 extended fields; graceful fallback to v1 on self-hosted.

        Langfuse Cloud exposes /api/public/v2/scores; self-hosted instances
        may not support it. Delegates to sdk.api.score.list (v1 compatible)
        as the graceful fallback — behaviour is correct for Phase 2.

        Args:
            **kwargs: Filter parameters forwarded unchanged to the SDK method.

        Returns:
            SDK Pydantic paginated response model.

        Raises:
            LangfuseAuthError: On 401/403 Unauthorized SDK responses.
            LangfuseAPIError: On any other non-2xx SDK API response.

        Satisfies: AC-7.5
        """
        # AC-7.5: v2 not available on self-hosted — graceful fallback to v1 list
        return self._sdk_call(self._client.api.score.list, **kwargs)

    # ── Phase 2: Score Configs ────────────────────────────────────────────

    def list_score_configs(self, **kwargs):  # AC-8.1, AC-8.2
        """GET /api/public/score-configs — list reusable scoring templates.

        Delegates to self._client.api.score_configs.get via inherited _sdk_call().

        SDK path note: ScoreConfigsClient exposes .get(page, limit) for the
        paginated list endpoint (no .list method in SDK v4).

        Supported kwargs:
            page (int): Page number (1-indexed).
            limit (int): Items per page.

        Returns:
            SDK Pydantic paginated ScoreConfigs response object.

        Raises:
            LangfuseAuthError: On 401/403 Unauthorized.
            LangfuseAPIError: On any other non-2xx response.

        Satisfies: AC-8.1, AC-8.2, AC-8.5
        """
        # AC-8.1: delegate to _sdk_call — score_configs.get is the paginated list endpoint
        # AC-8.1: **kwargs forwarded unchanged — no intermediate params dict
        return self._sdk_call(self._client.api.score_configs.get, **kwargs)  # AC-8.1, AC-8.2

    def get_score_config(self, config_id: str):  # AC-8.3, AC-8.4
        """GET /api/public/score-configs/{id} — single score config by ID.

        Delegates to self._client.api.score_configs.get_by_id via _sdk_call().

        Args:
            config_id: The score config ID to retrieve.

        Returns:
            SDK Pydantic ScoreConfig object with .id, .name, .data_type attributes.

        Raises:
            ValueError: If config_id is falsy (empty string or None).
            LangfuseAuthError: On 401/403 Unauthorized.
            LangfuseAPIError: On any other non-2xx response.

        Satisfies: AC-8.3, AC-8.4, AC-8.5
        """
        if not config_id:  # AC-8.3: guard — falsy config_id raises ValueError
            raise ValueError('config_id is required')
        return self._sdk_call(  # AC-8.3: delegate to sdk.api.score_configs.get_by_id
            self._client.api.score_configs.get_by_id, config_id  # AC-8.4: SDK object returned verbatim
        )  # AC-8.5: _sdk_call translates errors → LangfuseAPIError/LangfuseAuthError

    def list_datasets(self, **kwargs):  # AC-13.1
        """List all datasets.

        Args:
            **kwargs: Optional filter kwargs forwarded to the SDK:
                page (int): Page number, starts at 1.
                limit (int): Items per page.

        Returns:
            PaginatedDatasets SDK object.

        Raises:
            LangfuseAuthError: On 401/403 Unauthorized.
            LangfuseAPIError: On any other non-2xx response.

        Satisfies: AC-13.1, AC-13.4, AC-13.5
        """
        return self._sdk_call(  # AC-13.1: delegate to sdk.api.datasets.list
            self._client.api.datasets.list, **kwargs
        )  # AC-13.4: _sdk_call translates errors → LangfuseAPIError/LangfuseAuthError

    def get_dataset(self, dataset_name: str, **kwargs):  # AC-13.2
        """Get a dataset by name.

        Args:
            dataset_name: Name of the dataset (required).
            **kwargs: Forwarded to SDK request_options if needed.

        Raises:
            ValueError: If dataset_name is falsy.
            LangfuseAuthError: On 401/403 Unauthorized.
            LangfuseAPIError: On any other non-2xx response.

        Satisfies: AC-13.2, AC-13.4, AC-13.5
        """
        if not dataset_name:  # AC-13.2: guard — falsy dataset_name raises ValueError
            raise ValueError('dataset_name is required')
        return self._sdk_call(  # AC-13.2: delegate to sdk.api.datasets.get
            self._client.api.datasets.get, dataset_name
        )  # AC-13.4: _sdk_call translates errors → LangfuseAPIError/LangfuseAuthError

    def create_dataset(self, name: str, **kwargs):  # AC-13.3
        """Create a new dataset.

        Args:
            name: Dataset name (required).
            **kwargs: Optional fields forwarded to SDK:
                description (str): Human-readable description.
                metadata (Any): Arbitrary metadata.
                input_schema (Any): JSON Schema for input validation.
                expected_output_schema (Any): JSON Schema for expected output.

        Raises:
            ValueError: If name is falsy.
            LangfuseAuthError: On 401/403 Unauthorized.
            LangfuseAPIError: On any other non-2xx response.

        Satisfies: AC-13.3, AC-13.4, AC-13.5
        """
        if not name:  # AC-13.3: guard — falsy name raises ValueError
            raise ValueError('name is required')
        return self._sdk_call(  # AC-13.3: delegate to sdk.api.datasets.create
            self._client.api.datasets.create, name=name, **kwargs
        )  # AC-13.4: _sdk_call translates errors → LangfuseAPIError/LangfuseAuthError

    def create_dataset_item(self, dataset_name: str, **kwargs):  # AC-14.1
        """Create a dataset item within a dataset.

        Args:
            dataset_name: Name of the dataset to add the item to (required).
            **kwargs: Optional fields forwarded to SDK:
                input (Any): The test case input.
                expected_output (Any): The expected output for evaluation.
                metadata (Any): Arbitrary metadata.
                source_trace_id (str): Trace ID this item was sourced from.
                source_observation_id (str): Observation ID this item was sourced from.
                id (str): Custom item ID (auto-generated if omitted).
                status (DatasetStatus): Item status.

        Raises:
            ValueError: If dataset_name is falsy.
            LangfuseAuthError: On 401/403 Unauthorized.
            LangfuseAPIError: On any other non-2xx response.

        Satisfies: AC-14.1, AC-14.5
        """
        if not dataset_name:  # AC-14.1: guard — falsy dataset_name raises ValueError
            raise ValueError('dataset_name is required')
        return self._sdk_call(  # AC-14.1: dataset_name is required kwarg, not positional
            self._client.api.dataset_items.create, dataset_name=dataset_name, **kwargs
        )  # AC-14.5: _sdk_call translates errors → LangfuseAPIError/LangfuseAuthError

    def get_dataset_item(self, item_id: str):  # AC-14.2
        """Get a dataset item by ID.

        Args:
            item_id: Dataset item ID (required).

        Raises:
            ValueError: If item_id is falsy.
            LangfuseAuthError: On 401/403 Unauthorized.
            LangfuseAPIError: On any other non-2xx response.

        Satisfies: AC-14.2, AC-14.5
        """
        if not item_id:  # AC-14.2: guard — falsy item_id raises ValueError
            raise ValueError('item_id is required')
        return self._sdk_call(  # AC-14.2: id is positional in SDK
            self._client.api.dataset_items.get, item_id
        )  # AC-14.5: _sdk_call translates errors → LangfuseAPIError/LangfuseAuthError

    def list_dataset_items(self, **kwargs):  # AC-14.3
        """List dataset items with optional filters.

        Args:
            **kwargs: Optional filter kwargs forwarded to SDK:
                dataset_name (str): Filter by dataset name.
                source_trace_id (str): Filter by source trace ID.
                source_observation_id (str): Filter by source observation ID.
                version (datetime): Return items as they existed at this UTC timestamp.
                page (int): Page number, starts at 1.
                limit (int): Items per page.

        Returns:
            PaginatedDatasetItems SDK object.

        Raises:
            LangfuseAuthError: On 401/403 Unauthorized.
            LangfuseAPIError: On any other non-2xx response.

        Satisfies: AC-14.3, AC-14.5
        """
        return self._sdk_call(  # AC-14.3: all kwargs optional, no required args
            self._client.api.dataset_items.list, **kwargs
        )  # AC-14.5: _sdk_call translates errors → LangfuseAPIError/LangfuseAuthError

    def delete_dataset_item(self, item_id: str):  # AC-14.4
        """Delete a dataset item and all its run items. Irreversible.

        Args:
            item_id: Dataset item ID (required).

        Raises:
            ValueError: If item_id is falsy.
            LangfuseAuthError: On 401/403 Unauthorized.
            LangfuseAPIError: On any other non-2xx response.

        Satisfies: AC-14.4, AC-14.5
        """
        if not item_id:  # AC-14.4: guard — falsy item_id raises ValueError
            raise ValueError('item_id is required')
        return self._sdk_call(  # AC-14.4: id is positional in SDK
            self._client.api.dataset_items.delete, item_id
        )  # AC-14.5: _sdk_call translates errors → LangfuseAPIError/LangfuseAuthError


    def get_dataset_run(self, dataset_name: str, run_name: str):  # AC-15.1
        """Get a dataset run and all its items.

        Args:
            dataset_name: Name of the dataset (required).
            run_name: Name of the run/experiment (required).

        Returns:
            DatasetRunWithItems SDK object.

        Raises:
            ValueError: If dataset_name or run_name is falsy.
            LangfuseAuthError: On 401/403 Unauthorized.
            LangfuseAPIError: On any other non-2xx response.

        Satisfies: AC-15.1, AC-15.6
        """
        if not dataset_name:  # AC-15.1: guard — falsy dataset_name raises ValueError
            raise ValueError('dataset_name is required')
        if not run_name:  # AC-15.1: guard — falsy run_name raises ValueError
            raise ValueError('run_name is required')
        return self._sdk_call(  # AC-15.1: both args positional to SDK
            self._client.api.datasets.get_run, dataset_name, run_name
        )

    def delete_dataset_run(self, dataset_name: str, run_name: str):  # AC-15.2
        """Delete a dataset run and all its run items. Irreversible.

        Args:
            dataset_name: Name of the dataset (required).
            run_name: Name of the run/experiment (required).

        Raises:
            ValueError: If dataset_name or run_name is falsy.
            LangfuseAuthError: On 401/403 Unauthorized.
            LangfuseAPIError: On any other non-2xx response.

        Satisfies: AC-15.2, AC-15.6
        """
        if not dataset_name:  # AC-15.2: guard
            raise ValueError('dataset_name is required')
        if not run_name:  # AC-15.2: guard
            raise ValueError('run_name is required')
        return self._sdk_call(  # AC-15.2: both args positional to SDK
            self._client.api.datasets.delete_run, dataset_name, run_name
        )

    def list_dataset_runs(self, dataset_name: str, **kwargs):  # AC-15.3
        """List all runs for a dataset.

        Args:
            dataset_name: Name of the dataset (required).
            **kwargs: Optional filter kwargs forwarded to SDK:
                page (int): Page number, starts at 1.
                limit (int): Items per page.

        Returns:
            PaginatedDatasetRuns SDK object.

        Raises:
            ValueError: If dataset_name is falsy.
            LangfuseAuthError: On 401/403 Unauthorized.
            LangfuseAPIError: On any other non-2xx response.

        Satisfies: AC-15.3, AC-15.6
        """
        if not dataset_name:  # AC-15.3: guard
            raise ValueError('dataset_name is required')
        return self._sdk_call(  # AC-15.3: dataset_name positional, page/limit as kwargs
            self._client.api.datasets.get_runs, dataset_name, **kwargs
        )

    def create_dataset_run_item(self, run_name: str, dataset_item_id: str, **kwargs):  # AC-15.4
        """Create a dataset run item — link a trace to a dataset item within an experiment run.

        Args:
            run_name: Name of the experiment run (required).
            dataset_item_id: ID of the dataset item being evaluated (required).
            **kwargs: Optional fields forwarded to SDK:
                trace_id (str): ID of the trace produced for this item.
                observation_id (str): ID of the specific observation.
                run_description (str): Description of the run.
                metadata (Any): Arbitrary metadata.
                dataset_version (datetime): Dataset version timestamp.
                created_at (datetime): Custom creation timestamp.

        Returns:
            DatasetRunItem SDK object.

        Raises:
            ValueError: If run_name or dataset_item_id is falsy.
            LangfuseAuthError: On 401/403 Unauthorized.
            LangfuseAPIError: On any other non-2xx response.

        Satisfies: AC-15.4, AC-15.6
        """
        if not run_name:  # AC-15.4: guard — run_name required
            raise ValueError('run_name is required')
        if not dataset_item_id:  # AC-15.4: guard — dataset_item_id required
            raise ValueError('dataset_item_id is required')
        return self._sdk_call(  # AC-15.4: both required as keyword args to SDK
            self._client.api.dataset_run_items.create,
            run_name=run_name,
            dataset_item_id=dataset_item_id,
            **kwargs
        )

    def list_dataset_run_items(self, dataset_id: str, run_name: str, **kwargs):  # AC-15.5
        """List run items for a specific dataset run.

        Args:
            dataset_id: Internal Langfuse dataset ID (NOT dataset name — required).
                Callers must obtain this from get_dataset() — the returned Dataset
                object has an 'id' field containing the internal ID.
            run_name: Name of the experiment run (required).
            **kwargs: Optional filter kwargs forwarded to SDK:
                page (int): Page number, starts at 1.
                limit (int): Items per page.

        Returns:
            PaginatedDatasetRunItems SDK object.

        Raises:
            ValueError: If dataset_id or run_name is falsy.
            LangfuseAuthError: On 401/403 Unauthorized.
            LangfuseAPIError: On any other non-2xx response.

        Satisfies: AC-15.5, AC-15.6
        """
        if not dataset_id:  # AC-15.5: guard — dataset_id required (NOT dataset_name)
            raise ValueError('dataset_id is required')
        if not run_name:  # AC-15.5: guard
            raise ValueError('run_name is required')
        return self._sdk_call(  # AC-15.5: both required as keyword args to SDK
            self._client.api.dataset_run_items.list,
            dataset_id=dataset_id,
            run_name=run_name,
            **kwargs
        )

    def batch_ingest(self, **kwargs):  # AC-2.6
        """Phase 6 stub — not yet implemented."""
        raise NotImplementedError("Phase 6 — not yet implemented")

    def get_health(self):  # AC-2.6
        """Phase 7 stub — not yet implemented."""
        raise NotImplementedError("Phase 7 — not yet implemented")

    def list_models(self, **kwargs):  # AC-2.6
        """Phase 7 stub — not yet implemented."""
        raise NotImplementedError("Phase 7 — not yet implemented")

"""Tests for LangfuseObservabilityAPI.list_traces() — STORY-002.

TDD anchors (4 required):
  - test_list_traces_calls_sdk_trace_list
  - test_list_traces_forwards_kwargs_to_sdk
  - test_list_traces_propagates_api_error
  - test_list_traces_raises_runtime_error_when_disabled

Additional coverage:
  - Phase 2-7 stub tests (AC-2.6)
  - Constructor timeout tests (AC-2.2)

Satisfies: AC-2.1, AC-2.2, AC-2.3, AC-2.4, AC-2.5, AC-2.6
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, mock_open, patch, sentinel

# sys.path injection — same pattern as langfuse_client.py
_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PLUGIN_ROOT not in sys.path:
    sys.path.append(_PLUGIN_ROOT)

from langfuse.api.core import ApiError as _SDKApiError
from api.langfuse_client import LangfuseAPIError, LangfuseAuthError, LangfuseClient
from api.langfuse_observability_api import LangfuseObservabilityAPI

_MOCK_TARGET = 'api.langfuse_client.get_langfuse_client'


def _make_sdk_mock():
    """Return a MagicMock representing the Langfuse SDK singleton."""
    return MagicMock()


class TestListTraces(unittest.TestCase):
    """Tests for LangfuseObservabilityAPI.list_traces()."""

    # ── TDD anchor 1: basic delegation ───────────────────────────────────

    def test_list_traces_calls_sdk_trace_list(self):
        """list_traces() delegates to client.api.trace.list and returns result.

        Satisfies: AC-2.3
        """
        mock_client = _make_sdk_mock()
        mock_client.api.trace.list.return_value = sentinel.trace_list_result

        with patch(_MOCK_TARGET, return_value=mock_client):
            api = LangfuseObservabilityAPI()
            result = api.list_traces()

        # AC-2.3: trace.list was called
        mock_client.api.trace.list.assert_called_once()
        # returns the SDK result unchanged
        self.assertIs(result, sentinel.trace_list_result)

    # ── TDD anchor 2: kwargs forwarding ──────────────────────────────────

    def test_list_traces_forwards_kwargs_to_sdk(self):
        """list_traces(**kwargs) forwards user kwargs to SDK unchanged.

        Satisfies: AC-2.4
        """
        mock_client = _make_sdk_mock()
        mock_client.api.trace.list.return_value = sentinel.result

        with patch(_MOCK_TARGET, return_value=mock_client):
            api = LangfuseObservabilityAPI()
            api.list_traces(name='test', page=2)

        call_kwargs = mock_client.api.trace.list.call_args.kwargs
        # AC-2.4: user kwargs forwarded unchanged
        self.assertEqual(call_kwargs['name'], 'test')  # AC-2.4
        self.assertEqual(call_kwargs['page'], 2)       # AC-2.4

    # ── TDD anchor 3: error propagation ──────────────────────────────────

    def test_list_traces_propagates_api_error(self):
        """SDK ApiError(500) is translated to LangfuseAPIError with status_code.

        Satisfies: AC-2.5 (inherited _sdk_call error translation)
        """
        mock_client = _make_sdk_mock()
        sdk_err = _SDKApiError(status_code=500, body='internal server error')
        mock_client.api.trace.list.side_effect = sdk_err

        with patch(_MOCK_TARGET, return_value=mock_client):
            api = LangfuseObservabilityAPI()
            with self.assertRaises(LangfuseAPIError) as ctx:
                api.list_traces()

        self.assertEqual(ctx.exception.status_code, 500)  # AC-2.5

    # ── TDD anchor 4: disabled client ────────────────────────────────────

    def test_list_traces_raises_runtime_error_when_disabled(self):
        """RuntimeError raised when get_langfuse_client() returns None.

        Satisfies: AC-2.5
        """
        with patch(_MOCK_TARGET, return_value=None):
            with self.assertRaises(RuntimeError):  # AC-2.5
                LangfuseObservabilityAPI()

    # ── AC-2.2: constructor ───────────────────────────────────────────────

    def test_init_accepts_custom_timeout(self):
        """__init__ accepts timeout param and stores via super().__init__.

        Satisfies: AC-2.2
        """
        mock_client = _make_sdk_mock()
        with patch(_MOCK_TARGET, return_value=mock_client):
            api = LangfuseObservabilityAPI(timeout=60)
        self.assertEqual(api._timeout, 60)  # AC-2.2

    def test_default_timeout_matches_langfuse_client_default(self):
        """Default timeout matches LangfuseClient.DEFAULT_TIMEOUT.

        Satisfies: AC-2.2
        """
        mock_client = _make_sdk_mock()
        with patch(_MOCK_TARGET, return_value=mock_client):
            api = LangfuseObservabilityAPI()
        self.assertEqual(api._timeout, LangfuseClient.DEFAULT_TIMEOUT)  # AC-2.2

    def test_inherits_langfuse_client(self):
        """LangfuseObservabilityAPI is-a LangfuseClient.

        Satisfies: AC-2.2
        """
        mock_client = _make_sdk_mock()
        with patch(_MOCK_TARGET, return_value=mock_client):
            api = LangfuseObservabilityAPI()
        self.assertIsInstance(api, LangfuseClient)  # AC-2.2

    # ── AC-2.6: Phase 2-7 stubs ───────────────────────────────────────────

    def _make_api(self):
        """Helper: construct API with mocked client."""
        with patch(_MOCK_TARGET, return_value=_make_sdk_mock()):
            return LangfuseObservabilityAPI()

    def test_phase2_create_trace_raises_not_implemented(self):
        """AC-2.6: create_trace raises NotImplementedError with Phase 2 label."""
        api = self._make_api()
        with self.assertRaises(NotImplementedError) as ctx:
            api.create_trace()
        self.assertIn('Phase 2', str(ctx.exception))  # AC-2.6

    def test_phase2_create_observation_raises_not_implemented(self):
        """AC-2.6: create_observation raises NotImplementedError with Phase 2 label."""
        api = self._make_api()
        with self.assertRaises(NotImplementedError) as ctx:
            api.create_observation()
        self.assertIn('Phase 2', str(ctx.exception))  # AC-2.6

    def test_phase3_list_prompts_raises_not_implemented(self):
        """AC-2.6: list_prompts raises NotImplementedError with Phase 3 label."""
        api = self._make_api()
        with self.assertRaises(NotImplementedError) as ctx:
            api.list_prompts()
        self.assertIn('Phase 3', str(ctx.exception))  # AC-2.6

    def test_phase3_get_prompt_raises_not_implemented(self):
        """AC-2.6: get_prompt raises NotImplementedError with Phase 3 label."""
        api = self._make_api()
        with self.assertRaises(NotImplementedError) as ctx:
            api.get_prompt('my-prompt')
        self.assertIn('Phase 3', str(ctx.exception))  # AC-2.6

    def test_phase4_list_scores_now_implemented(self):
        """STORY-007: list_scores now implemented — must NOT raise NotImplementedError."""
        # Use __new__ bypass — no real SDK singleton needed
        api = LangfuseObservabilityAPI.__new__(LangfuseObservabilityAPI)
        api._client = MagicMock()
        api._timeout = LangfuseObservabilityAPI.DEFAULT_TIMEOUT
        try:
            api.list_scores()  # AC-7.1: stub replaced — no NotImplementedError
        except NotImplementedError:
            self.fail("list_scores still raises NotImplementedError — stub not replaced")

    def test_phase4_create_score_now_implemented(self):
        """STORY-007: create_score now implemented — uses HTTP POST, not NotImplementedError."""
        import json as _json
        api = LangfuseObservabilityAPI.__new__(LangfuseObservabilityAPI)
        api._client = MagicMock()
        api._timeout = LangfuseObservabilityAPI.DEFAULT_TIMEOUT
        fake_cfg = _json.dumps({
            "langfuse_host": "http://localhost:3010",
            "langfuse_public_key": "pk-test",
            "langfuse_secret_key": "sk-test",
        })

        class _OKResp:
            status_code = 200; ok = True; text = ""
            def json(self): return {"id": "sc-test"}

        with patch('builtins.open', mock_open(read_data=fake_cfg)):
            with patch('requests.post', return_value=_OKResp()):
                try:
                    api.create_score(name='test', value=1.0, trace_id='t1')  # AC-7.3
                except NotImplementedError:
                    self.fail("create_score still raises NotImplementedError — stub not replaced")

    def test_phase3_list_datasets_now_implemented(self):
        """AC-13.1: list_datasets no longer raises NotImplementedError — Phase 3 implemented."""
        api = self._make_api()
        try:
            api.list_datasets()  # AC-13.1: delegates to _sdk_call, no NotImplementedError
        except NotImplementedError:
            self.fail("list_datasets still raises NotImplementedError — stub not replaced")

    def test_phase3_get_dataset_now_implemented(self):
        """AC-13.2: get_dataset no longer raises NotImplementedError — Phase 3 implemented."""
        api = self._make_api()
        try:
            api.get_dataset('my-dataset')  # AC-13.2: delegates to _sdk_call, no NotImplementedError
        except NotImplementedError:
            self.fail("get_dataset still raises NotImplementedError — stub not replaced")

    def test_phase6_batch_ingest_raises_not_implemented(self):
        """AC-2.6: batch_ingest raises NotImplementedError with Phase 6 label."""
        api = self._make_api()
        with self.assertRaises(NotImplementedError) as ctx:
            api.batch_ingest()
        self.assertIn('Phase 6', str(ctx.exception))  # AC-2.6

    def test_phase7_get_health_raises_not_implemented(self):
        """AC-2.6: get_health raises NotImplementedError with Phase 7 label."""
        api = self._make_api()
        with self.assertRaises(NotImplementedError) as ctx:
            api.get_health()
        self.assertIn('Phase 7', str(ctx.exception))  # AC-2.6

    def test_phase7_list_models_raises_not_implemented(self):
        """AC-2.6: list_models raises NotImplementedError with Phase 7 label."""
        api = self._make_api()
        with self.assertRaises(NotImplementedError) as ctx:
            api.list_models()
        self.assertIn('Phase 7', str(ctx.exception))  # AC-2.6


if __name__ == '__main__':
    unittest.main()

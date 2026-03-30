"""Tests for LangfuseObservabilityAPI.get_trace() — STORY-003.

TDD anchors (4 required):
  - test_get_trace_calls_sdk_trace_get_with_id
  - test_get_trace_raises_value_error_on_empty_id
  - test_get_trace_raises_value_error_on_none_id
  - test_get_trace_propagates_404_as_api_error

Satisfies: AC-3.1, AC-3.2, AC-3.3, AC-3.4, AC-3.5
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, sentinel

# sys.path injection — same pattern as langfuse_client.py and test_list_traces.py
_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PLUGIN_ROOT not in sys.path:
    sys.path.append(_PLUGIN_ROOT)

from langfuse.api.core import ApiError as _SDKApiError
from api.langfuse_client import LangfuseAPIError, LangfuseAuthError
from api.langfuse_observability_api import LangfuseObservabilityAPI

_MOCK_TARGET = 'api.langfuse_client.get_langfuse_client'


def _make_sdk_mock():
    """Return a MagicMock representing the Langfuse SDK singleton."""
    return MagicMock()


class TestGetTrace(unittest.TestCase):
    """Tests for LangfuseObservabilityAPI.get_trace().

    Satisfies: AC-3.1, AC-3.2, AC-3.3, AC-3.4, AC-3.5
    """

    # ── TDD anchor 1: basic delegation ───────────────────────────────────

    def test_get_trace_calls_sdk_trace_get_with_id(self):
        """get_trace() delegates to client.api.trace.get and returns result unchanged.

        Satisfies: AC-3.1, AC-3.3, AC-3.4
        """
        mock_client = _make_sdk_mock()
        trace_mock = MagicMock()
        trace_mock.observations = []  # AC-3.4: .observations attribute accessible
        mock_client.api.trace.get.return_value = trace_mock

        with patch(_MOCK_TARGET, return_value=mock_client):
            api = LangfuseObservabilityAPI()
            result = api.get_trace('trace-123')

        # AC-3.3: trace.get was called with the supplied trace_id
        mock_client.api.trace.get.assert_called_once()
        call_args = mock_client.api.trace.get.call_args
        self.assertEqual(call_args.args[0], 'trace-123')  # AC-3.3
        # AC-3.3: returns SDK result verbatim
        self.assertIs(result, trace_mock)
        # AC-3.4: .observations attribute accessible without modification
        self.assertEqual(result.observations, [])  # AC-3.4

    # ── TDD anchor 2: empty string guard ─────────────────────────────────

    def test_get_trace_raises_value_error_on_empty_id(self):
        """get_trace('') raises ValueError with correct message before calling SDK.

        Satisfies: AC-3.2
        """
        mock_client = _make_sdk_mock()

        with patch(_MOCK_TARGET, return_value=mock_client):
            api = LangfuseObservabilityAPI()
            with self.assertRaises(ValueError) as ctx:  # AC-3.2
                api.get_trace('')

        # AC-3.2: message must be 'trace_id is required'
        self.assertIn('trace_id is required', str(ctx.exception))  # AC-3.2
        # Guard fires BEFORE _sdk_call — SDK must not be touched
        mock_client.api.trace.get.assert_not_called()

    # ── TDD anchor 3: None guard ──────────────────────────────────────────

    def test_get_trace_raises_value_error_on_none_id(self):
        """get_trace(None) raises ValueError before calling SDK.

        Satisfies: AC-3.2
        """
        mock_client = _make_sdk_mock()

        with patch(_MOCK_TARGET, return_value=mock_client):
            api = LangfuseObservabilityAPI()
            with self.assertRaises(ValueError):  # AC-3.2
                api.get_trace(None)

        mock_client.api.trace.get.assert_not_called()

    # ── TDD anchor 4: 404 propagation ────────────────────────────────────

    def test_get_trace_propagates_404_as_api_error(self):
        """SDK ApiError(404) is translated to LangfuseAPIError by _sdk_call.

        Satisfies: AC-3.5
        """
        mock_client = _make_sdk_mock()
        sdk_error = _SDKApiError(status_code=404, body='not found')
        mock_client.api.trace.get.side_effect = sdk_error

        with patch(_MOCK_TARGET, return_value=mock_client):
            api = LangfuseObservabilityAPI()
            with self.assertRaises(LangfuseAPIError) as ctx:  # AC-3.5
                api.get_trace('nonexistent-id')

        # AC-3.5: status_code preserved through translation
        self.assertEqual(ctx.exception.status_code, 404)  # AC-3.5


if __name__ == '__main__':
    unittest.main()

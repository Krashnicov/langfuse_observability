"""Tests for LangfuseObservabilityAPI.list_observations() — STORY-004.

TDD anchors (4 required):
  - test_list_observations_calls_sdk_observations_list
  - test_list_observations_forwards_type_filter
  - test_list_observations_forwards_all_kwargs
  - test_list_observations_propagates_api_error

Satisfies: AC-4.1, AC-4.2, AC-4.3, AC-4.4, AC-4.5
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, sentinel

# sys.path injection — same pattern as langfuse_client.py
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


class TestListObservations(unittest.TestCase):
    """Tests for LangfuseObservabilityAPI.list_observations()."""

    # ── TDD anchor 1: basic delegation ───────────────────────────────────

    def test_list_observations_calls_sdk_observations_list(self):
        """list_observations() delegates to client.api.observations.list and returns result.

        Satisfies: AC-4.2
        """
        mock_client = _make_sdk_mock()
        mock_client.api.observations.list.return_value = sentinel.obs_list_result

        with patch(_MOCK_TARGET, return_value=mock_client):
            api = LangfuseObservabilityAPI()
            result = api.list_observations()

        # AC-4.2: observations.list was called
        mock_client.api.observations.list.assert_called_once()
        # returns the SDK result unchanged
        self.assertIs(result, sentinel.obs_list_result)  # AC-4.2

    # ── TDD anchor 2: type filter forwarding ─────────────────────────────

    def test_list_observations_forwards_type_filter(self):
        """list_observations(type='GENERATION') forwards type kwarg to SDK unchanged.

        Satisfies: AC-4.3, AC-4.5
        """
        mock_client = _make_sdk_mock()
        mock_client.api.observations.list.return_value = sentinel.result

        with patch(_MOCK_TARGET, return_value=mock_client):
            api = LangfuseObservabilityAPI()
            api.list_observations(type='GENERATION', page=1)

        call_kwargs = mock_client.api.observations.list.call_args.kwargs
        self.assertEqual(call_kwargs['type'], 'GENERATION')  # AC-4.3
        self.assertEqual(call_kwargs['page'], 1)             # AC-4.3

    # ── TDD anchor 3: all kwargs forwarding ──────────────────────────────

    def test_list_observations_forwards_all_kwargs(self):
        """list_observations(**kwargs) forwards all provided kwargs to SDK unchanged.

        Satisfies: AC-4.3, AC-4.5
        """
        mock_client = _make_sdk_mock()
        mock_client.api.observations.list.return_value = sentinel.result

        with patch(_MOCK_TARGET, return_value=mock_client):
            api = LangfuseObservabilityAPI()
            api.list_observations(trace_id='t1', level='ERROR', page=3)

        call_kwargs = mock_client.api.observations.list.call_args.kwargs
        self.assertEqual(call_kwargs['trace_id'], 't1')  # AC-4.3, AC-4.5
        self.assertEqual(call_kwargs['level'], 'ERROR')  # AC-4.3, AC-4.5
        self.assertEqual(call_kwargs['page'], 3)         # AC-4.3

    # ── TDD anchor 4: error propagation ──────────────────────────────────

    def test_list_observations_propagates_api_error(self):
        """SDK ApiError(400) is translated to LangfuseAPIError with status_code.

        Satisfies: AC-4.4 (inherited _sdk_call error translation)
        """
        mock_client = _make_sdk_mock()
        sdk_err = _SDKApiError(status_code=400, body='bad request')
        mock_client.api.observations.list.side_effect = sdk_err

        with patch(_MOCK_TARGET, return_value=mock_client):
            api = LangfuseObservabilityAPI()
            with self.assertRaises(LangfuseAPIError) as ctx:
                api.list_observations()

        self.assertEqual(ctx.exception.status_code, 400)  # AC-4.4


if __name__ == '__main__':
    unittest.main()

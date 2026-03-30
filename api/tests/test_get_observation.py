"""Tests for LangfuseObservabilityAPI.get_observation() — STORY-005.

TDD anchors (4 required):
  - test_get_observation_calls_sdk_observations_get
  - test_get_observation_raises_value_error_on_empty_id
  - test_get_observation_raises_value_error_on_none_id
  - test_get_observation_propagates_404

Deviations applied:
  D-001: UnauthorizedError at langfuse.api.commons.errors.UnauthorizedError
  D-002: mock target is api.langfuse_client.get_langfuse_client

Satisfies: AC-5.1, AC-5.2, AC-5.3, AC-5.4, AC-5.5
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, sentinel

# sys.path injection — same pattern as test_get_trace.py
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


class TestGetObservation(unittest.TestCase):
    """Tests for LangfuseObservabilityAPI.get_observation().

    Satisfies: AC-5.1, AC-5.2, AC-5.3, AC-5.4, AC-5.5
    """

    # ── TDD anchor 1: basic delegation ───────────────────────────────────

    def test_get_observation_calls_sdk_observations_get(self):
        """get_observation() delegates to client.api.observations.get and returns result unchanged.

        Satisfies: AC-5.1, AC-5.3, AC-5.4
        """
        mock_client = _make_sdk_mock()
        obs_mock = MagicMock()
        obs_mock.type = 'GENERATION'  # AC-5.4: .type attribute accessible
        mock_client.api.observations.get.return_value = obs_mock

        with patch(_MOCK_TARGET, return_value=mock_client):
            api = LangfuseObservabilityAPI()
            result = api.get_observation('obs-123')

        # AC-5.3: observations.get was called with the supplied observation_id
        mock_client.api.observations.get.assert_called_once()
        call_args = mock_client.api.observations.get.call_args
        self.assertEqual(call_args.args[0], 'obs-123')  # AC-5.3
        # AC-5.3: returns SDK result verbatim
        self.assertIs(result, obs_mock)
        # AC-5.4: .type attribute accessible without modification
        self.assertEqual(result.type, 'GENERATION')  # AC-5.4

    # ── TDD anchor 2: empty string guard ─────────────────────────────────

    def test_get_observation_raises_value_error_on_empty_id(self):
        """get_observation('') raises ValueError with correct message before calling SDK.

        Satisfies: AC-5.2
        """
        mock_client = _make_sdk_mock()

        with patch(_MOCK_TARGET, return_value=mock_client):
            api = LangfuseObservabilityAPI()
            with self.assertRaises(ValueError) as ctx:  # AC-5.2
                api.get_observation('')

        # AC-5.2: message must be 'observation_id is required'
        self.assertIn('observation_id is required', str(ctx.exception))  # AC-5.2
        # Guard fires BEFORE _sdk_call — SDK must not be touched
        mock_client.api.observations.get.assert_not_called()

    # ── TDD anchor 3: None guard ──────────────────────────────────────────

    def test_get_observation_raises_value_error_on_none_id(self):
        """get_observation(None) raises ValueError before calling SDK.

        Satisfies: AC-5.2
        """
        mock_client = _make_sdk_mock()

        with patch(_MOCK_TARGET, return_value=mock_client):
            api = LangfuseObservabilityAPI()
            with self.assertRaises(ValueError):  # AC-5.2
                api.get_observation(None)

        mock_client.api.observations.get.assert_not_called()

    # ── TDD anchor 4: 404 propagation ────────────────────────────────────

    def test_get_observation_propagates_404(self):
        """SDK ApiError(404) is translated to LangfuseAPIError by _sdk_call.

        Satisfies: AC-5.5
        """
        mock_client = _make_sdk_mock()
        sdk_error = _SDKApiError(status_code=404, body='observation not found')
        mock_client.api.observations.get.side_effect = sdk_error

        with patch(_MOCK_TARGET, return_value=mock_client):
            api = LangfuseObservabilityAPI()
            with self.assertRaises(LangfuseAPIError) as ctx:  # AC-5.5
                api.get_observation('nonexistent-obs-id')

        # AC-5.5: status_code preserved through translation
        self.assertEqual(ctx.exception.status_code, 404)  # AC-5.5


if __name__ == '__main__':
    unittest.main()

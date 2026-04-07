"""
STORY-008 -- Score Configs tests
AC coverage: AC-8.1 through AC-8.5

Pattern: _make_api (direct _client injection, sets _timeout for _sdk_call).
Both list_score_configs and get_score_config use SDK dispatch via _sdk_call.

SDK path deviation (T1 finding): ScoreConfigsClient has no .list method.
  - list_score_configs → _sdk_call(score_configs.get, **kwargs)   (get = paginated list)
  - get_score_config   → _sdk_call(score_configs.get_by_id, config_id)
"""
import unittest
from unittest.mock import MagicMock

from langfuse.api.core import ApiError
from langfuse.api.commons.errors import UnauthorizedError

from api.langfuse_observability_api import LangfuseObservabilityAPI
from api.langfuse_client import LangfuseAPIError, LangfuseAuthError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_api(mock_client=None):
    """Construct LangfuseObservabilityAPI bypassing __init__ (no SDK singleton needed)."""
    api = LangfuseObservabilityAPI.__new__(LangfuseObservabilityAPI)
    api._client = mock_client if mock_client is not None else MagicMock()
    api._timeout = LangfuseObservabilityAPI.DEFAULT_TIMEOUT  # required by _sdk_call
    return api


# ---------------------------------------------------------------------------
# AC-8.1 / AC-8.2: list_score_configs
# ---------------------------------------------------------------------------

class TestListScoreConfigs(unittest.TestCase):
    """AC-8.1, AC-8.2: list_score_configs delegates to score_configs.get via _sdk_call."""

    def test_list_score_configs_delegates_to_sdk(self):
        """AC-8.1: list_score_configs calls score_configs.get (paginated list endpoint)."""
        mock_client = MagicMock()
        api = _make_api(mock_client)
        result = api.list_score_configs()
        # AC-8.1: SDK method was called
        mock_client.api.score_configs.get.assert_called_once()
        self.assertIs(result, mock_client.api.score_configs.get.return_value)

    def test_list_score_configs_passes_page_and_limit(self):
        """AC-8.1: page and limit kwargs forwarded unchanged to SDK."""
        mock_client = MagicMock()
        api = _make_api(mock_client)
        api.list_score_configs(page=2, limit=25)
        actual = str(mock_client.api.score_configs.get.call_args)
        # AC-8.1: kwargs appear in the actual call
        self.assertIn('page', actual)
        self.assertIn('2', actual)
        self.assertIn('25', actual)

    def test_list_score_configs_returns_sdk_object(self):
        """AC-8.2: list_score_configs returns SDK object unchanged — no dict conversion."""
        mock_client = MagicMock()
        sentinel = object()
        mock_client.api.score_configs.get.return_value = sentinel
        api = _make_api(mock_client)
        result = api.list_score_configs()
        # AC-8.2: sentinel returned as-is (no conversion)
        self.assertIs(result, sentinel)

    def test_list_score_configs_raises_auth_error_on_401(self):
        """AC-8.5: UnauthorizedError from SDK translated to LangfuseAuthError."""
        mock_client = MagicMock()
        mock_client.api.score_configs.get.side_effect = UnauthorizedError(body='unauthorized')
        api = _make_api(mock_client)
        with self.assertRaises(LangfuseAuthError):  # AC-8.5
            api.list_score_configs()


# ---------------------------------------------------------------------------
# AC-8.3 / AC-8.4 / AC-8.5: get_score_config
# ---------------------------------------------------------------------------

class TestGetScoreConfig(unittest.TestCase):
    """AC-8.3, AC-8.4, AC-8.5: get_score_config delegates to score_configs.get_by_id."""

    def test_get_score_config_delegates_to_sdk(self):
        """AC-8.3: get_score_config calls score_configs.get_by_id with config_id."""
        mock_client = MagicMock()
        api = _make_api(mock_client)
        result = api.get_score_config('cfg-abc')
        # AC-8.3: positional arg must be 'cfg-abc'
        self.assertEqual(
            mock_client.api.score_configs.get_by_id.call_args.args[0], 'cfg-abc'
        )
        self.assertIs(result, mock_client.api.score_configs.get_by_id.return_value)

    def test_get_score_config_raises_value_error_on_empty_id(self):
        """AC-8.3: get_score_config('') and get_score_config(None) raise ValueError."""
        api = _make_api()
        with self.assertRaises(ValueError):  # AC-8.3: empty string
            api.get_score_config('')
        with self.assertRaises(ValueError):  # AC-8.3: None
            api.get_score_config(None)

    def test_get_score_config_returns_sdk_object(self):
        """AC-8.4: get_score_config returns SDK Pydantic object with accessible attributes."""
        mock_client = MagicMock()
        sentinel = MagicMock()
        sentinel.name = 'quality'
        mock_client.api.score_configs.get_by_id.return_value = sentinel
        api = _make_api(mock_client)
        result = api.get_score_config('cfg-abc')
        # AC-8.4: sentinel returned with accessible .name attribute
        self.assertIs(result, sentinel)
        self.assertEqual(result.name, 'quality')

    def test_get_score_config_raises_api_error_on_non_2xx(self):
        """AC-8.5: SDK ApiError 404 wrapped as LangfuseAPIError with correct status_code."""
        mock_client = MagicMock()
        mock_client.api.score_configs.get_by_id.side_effect = ApiError(
            status_code=404, body='not found'
        )
        api = _make_api(mock_client)
        with self.assertRaises(LangfuseAPIError) as ctx:  # AC-8.5
            api.get_score_config('missing')
        self.assertEqual(ctx.exception.status_code, 404)  # AC-8.5: status_code preserved


if __name__ == '__main__':
    unittest.main()

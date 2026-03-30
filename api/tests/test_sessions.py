"""
STORY-006 -- list_sessions() and get_session() tests
AC coverage: AC-6.1 through AC-6.8

Pattern: _make_api (direct _client injection, sets _timeout for _sdk_call).
_sdk_call injects request_options={"timeout_in_seconds": self._timeout} when
request_options is absent; _timeout must be set on instance explicitly since
__new__ bypasses __init__.
"""
import unittest
from unittest.mock import MagicMock

from langfuse.api.core import ApiError
from langfuse.api.commons.errors import UnauthorizedError

from api.langfuse_observability_api import LangfuseObservabilityAPI
from api.langfuse_client import LangfuseAPIError, LangfuseAuthError


class TestListSessions(unittest.TestCase):

    def _make_api(self, mock_client):
        api = LangfuseObservabilityAPI.__new__(LangfuseObservabilityAPI)
        api._client = mock_client
        api._timeout = LangfuseObservabilityAPI.DEFAULT_TIMEOUT  # required by _sdk_call
        return api

    def test_list_sessions_calls_sdk_sessions_list(self):
        """AC-6.1 / AC-6.2: list_sessions delegates to api.sessions.list."""
        mock_client = MagicMock()
        api = self._make_api(mock_client)
        result = api.list_sessions()
        mock_client.api.sessions.list.assert_called_once()  # AC-6.1
        self.assertIs(result, mock_client.api.sessions.list.return_value)  # AC-6.2

    def test_list_sessions_forwards_kwargs(self):
        """AC-6.3: kwargs forwarded unchanged to SDK call."""
        mock_client = MagicMock()
        api = self._make_api(mock_client)
        api.list_sessions(page=2, environment='production')
        # AC-6.3: page and environment present in actual call
        actual = mock_client.api.sessions.list.call_args
        self.assertIn('page', str(actual))
        self.assertIn('production', str(actual))

    def test_list_sessions_no_kwargs(self):
        """AC-6.2: list_sessions() with no args calls sessions.list once."""
        mock_client = MagicMock()
        api = self._make_api(mock_client)
        api.list_sessions()
        mock_client.api.sessions.list.assert_called_once()

    def test_list_sessions_propagates_api_error(self):
        """AC-6.8: SDK ApiError is wrapped as LangfuseAPIError."""
        mock_client = MagicMock()
        mock_client.api.sessions.list.side_effect = ApiError(status_code=500, body='error')
        api = self._make_api(mock_client)
        with self.assertRaises(LangfuseAPIError):
            api.list_sessions()

    def test_list_sessions_propagates_auth_error(self):
        """AC-6.8: SDK UnauthorizedError is wrapped as LangfuseAuthError."""
        mock_client = MagicMock()
        mock_client.api.sessions.list.side_effect = UnauthorizedError(body='unauthorized')
        api = self._make_api(mock_client)
        with self.assertRaises(LangfuseAuthError):
            api.list_sessions()


class TestGetSession(unittest.TestCase):

    def _make_api(self, mock_client):
        api = LangfuseObservabilityAPI.__new__(LangfuseObservabilityAPI)
        api._client = mock_client
        api._timeout = LangfuseObservabilityAPI.DEFAULT_TIMEOUT  # required by _sdk_call
        return api

    def test_get_session_calls_sdk_sessions_get(self):
        """AC-6.6 / AC-6.7: get_session delegates to api.sessions.get(session_id)."""
        mock_client = MagicMock()
        sentinel = MagicMock()
        sentinel.traces = []
        mock_client.api.sessions.get.return_value = sentinel
        api = self._make_api(mock_client)
        result = api.get_session('sess-123')
        self.assertEqual(mock_client.api.sessions.get.call_args.args[0], 'sess-123')  # AC-6.6
        self.assertIs(result, sentinel)
        self.assertEqual(result.traces, [])  # AC-6.7

    def test_get_session_raises_value_error_on_empty_id(self):
        """AC-6.5: get_session raises ValueError for empty string."""
        mock_client = MagicMock()
        api = self._make_api(mock_client)
        with self.assertRaises(ValueError) as ctx:
            api.get_session('')
        self.assertIn('session_id is required', str(ctx.exception))

    def test_get_session_raises_value_error_on_none_id(self):
        """AC-6.5: get_session raises ValueError for None."""
        mock_client = MagicMock()
        api = self._make_api(mock_client)
        with self.assertRaises(ValueError):
            api.get_session(None)

    def test_get_session_propagates_404(self):
        """AC-6.8: SDK ApiError 404 propagates as LangfuseAPIError."""
        mock_client = MagicMock()
        mock_client.api.sessions.get.side_effect = ApiError(status_code=404, body='not found')
        api = self._make_api(mock_client)
        with self.assertRaises(LangfuseAPIError) as ctx:
            api.get_session('missing')
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == '__main__':
    unittest.main()

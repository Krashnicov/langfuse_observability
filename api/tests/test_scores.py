"""
STORY-007 -- Scores CRUD tests
AC coverage: AC-7.1 through AC-7.8

Pattern: _make_api (direct _client injection, sets _timeout for _sdk_call).
create_score uses HTTP POST (requests.post) NOT the SDK -- tested by mocking
requests.post and builtins.open (config.json read). SDK-dispatched methods
use the standard _sdk_call -> MagicMock pattern from Phase 1 tests.
"""
import base64
import json
import unittest
from unittest.mock import MagicMock, patch, mock_open

from langfuse.api.core import ApiError
from langfuse.api.commons.errors import UnauthorizedError

from api.langfuse_observability_api import LangfuseObservabilityAPI
from api.langfuse_client import LangfuseAPIError, LangfuseAuthError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Fake plugin config -- matches config.json schema used by _build_basic_auth_header
_FAKE_CONFIG = {
    "langfuse_host": "http://192.168.200.52:3010",
    "langfuse_public_key": "pk-lf-test",
    "langfuse_secret_key": "sk-lf-test",
}
_FAKE_CONFIG_JSON = json.dumps(_FAKE_CONFIG)


class MockResponse:
    """Minimal requests.Response stand-in for create_score tests."""

    def __init__(self, status_code: int, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}
        self.text = text
        self.ok = 200 <= status_code < 300

    def json(self):
        return self._json_data


def _make_api(mock_client=None):
    """Construct LangfuseObservabilityAPI bypassing __init__ (no SDK singleton needed)."""
    api = LangfuseObservabilityAPI.__new__(LangfuseObservabilityAPI)
    api._client = mock_client if mock_client is not None else MagicMock()
    api._timeout = LangfuseObservabilityAPI.DEFAULT_TIMEOUT  # required by _sdk_call
    return api


# ---------------------------------------------------------------------------
# AC-7.1 / AC-7.8: list_scores
# ---------------------------------------------------------------------------

class TestListScores(unittest.TestCase):
    """AC-7.1: list_scores delegates to self._client.api.score.list via _sdk_call."""

    def test_list_scores_delegates_to_sdk(self):
        """AC-7.1: list_scores calls api.score.list and returns its result."""
        mock_client = MagicMock()
        api = _make_api(mock_client)
        result = api.list_scores(name='quality')
        # AC-7.1: SDK method was called
        mock_client.api.score.list.assert_called_once()
        self.assertIs(result, mock_client.api.score.list.return_value)

    def test_list_scores_passes_all_filter_kwargs(self):
        """AC-7.1: All filter kwargs forwarded unchanged to SDK."""
        mock_client = MagicMock()
        api = _make_api(mock_client)
        api.list_scores(page=2, limit=50, source='API', data_type='NUMERIC')
        actual = str(mock_client.api.score.list.call_args)
        # AC-7.1: every kwarg appears in the actual call
        self.assertIn('page', actual)
        self.assertIn('50', actual)
        self.assertIn('API', actual)
        self.assertIn('NUMERIC', actual)

    def test_list_scores_propagates_api_error(self):
        """AC-7.8: SDK ApiError translated to LangfuseAPIError."""
        mock_client = MagicMock()
        mock_client.api.score.list.side_effect = ApiError(status_code=500, body='error')
        api = _make_api(mock_client)
        with self.assertRaises(LangfuseAPIError):  # AC-7.8
            api.list_scores()

    def test_list_scores_propagates_auth_error(self):
        """AC-7.8: SDK UnauthorizedError translated to LangfuseAuthError."""
        mock_client = MagicMock()
        mock_client.api.score.list.side_effect = UnauthorizedError(body='unauthorized')
        api = _make_api(mock_client)
        with self.assertRaises(LangfuseAuthError):  # AC-7.8
            api.list_scores()


# ---------------------------------------------------------------------------
# AC-7.2 / AC-7.8: get_score
# ---------------------------------------------------------------------------

class TestGetScore(unittest.TestCase):
    """AC-7.2: get_score delegates to self._client.api.score.get; guards on falsy id."""

    def test_get_score_delegates_to_sdk(self):
        """AC-7.2: get_score calls api.score.get with score_id positional arg."""
        mock_client = MagicMock()
        api = _make_api(mock_client)
        result = api.get_score('score-123')
        # AC-7.2: positional arg must be 'score-123'
        self.assertEqual(mock_client.api.score.get.call_args.args[0], 'score-123')
        self.assertIs(result, mock_client.api.score.get.return_value)

    def test_get_score_raises_value_error_on_empty_id(self):
        """AC-7.2: get_score('') raises ValueError before SDK call."""
        api = _make_api()
        with self.assertRaises(ValueError):  # AC-7.2
            api.get_score('')

    def test_get_score_raises_value_error_on_none_id(self):
        """AC-7.2: get_score(None) raises ValueError before SDK call."""
        api = _make_api()
        with self.assertRaises(ValueError):  # AC-7.2
            api.get_score(None)

    def test_get_score_propagates_api_error(self):
        """AC-7.8: SDK ApiError 404 wrapped as LangfuseAPIError with correct status_code."""
        mock_client = MagicMock()
        mock_client.api.score.get.side_effect = ApiError(status_code=404, body='not found')
        api = _make_api(mock_client)
        with self.assertRaises(LangfuseAPIError) as ctx:  # AC-7.8
            api.get_score('missing')
        self.assertEqual(ctx.exception.status_code, 404)


# ---------------------------------------------------------------------------
# AC-7.3 / AC-7.6 / AC-7.7: create_score (HTTP POST — NOT SDK)
# ---------------------------------------------------------------------------

class TestCreateScore(unittest.TestCase):
    """AC-7.3: create_score POSTs directly; AC-7.6: no hardcoded creds; AC-7.7: error handling."""

    def test_create_score_uses_http_post_not_sdk(self):
        """AC-7.3: create_score calls requests.post, NOT _sdk_call."""
        api = _make_api()
        api._sdk_call = MagicMock()  # sentinel -- must NOT be called
        ok_resp = MockResponse(200, {'id': 'sc-1'})
        with patch('builtins.open', mock_open(read_data=_FAKE_CONFIG_JSON)):
            with patch('requests.post', return_value=ok_resp) as mock_post:
                api.create_score(name='q', value=0.8, trace_id='t1')
        mock_post.assert_called_once()       # AC-7.3: HTTP POST used
        api._sdk_call.assert_not_called()   # AC-7.3: SDK NOT used

    def test_create_score_constructs_basic_auth_header(self):
        """AC-7.6: requests.post receives Authorization header starting with 'Basic '."""
        api = _make_api()
        ok_resp = MockResponse(200, {'id': 'sc-1'})
        with patch('builtins.open', mock_open(read_data=_FAKE_CONFIG_JSON)):
            with patch('requests.post', return_value=ok_resp) as mock_post:
                api.create_score(name='q', value=0.8, trace_id='t1')
        headers = mock_post.call_args.kwargs.get('headers', {})
        # AC-7.6: Authorization header present and correctly prefixed
        self.assertTrue(
            headers.get('Authorization', '').startswith('Basic '),
            f"Expected 'Basic ' prefix, got: {headers.get('Authorization', '')!r}"
        )

    def test_create_score_raises_auth_error_on_401(self):
        """AC-7.7: HTTP 401 from Langfuse raises LangfuseAuthError."""
        api = _make_api()
        auth_err = MockResponse(401, text='unauthorized')
        with patch('builtins.open', mock_open(read_data=_FAKE_CONFIG_JSON)):
            with patch('requests.post', return_value=auth_err):
                with self.assertRaises(LangfuseAuthError):  # AC-7.7
                    api.create_score(name='q', value=0.8, trace_id='t1')

    def test_create_score_raises_auth_error_on_403(self):
        """AC-7.7: HTTP 403 from Langfuse raises LangfuseAuthError."""
        api = _make_api()
        forbidden = MockResponse(403, text='forbidden')
        with patch('builtins.open', mock_open(read_data=_FAKE_CONFIG_JSON)):
            with patch('requests.post', return_value=forbidden):
                with self.assertRaises(LangfuseAuthError):  # AC-7.7
                    api.create_score(name='q', value=0.8, trace_id='t1')

    def test_create_score_raises_api_error_on_non_2xx(self):
        """AC-7.7: HTTP 500 raises LangfuseAPIError with status_code == 500."""
        api = _make_api()
        server_err = MockResponse(500, text='internal server error')
        with patch('builtins.open', mock_open(read_data=_FAKE_CONFIG_JSON)):
            with patch('requests.post', return_value=server_err):
                with self.assertRaises(LangfuseAPIError) as ctx:  # AC-7.7
                    api.create_score(name='q', value=0.8, trace_id='t1')
        self.assertEqual(ctx.exception.status_code, 500)  # AC-7.7: status_code preserved

    def test_create_score_returns_response_json(self):
        """AC-7.3: create_score returns response JSON as a plain dict."""
        api = _make_api()
        ok_resp = MockResponse(200, json_data={'id': 'sc-1'})
        with patch('builtins.open', mock_open(read_data=_FAKE_CONFIG_JSON)):
            with patch('requests.post', return_value=ok_resp):
                result = api.create_score(name='q', value=0.8, trace_id='t1')
        self.assertEqual(result, {'id': 'sc-1'})  # AC-7.3

    def test_create_score_no_hardcoded_credentials(self):
        """AC-7.6: Auth header token encodes keys read from config -- not hardcoded."""
        api = _make_api()
        ok_resp = MockResponse(200, {'id': 'sc-2'})
        custom_config = json.dumps({
            "langfuse_host": "http://192.168.200.52:3010",
            "langfuse_public_key": "pk-custom",
            "langfuse_secret_key": "sk-custom",
        })
        expected_token = base64.b64encode(b'pk-custom:sk-custom').decode()
        with patch('builtins.open', mock_open(read_data=custom_config)):
            with patch('requests.post', return_value=ok_resp) as mock_post:
                api.create_score(name='q', value=0.8, trace_id='t1')
        headers = mock_post.call_args.kwargs.get('headers', {})
        # AC-7.6: token matches base64(pk-custom:sk-custom) from config
        self.assertIn(expected_token, headers.get('Authorization', ''))


# ---------------------------------------------------------------------------
# AC-7.4 / AC-7.8: delete_score
# ---------------------------------------------------------------------------

class TestDeleteScore(unittest.TestCase):
    """AC-7.4: delete_score delegates to self._client.api.score.delete; guards on falsy id."""

    def test_delete_score_delegates_to_sdk(self):
        """AC-7.4: delete_score calls api.score.delete with score_id positional arg."""
        mock_client = MagicMock()
        api = _make_api(mock_client)
        api.delete_score('score-456')
        # AC-7.4: positional arg must be 'score-456'
        self.assertEqual(mock_client.api.score.delete.call_args.args[0], 'score-456')

    def test_delete_score_raises_value_error_on_empty_id(self):
        """AC-7.4: delete_score('') raises ValueError before SDK call."""
        api = _make_api()
        with self.assertRaises(ValueError):  # AC-7.4
            api.delete_score('')

    def test_delete_score_raises_value_error_on_none_id(self):
        """AC-7.4: delete_score(None) raises ValueError before SDK call."""
        api = _make_api()
        with self.assertRaises(ValueError):  # AC-7.4
            api.delete_score(None)

    def test_delete_score_propagates_api_error(self):
        """AC-7.8: SDK ApiError wrapped as LangfuseAPIError."""
        mock_client = MagicMock()
        mock_client.api.score.delete.side_effect = ApiError(status_code=404, body='not found')
        api = _make_api(mock_client)
        with self.assertRaises(LangfuseAPIError):  # AC-7.8
            api.delete_score('score-xyz')


# ---------------------------------------------------------------------------
# AC-7.5: list_scores_v2
# ---------------------------------------------------------------------------

class TestListScoresV2(unittest.TestCase):
    """AC-7.5: list_scores_v2 calls v2 or gracefully falls back to v1."""

    def test_list_scores_v2_returns_results(self):
        """AC-7.5: list_scores_v2 delegates to _sdk_call and returns results."""
        mock_client = MagicMock()
        api = _make_api(mock_client)
        result = api.list_scores_v2(session_id='sess-1')
        # AC-7.5: some SDK score list method was invoked
        mock_client.api.score.list.assert_called_once()
        self.assertIs(result, mock_client.api.score.list.return_value)

    def test_list_scores_v2_passes_kwargs(self):
        """AC-7.5: kwargs forwarded to underlying SDK call."""
        mock_client = MagicMock()
        api = _make_api(mock_client)
        api.list_scores_v2(session_id='sess-1', page=3)
        actual = str(mock_client.api.score.list.call_args)
        self.assertIn('sess-1', actual)


if __name__ == '__main__':
    unittest.main()

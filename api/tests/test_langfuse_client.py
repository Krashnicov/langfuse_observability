"""Tests for api/langfuse_client.py — LangfuseClient base class.

TDD anchors from STORY-001. All 6 required anchor tests plus coverage tests.
Run from plugin root: pytest api/tests/test_langfuse_client.py -v
"""
import os
import sys

# Ensure plugin root is on sys.path so api.langfuse_client is importable
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_API_DIR = os.path.dirname(_TEST_DIR)
_PLUGIN_ROOT = os.path.dirname(_API_DIR)
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)

import pytest
from unittest.mock import MagicMock, patch

from langfuse.api.core import ApiError as _SDKApiError
from langfuse.api.commons.errors import UnauthorizedError as _SDKUnauthorizedError

from api.langfuse_client import LangfuseClient, LangfuseAPIError, LangfuseAuthError

# Patch target: where get_langfuse_client is used (imported into langfuse_client module)
_PATCH = "api.langfuse_client.get_langfuse_client"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(timeout=30):
    """Return a LangfuseClient with a mocked SDK singleton."""
    mock_sdk = MagicMock()
    with patch(_PATCH, return_value=mock_sdk):
        client = LangfuseClient(timeout=timeout)
    return client, mock_sdk


# ---------------------------------------------------------------------------
# Error class tests  (AC-1.3)
# ---------------------------------------------------------------------------

class TestLangfuseErrorClasses:
    """Satisfies: AC-1.3"""

    def test_langfuse_api_error_has_status_code_and_body(self):
        # AC-1.3: LangfuseAPIError has status_code (int) and body attributes
        err = LangfuseAPIError(status_code=500, message="server error", body={"detail": "x"})
        assert err.status_code == 500
        assert err.body == {"detail": "x"}

    def test_langfuse_api_error_is_exception(self):
        # AC-1.3: LangfuseAPIError subclasses Exception
        assert issubclass(LangfuseAPIError, Exception)

    def test_langfuse_auth_error_subclasses_api_error(self):
        # AC-1.3: LangfuseAuthError(LangfuseAPIError) subclasses it
        assert issubclass(LangfuseAuthError, LangfuseAPIError)

    def test_langfuse_auth_error_attributes(self):
        # AC-1.3: LangfuseAuthError inherits status_code and body
        err = LangfuseAuthError(status_code=403, message="forbidden", body=None)
        assert err.status_code == 403
        assert err.body is None

    def test_langfuse_api_error_str_includes_status_code(self):
        # AC-1.3: str(err) includes status code for debuggability
        err = LangfuseAPIError(status_code=503, message="unavailable")
        assert "503" in str(err)


# ---------------------------------------------------------------------------
# LangfuseClient.__init__ tests  (AC-1.4)
# ---------------------------------------------------------------------------

class TestLangfuseClientInit:
    """Satisfies: AC-1.4"""

    def test_client_raises_runtime_error_when_sdk_is_none(self):
        # AC-1.4: raises RuntimeError with descriptive message when get_langfuse_client() returns None
        with patch(_PATCH, return_value=None):
            with pytest.raises(RuntimeError) as exc_info:
                LangfuseClient()
        assert "Langfuse is not available" in str(exc_info.value)

    def test_client_stores_sdk_and_timeout(self):
        # AC-1.4: stores SDK singleton as self._client and timeout as self._timeout
        client, mock_sdk = _make_client(timeout=30)
        assert client._client is mock_sdk
        assert client._timeout == 30

    def test_client_default_timeout_is_30(self):
        # AC-1.4: DEFAULT_TIMEOUT is 30 seconds
        assert LangfuseClient.DEFAULT_TIMEOUT == 30

    def test_client_accepts_custom_timeout(self):
        # AC-1.4: custom timeout stored correctly
        client, _ = _make_client(timeout=60)
        assert client._timeout == 60

    def test_runtime_error_message_mentions_config(self):
        # AC-1.4: error message guides user to fix — mentions key names
        with patch(_PATCH, return_value=None):
            with pytest.raises(RuntimeError) as exc_info:
                LangfuseClient()
        msg = str(exc_info.value)
        # Must mention how to fix (key names or config)
        assert any(kw in msg for kw in ["public_key", "secret_key", "LANGFUSE_PUBLIC_KEY", "langfuse_public_key"])


# ---------------------------------------------------------------------------
# LangfuseClient._sdk_call tests  (AC-1.5)
# ---------------------------------------------------------------------------

class TestLangfuseClientSdkCall:
    """Satisfies: AC-1.5"""

    def test_sdk_call_injects_timeout_when_not_present(self):
        # AC-1.5: injects request_options={"timeout_in_seconds": self._timeout} when absent
        client, _ = _make_client(timeout=30)
        mock_fn = MagicMock(return_value="result")
        result = client._sdk_call(mock_fn, "arg1")
        mock_fn.assert_called_once_with(
            "arg1", request_options={"timeout_in_seconds": 30}
        )
        assert result == "result"

    def test_sdk_call_does_not_override_existing_request_options(self):
        # AC-1.5: does not replace request_options already present in kwargs
        client, _ = _make_client(timeout=30)
        mock_fn = MagicMock(return_value="ok")
        client._sdk_call(mock_fn, request_options={"timeout_in_seconds": 5})
        mock_fn.assert_called_once_with(request_options={"timeout_in_seconds": 5})

    def test_sdk_call_raises_langfuse_auth_error_on_unauthorized(self):
        # AC-1.5: translates _SDKUnauthorizedError -> LangfuseAuthError
        client, _ = _make_client()
        mock_fn = MagicMock(
            side_effect=_SDKUnauthorizedError(body="Unauthorized")
        )
        with pytest.raises(LangfuseAuthError) as exc_info:
            client._sdk_call(mock_fn)
        assert exc_info.value.status_code == 401

    def test_sdk_call_raises_langfuse_api_error_on_api_error(self):
        # AC-1.5: translates _SDKApiError -> LangfuseAPIError preserving status_code
        client, _ = _make_client()
        mock_fn = MagicMock(
            side_effect=_SDKApiError(status_code=404, body="not found")
        )
        with pytest.raises(LangfuseAPIError) as exc_info:
            client._sdk_call(mock_fn)
        assert exc_info.value.status_code == 404

    def test_sdk_call_auth_error_is_caught_as_api_error(self):
        # AC-1.3 + AC-1.5: LangfuseAuthError IS-A LangfuseAPIError — catch as parent works
        client, _ = _make_client()
        mock_fn = MagicMock(
            side_effect=_SDKUnauthorizedError(body="Unauthorized")
        )
        with pytest.raises(LangfuseAPIError):
            client._sdk_call(mock_fn)

    def test_sdk_call_preserves_body_on_api_error(self):
        # AC-1.5: body preserved from SDK exception
        client, _ = _make_client()
        mock_fn = MagicMock(
            side_effect=_SDKApiError(status_code=422, body={"error": "validation"})
        )
        with pytest.raises(LangfuseAPIError) as exc_info:
            client._sdk_call(mock_fn)
        assert exc_info.value.body == {"error": "validation"}

    def test_sdk_call_returns_fn_result_on_success(self):
        # AC-1.5: returns fn result when no exception
        client, _ = _make_client()
        mock_fn = MagicMock(return_value={"id": "trace-123"})
        result = client._sdk_call(mock_fn)
        assert result == {"id": "trace-123"}

    def test_sdk_call_passes_positional_args(self):
        # AC-1.5: positional args forwarded to fn
        client, _ = _make_client()
        mock_fn = MagicMock(return_value=None)
        client._sdk_call(mock_fn, "pos1", "pos2")
        mock_fn.assert_called_once_with(
            "pos1", "pos2", request_options={"timeout_in_seconds": 30}
        )

    def test_sdk_call_auth_error_with_body_preserved(self):
        # AC-1.5: body preserved from UnauthorizedError
        client, _ = _make_client()
        mock_fn = MagicMock(
            side_effect=_SDKUnauthorizedError(body="invalid credentials")
        )
        with pytest.raises(LangfuseAuthError) as exc_info:
            client._sdk_call(mock_fn)
        assert exc_info.value.body == "invalid credentials"

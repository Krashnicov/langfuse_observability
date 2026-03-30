"""LangfuseClient — thin wrapper over the Langfuse SDK singleton.

Provides typed error handling and timeout injection for all Phase 1
endpoint methods in LangfuseObservabilityAPI. HTTP dispatch is delegated
to the SDK internals via _sdk_call(); this module contains NO credentials,
NO direct HTTP calls, and NO config.json reads.

Satisfies: AC-1.1, AC-1.2, AC-1.3, AC-1.4, AC-1.5, AC-1.6
"""
# AC-1.2: sys.path injection using _PLUGIN_ROOT — matching api/langfuse_trace.py lines 1-6
import os
import sys

_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PLUGIN_ROOT not in sys.path:
    sys.path.append(_PLUGIN_ROOT)

# DEVIATION from story spec: UnauthorizedError lives at langfuse.api.commons.errors,
# NOT langfuse.api.core (which only exports ApiError). UnauthorizedError IS-A ApiError
# with hardcoded status_code=401. Confirmed via SDK inspection 2026-03-30.
from langfuse.api.core import ApiError as _SDKApiError  # AC-1.5
from langfuse.api.commons.errors import UnauthorizedError as _SDKUnauthorizedError  # AC-1.5
from langfuse_helpers.langfuse_helper import get_langfuse_client  # AC-1.4, AC-1.6


# AC-1.3: LangfuseAPIError(Exception) with status_code (int) and body attributes
class LangfuseAPIError(Exception):
    """Raised when the Langfuse API returns a non-2xx response.

    Attributes:
        status_code: HTTP status code from the API response.
        body: Response body (may be dict, str, or None).

    Satisfies: AC-1.3
    """

    def __init__(self, status_code: int, message: str, body=None) -> None:
        # AC-1.3: status_code (int) and body attributes
        self.status_code = status_code
        self.body = body
        super().__init__(f"Langfuse API error {status_code}: {message}")


# AC-1.3: LangfuseAuthError(LangfuseAPIError) subclasses for 401/403 responses
class LangfuseAuthError(LangfuseAPIError):
    """Raised on 401/403 Unauthorized responses.

    Satisfies: AC-1.3
    """


# AC-1.4: LangfuseClient.__init__(timeout=30)
class LangfuseClient:
    """Thin wrapper over the Langfuse SDK singleton.

    Provides timeout injection and typed error translation for all
    endpoint methods. Does NOT read config.json or hold credentials.

    Satisfies: AC-1.2, AC-1.4, AC-1.5, AC-1.6
    """

    DEFAULT_TIMEOUT: int = 30  # AC-1.4

    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        """Initialise the client.

        Args:
            timeout: Request timeout in seconds (default 30).

        Raises:
            RuntimeError: If the Langfuse SDK singleton is unavailable
                (disabled or unconfigured).

        Satisfies: AC-1.4, AC-1.6
        """
        self._timeout = timeout  # AC-1.4: stored as self._timeout
        # AC-1.4: call get_langfuse_client() — all auth state lives there (AC-1.6)
        sdk = get_langfuse_client()
        if sdk is None:
            # AC-1.4: raise RuntimeError with descriptive message if result is None
            raise RuntimeError(
                "Langfuse is not available. Verify langfuse_enabled is True and "
                "langfuse_public_key / langfuse_secret_key are set in plugin config "
                "or LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY environment variables."
            )
        self._client = sdk  # AC-1.4: stored as self._client

    def _sdk_call(self, fn, *args, **kwargs):
        """Call an SDK function with timeout injection and error translation.

        Injects request_options={"timeout_in_seconds": self._timeout} when
        request_options is not already present in kwargs. Translates SDK
        exceptions into typed LangfuseAPIError / LangfuseAuthError.

        Args:
            fn: Callable SDK method to invoke.
            *args: Positional arguments forwarded to fn.
            **kwargs: Keyword arguments forwarded to fn.

        Returns:
            The return value of fn(*args, **kwargs).

        Raises:
            LangfuseAuthError: On 401/403 Unauthorized SDK responses.
            LangfuseAPIError: On any other SDK API error response.

        Satisfies: AC-1.5
        """
        # AC-1.5: inject timeout when not already present
        if "request_options" not in kwargs:
            kwargs["request_options"] = {"timeout_in_seconds": self._timeout}

        try:
            return fn(*args, **kwargs)
        except _SDKUnauthorizedError as e:
            # AC-1.5: translate _SDKUnauthorizedError -> LangfuseAuthError
            # Must catch BEFORE _SDKApiError since UnauthorizedError IS-A ApiError
            raise LangfuseAuthError(
                getattr(e, "status_code", 401),
                str(e),
                body=getattr(e, "body", None),
            ) from e
        except _SDKApiError as e:
            # AC-1.5: translate _SDKApiError -> LangfuseAPIError preserving status_code and body
            raise LangfuseAPIError(
                getattr(e, "status_code", 0),
                str(e),
                body=getattr(e, "body", None),
            ) from e

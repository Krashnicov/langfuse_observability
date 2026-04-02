"""Smoke tests for native A0 plugin config scoping.

Verifies that get_langfuse_config() reads flat keys directly from the resolved
config dict (as returned by plugins.get_plugin_config) without any custom
profile resolution logic.

All tests use _raw_config injection to avoid requiring a live A0 environment.
"""
import sys
import os
import pytest
import unittest.mock as _um

# Ensure plugin root is on the path for import
_plugin_root = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _plugin_root not in sys.path:
    sys.path.insert(0, _plugin_root)

from langfuse_helpers.langfuse_helper import get_langfuse_config


_FLAT_CONFIG = {
    "langfuse_enabled": True,
    "langfuse_public_key": "pk-lf-smoke",
    "langfuse_secret_key": "sk-lf-smoke",
    "langfuse_host": "http://localhost:3010/",
    "langfuse_sample_rate": 0.5,
    "langfuse_service_name": "test-service",
    "langfuse_environment": "testing",
    "langfuse_release": "v1.0.0",
    "langfuse_trace_name_template": "{profile}@{model}",
}


class TestGetLangfuseConfigFlatKeys:
    """Smoke tests verifying flat-key config resolution."""

    def test_returns_dict_with_all_expected_keys(self):
        result = get_langfuse_config(_raw_config=_FLAT_CONFIG)
        expected_keys = {
            "enabled", "public_key", "secret_key", "host",
            "sample_rate", "service_name", "environment",
            "release", "trace_name_template",
        }
        assert expected_keys.issubset(result.keys()), f"Missing keys: {expected_keys - result.keys()}"

    def test_no_legacy_profile_keys_in_result(self):
        result = get_langfuse_config(_raw_config=_FLAT_CONFIG)
        for removed_key in ("profile_name", "label"):
            assert removed_key not in result

    def test_enabled_read_from_flat_key(self):
        result = get_langfuse_config(_raw_config=_FLAT_CONFIG)
        assert result["enabled"] is True

    def test_public_key_read_from_flat_key(self):
        result = get_langfuse_config(_raw_config=_FLAT_CONFIG)
        assert result["public_key"] == "pk-lf-smoke"

    def test_secret_key_read_from_flat_key(self):
        result = get_langfuse_config(_raw_config=_FLAT_CONFIG)
        assert result["secret_key"] == "sk-lf-smoke"

    def test_host_read_from_flat_key(self):
        result = get_langfuse_config(_raw_config=_FLAT_CONFIG)
        assert result["host"] == "http://localhost:3010/"

    def test_sample_rate_read_from_flat_key(self):
        result = get_langfuse_config(_raw_config=_FLAT_CONFIG)
        assert result["sample_rate"] == 0.5

    def test_service_name_read_from_flat_key(self):
        result = get_langfuse_config(_raw_config=_FLAT_CONFIG)
        assert result["service_name"] == "test-service"

    def test_environment_read_from_flat_key(self):
        result = get_langfuse_config(_raw_config=_FLAT_CONFIG)
        assert result["environment"] == "testing"

    def test_release_read_from_flat_key(self):
        result = get_langfuse_config(_raw_config=_FLAT_CONFIG)
        assert result["release"] == "v1.0.0"

    def test_trace_name_template_read_from_flat_key(self):
        result = get_langfuse_config(_raw_config=_FLAT_CONFIG)
        assert result["trace_name_template"] == "{profile}@{model}"

    def test_agent_none_uses_raw_config_directly(self):
        result = get_langfuse_config(agent=None, _raw_config=_FLAT_CONFIG)
        assert result["public_key"] == "pk-lf-smoke"
        assert result["host"] == "http://localhost:3010/"

    def test_empty_raw_config_returns_safe_defaults(self):
        result = get_langfuse_config(_raw_config={})
        assert isinstance(result, dict)
        assert "enabled" in result
        assert "public_key" in result
        assert "secret_key" in result

    def test_disabled_flag_respected(self):
        cfg = dict(_FLAT_CONFIG, langfuse_enabled=False,
                   langfuse_public_key="", langfuse_secret_key="")
        result = get_langfuse_config(_raw_config=cfg)
        assert result["enabled"] is False

    def test_auto_enable_when_keys_present_but_flag_false(self):
        cfg = dict(_FLAT_CONFIG, langfuse_enabled=False)
        result = get_langfuse_config(_raw_config=cfg)
        assert result["enabled"] is True

    def test_org_id_not_in_result(self):
        """org_id must NOT be present in the resolved config dict (field removed)."""
        result = get_langfuse_config(_raw_config=_FLAT_CONFIG)
        assert "org_id" not in result

    def test_project_name_not_in_result(self):
        """project_name must NOT be present in the resolved config dict (field removed)."""
        result = get_langfuse_config(_raw_config=_FLAT_CONFIG)
        assert "project_name" not in result


# ---------------------------------------------------------------------------
# A0 framework stubs - allow importing api.langfuse_test without A0 runtime
# ---------------------------------------------------------------------------
_helpers_api_stub = _um.MagicMock()
_helpers_api_stub.ApiHandler = object
_helpers_api_stub.Request = dict
_helpers_api_stub.Response = dict

sys.modules.setdefault("helpers", _um.MagicMock())
sys.modules.setdefault("helpers.api", _helpers_api_stub)
sys.modules.setdefault("helpers.plugins", _um.MagicMock())

from api.langfuse_test import _resolve_project_info  # noqa: E402


class TestLangfuseTestProjectResolution:
    """Tests for the /api/public/projects extension in api/langfuse_test.py."""

    def test_resolve_success_returns_project_and_org(self):
        mock_resp = _um.MagicMock()
        mock_resp.json.return_value = {
            "data": [{"name": "Default Project", "organization": {"name": "Default Org"}}]
        }
        mock_resp.raise_for_status = _um.MagicMock()
        with _um.patch("httpx.get", return_value=mock_resp):
            result = _resolve_project_info("pk-lf-test", "sk-lf-test", "http://localhost:3010/")
        assert result["project"] == "Default Project"
        assert result["org"] == "Default Org"

    def test_resolve_empty_data_returns_empty_strings(self):
        mock_resp = _um.MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status = _um.MagicMock()
        with _um.patch("httpx.get", return_value=mock_resp):
            result = _resolve_project_info("pk-lf-test", "sk-lf-test", "http://localhost:3010/")
        assert result["project"] == ""
        assert result["org"] == ""

    def test_resolve_request_exception_returns_empty_strings(self):
        with _um.patch("httpx.get", side_effect=Exception("connection refused")):
            result = _resolve_project_info("pk-lf-test", "sk-lf-test", "http://localhost:3010/")
        assert result["project"] == ""
        assert result["org"] == ""

    def test_resolve_missing_org_key_returns_empty_org(self):
        mock_resp = _um.MagicMock()
        mock_resp.json.return_value = {"data": [{"name": "My Project"}]}
        mock_resp.raise_for_status = _um.MagicMock()
        with _um.patch("httpx.get", return_value=mock_resp):
            result = _resolve_project_info("pk-lf-test", "sk-lf-test", "http://localhost:3010/")
        assert result["project"] == "My Project"
        assert result["org"] == ""

    def test_success_response_includes_project_and_org_keys(self):
        import asyncio
        from api.langfuse_test import LangfuseTest
        handler = LangfuseTest()
        mock_client = _um.MagicMock()
        mock_client.auth_check.return_value = True
        mock_client.flush.return_value = None
        with _um.patch("api.langfuse_test._resolve_project_info",
                       return_value={"project": "Default Project", "org": "Default Org"}), \
             _um.patch("langfuse.Langfuse", return_value=mock_client):
            result = asyncio.run(handler.process(
                {"public_key": "pk-lf-default", "secret_key": "sk-lf-default",
                 "host": "http://localhost:3010/"},
                None
            ))
        assert result["success"] is True
        assert "project" in result
        assert "org" in result
        assert result["project"] == "Default Project"
        assert result["org"] == "Default Org"

    def test_failure_response_has_no_project_or_org_keys(self):
        import asyncio
        from api.langfuse_test import LangfuseTest
        handler = LangfuseTest()
        mock_client = _um.MagicMock()
        mock_client.auth_check.return_value = False
        mock_client.flush.return_value = None
        with _um.patch("langfuse.Langfuse", return_value=mock_client):
            result = asyncio.run(handler.process(
                {"public_key": "pk-lf-default", "secret_key": "sk-lf-default",
                 "host": "http://localhost:3010/"},
                None
            ))
        assert result["success"] is False
        assert "project" not in result
        assert "org" not in result

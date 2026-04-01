"""Smoke tests for native A0 plugin config scoping.

Verifies that get_langfuse_config() reads flat keys directly from the resolved
config dict (as returned by plugins.get_plugin_config) without any custom
profile resolution logic.

All tests use _raw_config injection to avoid requiring a live A0 environment.
"""
import sys
import os
import pytest

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
        """get_langfuse_config() must return a dict containing all required flat keys."""
        result = get_langfuse_config(_raw_config=_FLAT_CONFIG)
        expected_keys = {
            "enabled", "public_key", "secret_key", "host",
            "sample_rate", "service_name", "environment",
            "release", "trace_name_template",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - result.keys()}"
        )

    def test_no_legacy_profile_keys_in_result(self):
        """Result must NOT contain legacy custom-profile keys removed in this rework."""
        result = get_langfuse_config(_raw_config=_FLAT_CONFIG)
        for removed_key in ("profile_name", "org_id", "label"):
            assert removed_key not in result, (
                f"Legacy key '{removed_key}' should not be present in result"
            )

    def test_enabled_read_from_flat_key(self):
        """enabled maps directly from langfuse_enabled."""
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
        """agent=None with _raw_config returns the same flat-key result."""
        result = get_langfuse_config(agent=None, _raw_config=_FLAT_CONFIG)
        assert result["public_key"] == "pk-lf-smoke"
        assert result["host"] == "http://localhost:3010/"

    def test_empty_raw_config_returns_safe_defaults(self):
        """Empty config dict must not raise — returns safe defaults."""
        result = get_langfuse_config(_raw_config={})
        assert isinstance(result, dict)
        assert "enabled" in result
        assert "public_key" in result
        assert "secret_key" in result

    def test_disabled_flag_respected(self):
        """langfuse_enabled=False is returned as enabled=False."""
        cfg = dict(_FLAT_CONFIG, langfuse_enabled=False,
                   langfuse_public_key="", langfuse_secret_key="")
        result = get_langfuse_config(_raw_config=cfg)
        assert result["enabled"] is False

    def test_auto_enable_when_keys_present_but_flag_false(self):
        """If keys are set but enabled=False, auto-enable kicks in."""
        cfg = dict(_FLAT_CONFIG, langfuse_enabled=False)
        result = get_langfuse_config(_raw_config=cfg)
        # Keys are present → auto-enabled
        assert result["enabled"] is True

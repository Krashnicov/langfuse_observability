"""Tests for multi-profile configuration and resolution logic.

Covers:
  - _resolve_profile_name: agent_map → project_map → default chain
  - _get_profile_config: flat-key fallback, sparse overrides, full profile
  - get_langfuse_config (via _raw_config injection): backward compat + multi-profile

All tests are pure-unit — no file I/O, no Langfuse SDK required.

Run from plugin root:
    pytest api/tests/test_profile_resolution.py -v
"""
import os
import sys

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_API_DIR = os.path.dirname(_TEST_DIR)
_PLUGIN_ROOT = os.path.dirname(_API_DIR)
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)

import pytest
from unittest.mock import MagicMock, patch

from langfuse_helpers.langfuse_helper import (
    _resolve_profile_name,
    _get_profile_config,
    get_langfuse_config,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_agent(profile: str = "default", project=None):
    """Return a minimal mock agent mimicking A0 Agent attributes."""
    agent = MagicMock()
    agent.config.profile = profile
    agent.agent_name = profile  # fallback attribute
    # context.project can be a str or object with .name
    if project is None:
        agent.context.project = None
    elif isinstance(project, str):
        agent.context.project = project
    else:
        # project object with .name
        proj_obj = MagicMock()
        proj_obj.name = project
        agent.context.project = proj_obj
    return agent


def _raw(profiles=None, agent_map=None, project_map=None, **flat_keys):
    """Build a minimal raw_config dict."""
    cfg = dict(flat_keys)
    if profiles is not None:
        cfg["profiles"] = profiles
    if agent_map is not None:
        cfg["agent_map"] = agent_map
    if project_map is not None:
        cfg["project_map"] = project_map
    return cfg


_STAGING_PROFILE = {
    "langfuse_public_key": "pk-staging",
    "langfuse_secret_key": "sk-staging",
    "langfuse_host": "https://staging.example.com",
}

_PROD_PROFILE = {
    "langfuse_public_key": "pk-prod",
    "langfuse_secret_key": "sk-prod",
    "langfuse_host": "https://prod.example.com",
}


# ===========================================================================
# _resolve_profile_name tests
# ===========================================================================

class TestResolveProfileName:

    # -----------------------------------------------------------------------
    # No-agent / no-maps → always "default"
    # -----------------------------------------------------------------------

    def test_returns_default_when_agent_is_none(self):
        """No agent → always default profile."""
        cfg = _raw()
        assert _resolve_profile_name(cfg, None) == "default"

    def test_returns_default_when_maps_empty(self):
        """Agent present but empty maps → default."""
        cfg = _raw(profiles={"staging": _STAGING_PROFILE}, agent_map={}, project_map={})
        agent = _make_agent(profile="researcher")
        assert _resolve_profile_name(cfg, agent) == "default"

    def test_returns_default_when_no_profiles_key(self):
        """Backward compat: flat-only config (no 'profiles' key) → default."""
        cfg = _raw(
            langfuse_public_key="pk-flat",
            langfuse_secret_key="sk-flat",
        )
        agent = _make_agent(profile="hacker")
        assert _resolve_profile_name(cfg, agent) == "default"

    # -----------------------------------------------------------------------
    # Tier 1: agent_map
    # -----------------------------------------------------------------------

    def test_agent_map_match_returns_mapped_profile(self):
        """agent_map[agent_profile] → mapped profile name (profile exists)."""
        cfg = _raw(
            profiles={"staging": _STAGING_PROFILE},
            agent_map={"researcher": "staging"},
        )
        agent = _make_agent(profile="researcher")
        assert _resolve_profile_name(cfg, agent) == "staging"

    def test_agent_map_skipped_when_mapped_profile_missing(self):
        """agent_map maps to a profile name not in profiles → skip to next tier."""
        cfg = _raw(
            profiles={"staging": _STAGING_PROFILE},
            agent_map={"researcher": "nonexistent"},
            project_map={},
        )
        agent = _make_agent(profile="researcher")
        # nonexistent not in profiles → falls through to default
        assert _resolve_profile_name(cfg, agent) == "default"

    def test_agent_map_uses_agent_name_fallback(self):
        """Falls back to agent.agent_name when agent.config.profile is empty."""
        cfg = _raw(
            profiles={"prod": _PROD_PROFILE},
            agent_map={"custom-agent": "prod"},
        )
        agent = _make_agent(profile="")
        agent.agent_name = "custom-agent"
        assert _resolve_profile_name(cfg, agent) == "prod"

    def test_agent_map_takes_priority_over_project_map(self):
        """agent_map match wins even when project_map also has a valid entry."""
        cfg = _raw(
            profiles={"staging": _STAGING_PROFILE, "prod": _PROD_PROFILE},
            agent_map={"researcher": "staging"},
            project_map={"my-project": "prod"},
        )
        agent = _make_agent(profile="researcher", project="my-project")
        assert _resolve_profile_name(cfg, agent) == "staging"

    # -----------------------------------------------------------------------
    # Tier 2: project_map
    # -----------------------------------------------------------------------

    def test_project_map_match_returns_mapped_profile(self):
        """project_map[project_name] → mapped profile when agent_map has no match."""
        cfg = _raw(
            profiles={"prod": _PROD_PROFILE},
            agent_map={},
            project_map={"my-project": "prod"},
        )
        agent = _make_agent(project="my-project")
        assert _resolve_profile_name(cfg, agent) == "prod"

    def test_project_map_skipped_when_mapped_profile_missing(self):
        """project_map maps to a profile name not in profiles → default."""
        cfg = _raw(
            profiles={"prod": _PROD_PROFILE},
            agent_map={},
            project_map={"my-project": "nonexistent"},
        )
        agent = _make_agent(project="my-project")
        assert _resolve_profile_name(cfg, agent) == "default"

    def test_project_map_accepts_project_object_with_name(self):
        """context.project may be an object with a .name attribute."""
        cfg = _raw(
            profiles={"staging": _STAGING_PROFILE},
            project_map={"langfuse-observability-project": "staging"},
        )
        agent = _make_agent(project="langfuse-observability-project")
        assert _resolve_profile_name(cfg, agent) == "staging"

    def test_project_map_no_project_returns_default(self):
        """Agent with no project set → project_map not applicable → default."""
        cfg = _raw(
            profiles={"prod": _PROD_PROFILE},
            project_map={"my-project": "prod"},
        )
        agent = _make_agent(project=None)
        assert _resolve_profile_name(cfg, agent) == "default"

    def test_agent_exception_falls_through_to_default(self):
        """If reading agent attributes raises, resolution falls back to default."""
        cfg = _raw(
            profiles={"staging": _STAGING_PROFILE},
            agent_map={"researcher": "staging"},
        )
        broken_agent = MagicMock()
        type(broken_agent).config = property(lambda s: (_ for _ in ()).throw(RuntimeError("boom")))
        # Should NOT raise — resolution catches exceptions and returns default
        result = _resolve_profile_name(cfg, broken_agent)
        assert result == "default"


# ===========================================================================
# _get_profile_config tests
# ===========================================================================

class TestGetProfileConfig:

    def test_flat_only_config_returns_flat_values(self):
        """Backward compat: flat config with no profiles → default picks flat keys."""
        cfg = _raw(
            langfuse_public_key="pk-flat",
            langfuse_secret_key="sk-flat",
            langfuse_host="http://localhost:3010/",
            langfuse_enabled=True,
            langfuse_sample_rate=1.0,
        )
        result = _get_profile_config(cfg, "default")
        assert result["langfuse_public_key"] == "pk-flat"
        assert result["langfuse_secret_key"] == "sk-flat"
        assert result["langfuse_host"] == "http://localhost:3010/"
        assert result["langfuse_enabled"] is True

    def test_named_profile_overrides_flat_keys(self):
        """Named profile values override flat config keys."""
        cfg = _raw(
            langfuse_public_key="pk-flat",
            langfuse_secret_key="sk-flat",
            langfuse_host="http://localhost:3010/",
            langfuse_enabled=True,
            profiles={
                "staging": {
                    "langfuse_public_key": "pk-staging",
                    "langfuse_secret_key": "sk-staging",
                    "langfuse_host": "https://staging.example.com",
                }
            },
        )
        result = _get_profile_config(cfg, "staging")
        assert result["langfuse_public_key"] == "pk-staging"
        assert result["langfuse_secret_key"] == "sk-staging"
        assert result["langfuse_host"] == "https://staging.example.com"
        # langfuse_enabled not set in profile → inherits flat value
        assert result["langfuse_enabled"] is True

    def test_sparse_profile_inherits_missing_fields_from_flat(self):
        """Profile only overrides some fields; others fall back to flat config."""
        cfg = _raw(
            langfuse_public_key="pk-flat",
            langfuse_secret_key="sk-flat",
            langfuse_host="http://localhost:3010/",
            langfuse_sample_rate=0.5,
            langfuse_service_name="agent-zero",
            profiles={
                "custom": {
                    "langfuse_public_key": "pk-custom",
                    "langfuse_secret_key": "sk-custom",
                    # host, sample_rate, service_name NOT overridden
                }
            },
        )
        result = _get_profile_config(cfg, "custom")
        assert result["langfuse_public_key"] == "pk-custom"
        assert result["langfuse_secret_key"] == "sk-custom"
        # Falls back to flat values
        assert result["langfuse_host"] == "http://localhost:3010/"
        assert result["langfuse_sample_rate"] == 0.5
        assert result["langfuse_service_name"] == "agent-zero"

    def test_unknown_profile_returns_flat_values(self):
        """Requesting a profile name not in profiles → flat values used as default."""
        cfg = _raw(
            langfuse_public_key="pk-flat",
            langfuse_secret_key="sk-flat",
            langfuse_host="http://localhost/",
            profiles={"staging": _STAGING_PROFILE},
        )
        result = _get_profile_config(cfg, "nonexistent")
        assert result["langfuse_public_key"] == "pk-flat"
        assert result["langfuse_secret_key"] == "sk-flat"

    def test_profile_extras_org_id_and_label(self):
        """org_id and label are profile-only fields surfaced in result."""
        cfg = _raw(
            profiles={
                "acme": {
                    "langfuse_public_key": "pk-acme",
                    "langfuse_secret_key": "sk-acme",
                    "org_id": "org-acme-123",
                    "label": "ACME Corp",
                }
            },
        )
        result = _get_profile_config(cfg, "acme")
        assert result["org_id"] == "org-acme-123"
        assert result["label"] == "ACME Corp"

    def test_false_value_in_profile_is_respected(self):
        """False is a valid override for boolean fields (e.g. langfuse_enabled=False)."""
        cfg = _raw(
            langfuse_enabled=True,  # flat: enabled
            profiles={
                "disabled-profile": {
                    "langfuse_enabled": False,  # profile: disabled
                    "langfuse_public_key": "pk-x",
                    "langfuse_secret_key": "sk-x",
                }
            },
        )
        result = _get_profile_config(cfg, "disabled-profile")
        assert result["langfuse_enabled"] is False

    def test_zero_sample_rate_in_profile_is_respected(self):
        """0 is a valid override for numeric fields (sample_rate=0 disables sampling)."""
        cfg = _raw(
            langfuse_sample_rate=1.0,
            profiles={
                "silent": {
                    "langfuse_sample_rate": 0,
                    "langfuse_public_key": "pk-x",
                    "langfuse_secret_key": "sk-x",
                }
            },
        )
        result = _get_profile_config(cfg, "silent")
        assert result["langfuse_sample_rate"] == 0


# ===========================================================================
# get_langfuse_config integration tests (via _raw_config injection)
# ===========================================================================

class TestGetLangfuseConfigIntegration:
    """Integration tests using _raw_config= injection to bypass file I/O."""

    def test_backward_compat_flat_only_config(self):
        """Flat-only config.json (pre-profiles) resolves correctly as default."""
        raw = {
            "langfuse_enabled": True,
            "langfuse_public_key": "pk-lf-default",
            "langfuse_secret_key": "sk-lf-default",
            "langfuse_host": "http://192.168.200.52:3010/",
            "langfuse_sample_rate": 1,
            "langfuse_service_name": "agent-zero",
            "langfuse_environment": "",
            "langfuse_release": "",
        }
        cfg = get_langfuse_config(_raw_config=raw)
        assert cfg["profile_name"] == "default"
        assert cfg["public_key"] == "pk-lf-default"
        assert cfg["secret_key"] == "sk-lf-default"
        assert cfg["host"] == "http://192.168.200.52:3010/"
        assert cfg["enabled"] is True

    def test_named_profile_via_agent_map(self):
        """Agent whose profile is in agent_map gets the correct credentials."""
        raw = {
            "langfuse_enabled": True,
            "langfuse_public_key": "pk-default",
            "langfuse_secret_key": "sk-default",
            "langfuse_host": "http://default.example.com/",
            "langfuse_sample_rate": 1,
            "profiles": {
                "prod": {
                    "langfuse_public_key": "pk-prod",
                    "langfuse_secret_key": "sk-prod",
                    "langfuse_host": "https://prod.example.com/",
                }
            },
            "agent_map": {"agent0": "prod"},
            "project_map": {},
        }
        agent = _make_agent(profile="agent0")
        cfg = get_langfuse_config(agent=agent, _raw_config=raw)
        assert cfg["profile_name"] == "prod"
        assert cfg["public_key"] == "pk-prod"
        assert cfg["secret_key"] == "sk-prod"
        assert cfg["host"] == "https://prod.example.com/"

    def test_named_profile_via_project_map(self):
        """Agent in a mapped project gets the project-scoped Langfuse profile."""
        raw = {
            "langfuse_enabled": True,
            "langfuse_public_key": "pk-default",
            "langfuse_secret_key": "sk-default",
            "langfuse_host": "http://default/",
            "langfuse_sample_rate": 1,
            "profiles": {
                "staging": {
                    "langfuse_public_key": "pk-staging",
                    "langfuse_secret_key": "sk-staging",
                    "langfuse_host": "https://staging/",
                }
            },
            "agent_map": {},
            "project_map": {"my-project": "staging"},
        }
        agent = _make_agent(project="my-project")
        cfg = get_langfuse_config(agent=agent, _raw_config=raw)
        assert cfg["profile_name"] == "staging"
        assert cfg["public_key"] == "pk-staging"
        assert cfg["secret_key"] == "sk-staging"

    def test_default_profile_returned_when_no_agent(self):
        """No agent → default profile always used."""
        raw = {
            "langfuse_enabled": True,
            "langfuse_public_key": "pk-default",
            "langfuse_secret_key": "sk-default",
            "langfuse_host": "http://default/",
            "langfuse_sample_rate": 1,
            "profiles": {"staging": _STAGING_PROFILE},
            "agent_map": {"researcher": "staging"},
            "project_map": {},
        }
        cfg = get_langfuse_config(agent=None, _raw_config=raw)
        assert cfg["profile_name"] == "default"
        assert cfg["public_key"] == "pk-default"

    def test_agent_map_priority_over_project_map(self):
        """agent_map match wins over project_map for the same agent."""
        raw = {
            "langfuse_enabled": True,
            "langfuse_public_key": "pk-default",
            "langfuse_secret_key": "sk-default",
            "langfuse_host": "http://default/",
            "langfuse_sample_rate": 1,
            "profiles": {
                "staging": {
                    "langfuse_public_key": "pk-staging",
                    "langfuse_secret_key": "sk-staging",
                    "langfuse_host": "https://staging/",
                },
                "prod": {
                    "langfuse_public_key": "pk-prod",
                    "langfuse_secret_key": "sk-prod",
                    "langfuse_host": "https://prod/",
                },
            },
            "agent_map": {"researcher": "staging"},
            "project_map": {"my-project": "prod"},
        }
        agent = _make_agent(profile="researcher", project="my-project")
        cfg = get_langfuse_config(agent=agent, _raw_config=raw)
        assert cfg["profile_name"] == "staging"  # agent_map wins
        assert cfg["public_key"] == "pk-staging"

    def test_profile_name_in_returned_config(self):
        """Returned config always includes 'profile_name' key."""
        raw = {"langfuse_enabled": False, "langfuse_public_key": "", "langfuse_secret_key": ""}
        cfg = get_langfuse_config(_raw_config=raw)
        assert "profile_name" in cfg

    def test_org_id_and_label_surfaced_in_config(self):
        """org_id and label from profile are present in returned config."""
        raw = {
            "langfuse_enabled": True,
            "langfuse_public_key": "pk-default",
            "langfuse_secret_key": "sk-default",
            "langfuse_host": "http://default/",
            "langfuse_sample_rate": 1,
            "profiles": {
                "acme": {
                    "langfuse_public_key": "pk-acme",
                    "langfuse_secret_key": "sk-acme",
                    "org_id": "org-123",
                    "label": "ACME Corp",
                }
            },
            "agent_map": {"researcher": "acme"},
            "project_map": {},
        }
        agent = _make_agent(profile="researcher")
        cfg = get_langfuse_config(agent=agent, _raw_config=raw)
        assert cfg["org_id"] == "org-123"
        assert cfg["label"] == "ACME Corp"

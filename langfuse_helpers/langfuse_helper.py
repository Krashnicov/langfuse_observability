import asyncio
import os
import sys
import random
import logging
import subprocess
from typing import Any
from contextvars import ContextVar

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client registry — keyed by credential fingerprint (public_key@host) so that
# project-scoped or agent-scoped configs each get their own client instance.
# A0's native get_plugin_config() handles scope resolution; no custom logic
# needed here.
# ---------------------------------------------------------------------------
_clients: dict[str, Any] = {}
_clients_initialized: set[str] = set()
_install_attempted = False

# Cached version info
_version_info: dict[str, str] | None = None


def get_version_info() -> dict[str, str]:
    """Return auto-detected version info (cached after first call).

    Keys:
      plugin_version   - from plugin.yaml (e.g. '1.0.0')
      langfuse_sdk     - from importlib.metadata (e.g. '4.0.1')
      agent_zero       - from usr/settings.json version key (e.g. 'v1.2')
    """
    global _version_info
    if _version_info is not None:
        return _version_info

    info: dict[str, str] = {}

    # Plugin version from plugin.yaml
    try:
        import yaml as _yaml
        _plugin_yaml = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plugin.yaml")
        with open(_plugin_yaml) as _f:
            _data = _yaml.safe_load(_f)
        info["plugin_version"] = str(_data.get("version", ""))
    except Exception:
        info["plugin_version"] = ""

    # Langfuse SDK version
    try:
        import importlib.metadata as _meta
        info["langfuse_sdk"] = _meta.version("langfuse")
    except Exception:
        info["langfuse_sdk"] = ""

    # Agent Zero version — resolved via helpers.git (same source as web UI),
    # fallback to subprocess git describe at /a0 root.
    try:
        # Resolve /a0 root as 4 levels up from this file:
        # langfuse_helpers/langfuse_helper.py → plugin_root → usr/plugins → usr → /a0
        _a0_root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
        )
        _helpers_path = os.path.join(_a0_root, "helpers")
        if _helpers_path not in sys.path:
            sys.path.insert(0, _helpers_path)
            sys.path.insert(0, _a0_root)
        import helpers.git as _git_helper
        _gitinfo = _git_helper.get_git_info()
        _ver = str(_gitinfo.get("version", "")).strip()
        info["agent_zero"] = _ver if _ver and _ver != "unknown" else ""
    except Exception:
        # Fallback: git describe --tags --abbrev=0 at /a0 root
        try:
            _a0_root = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
            )
            _ver = subprocess.check_output(
                ["git", "describe", "--tags", "--abbrev=0"],
                cwd=_a0_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            info["agent_zero"] = _ver if _ver else ""
        except Exception:
            info["agent_zero"] = ""

    _version_info = info
    return _version_info


# ---------------------------------------------------------------------------
# Pending generation registry — correlates LiteLLM callbacks with Langfuse spans
# ---------------------------------------------------------------------------
_pending_generations: dict[int, tuple] = {}


def _task_key() -> int:
    """Return asyncio task id as a per-coroutine correlation key."""
    try:
        task = asyncio.current_task()
        return id(task) if task else 0
    except Exception:
        return 0


def register_pending_generation(span: Any, loop_data: Any) -> None:
    """Store (span, loop_data) for retrieval by LiteLLM callback in the same task."""
    key = _task_key()
    if key:
        _pending_generations[key] = (span, loop_data)


def pop_pending_generation() -> tuple | None:
    """Retrieve and remove (span, loop_data) for the current asyncio task."""
    key = _task_key()
    return _pending_generations.pop(key, None) if key else None


# ---------------------------------------------------------------------------
# Active span context variable — propagates to child tasks automatically
# ---------------------------------------------------------------------------
_active_span_var: ContextVar = ContextVar("lf_active_span", default=None)


def set_active_span(span: Any) -> None:
    """Register the active Langfuse span for the current asyncio context.

    Called in monologue_start so child tasks (e.g. memory embedding recall)
    inherit the span via Python's context variable propagation at task creation.
    """
    _active_span_var.set(span)


def get_active_span() -> Any:
    """Return the active Langfuse span for the current asyncio context, or None."""
    return _active_span_var.get()


class LangfuseUsageCallback:
    """LiteLLM callback that writes real token usage and cost to Langfuse generation spans.

    Registered once per process via ensure_usage_callback_registered().
    Correlates with spans via asyncio task identity.
    """

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            call_type = (kwargs.get("call_type") or "").lower()
            if call_type == "embedding":
                await self._handle_embedding_event(kwargs, response_obj, start_time, end_time)
                return
            entry = pop_pending_generation()
            if not entry:
                return
            span, loop_data = entry

            usage = getattr(response_obj, "usage", None)
            if not usage:
                return

            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            cost = getattr(usage, "cost", None)

            # Cached input tokens (Anthropic / OpenAI prompt cache)
            cached_tokens = 0
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details:
                cached_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)

            # Reasoning tokens (o1 / o3 models)
            reasoning_tokens = 0
            completion_details = getattr(usage, "completion_tokens_details", None)
            if completion_details:
                reasoning_tokens = int(getattr(completion_details, "reasoning_tokens", 0) or 0)

            usage_details: dict[str, int] = {
                "input": prompt_tokens,
                "output": completion_tokens,
                "total": prompt_tokens + completion_tokens,
            }
            if cached_tokens:
                usage_details["cache_read"] = cached_tokens
            if reasoning_tokens:
                usage_details["output_reasoning"] = reasoning_tokens

            update_kwargs: dict[str, Any] = {"usage_details": usage_details}
            if cost is not None:
                try:
                    total_cost = float(cost)
                    cost_details: dict[str, float] = {"total": total_cost}
                    # Attempt per-token cost breakdown via litellm
                    try:
                        import litellm as _litellm
                        _model = kwargs.get("model") or ""
                        _in_cost, _out_cost = _litellm.cost_per_token(
                            model=_model,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                        )
                        cost_details["input"] = float(_in_cost)
                        cost_details["output"] = float(_out_cost)
                    except Exception:
                        # Fallback: split proportionally by token ratio
                        _total_tok = prompt_tokens + completion_tokens
                        if _total_tok > 0:
                            cost_details["input"] = round(total_cost * prompt_tokens / _total_tok, 10)
                            cost_details["output"] = round(total_cost * completion_tokens / _total_tok, 10)
                    update_kwargs["cost_details"] = cost_details
                except (TypeError, ValueError):
                    pass

            span.update(**update_kwargs)

            # Signal to _end extension that real usage was applied — skip approximations
            if loop_data is not None and hasattr(loop_data, "params_temporary"):
                loop_data.params_temporary["lf_real_usage_applied"] = True

        except Exception:
            pass


    async def _handle_embedding_event(self, kwargs, response_obj, start_time, end_time):
        """Emit a Langfuse observation for embedding model calls.

        Attaches to the active monologue span via ContextVar (set in monologue_start).
        This ensures embedding calls from memory recall and tool execution are
        captured under the correct trace without needing a before/after extension pair.
        """
        try:
            parent = get_active_span()
            if not parent:
                return

            model = kwargs.get("model") or "unknown"
            if "/" in model and not model.startswith("ft:"):
                model = model.split("/", 1)[1]

            input_data = kwargs.get("input") or ""
            if isinstance(input_data, list):
                input_count = len(input_data)
                # Show first input truncated as preview
                first = str(input_data[0])[:200] if input_data else ""
                input_preview = f"{input_count} text(s): {first}" + ("\u2026" if len(str(input_data[0])) > 200 else "")
            else:
                input_str = str(input_data)
                input_count = 1
                input_preview = input_str[:200] + ("\u2026" if len(input_str) > 200 else "")

            usage = getattr(response_obj, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
            cost = getattr(usage, "cost", None) if usage else None

            latency_ms: int | None = None
            try:
                latency_ms = int((end_time - start_time).total_seconds() * 1000)
            except Exception:
                pass

            metadata: dict[str, Any] = {
                "call_type": "embedding",
                "model": model,
                "input_texts": input_count,
            }
            if latency_ms is not None:
                metadata["latency_ms"] = latency_ms

            span = parent.start_observation(
                name="embedding",
                as_type="span",
                input=input_preview or None,
                metadata=metadata,
            )

            update_kwargs: dict[str, Any] = {}
            if prompt_tokens:
                update_kwargs["usage_details"] = {"input": prompt_tokens, "output": 0}
            if cost is not None:
                try:
                    update_kwargs["cost"] = float(cost)
                except (TypeError, ValueError):
                    pass
            if update_kwargs:
                span.update(**update_kwargs)
            span.end()
        except Exception:
            pass

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Sync fallback — not used for async paths but required by LiteLLM interface."""
        pass


# Guard: register callback only once per process
_callback_registered = False


def ensure_usage_callback_registered() -> None:
    """Register LangfuseUsageCallback with LiteLLM exactly once per process."""
    global _callback_registered
    if _callback_registered:
        return
    try:
        import litellm
        existing = getattr(litellm, "callbacks", []) or []
        if not any(isinstance(c, LangfuseUsageCallback) for c in existing):
            litellm.callbacks = list(existing) + [LangfuseUsageCallback()]
        _callback_registered = True
        logger.info("LangfuseUsageCallback registered with LiteLLM")
    except Exception as e:
        logger.warning(f"Could not register LangfuseUsageCallback: {e}")

def _ensure_langfuse_installed():
    """Auto-install langfuse package if not present."""
    global _install_attempted
    if _install_attempted:
        return
    _install_attempted = True
    try:
        import langfuse  # noqa: F401
    except ImportError:
        logger.info("langfuse package not found, installing...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "langfuse"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("langfuse package installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install langfuse: {e}")

    # Always attempt model compat patch after install check
    _patch_langfuse_models()


def _patch_langfuse_models() -> None:
    """Make 'organization' and 'metadata' Optional in the langfuse Project model.

    langfuse SDK >= 3.14 added required fields (organization, metadata) that
    older self-hosted servers do not return. Patching the Pydantic model to
    default them to None keeps auth_check() working without downgrading the SDK.
    Safe to call multiple times; does nothing if already patched or on failure.
    """
    try:
        import typing
        from pydantic_core import PydanticUndefined
        from langfuse.api.projects.types.project import Project

        needs_rebuild = False
        for field_name in ("organization", "metadata"):
            if field_name not in Project.model_fields:
                continue
            fi = Project.model_fields[field_name]
            if fi.default is PydanticUndefined:
                fi.default = None
                needs_rebuild = True
                # Widen annotation to Optional so Pydantic rebuilds correctly
                if field_name in Project.__annotations__:
                    orig = Project.__annotations__[field_name]
                    Project.__annotations__[field_name] = typing.Optional[orig]

        if needs_rebuild:
            Project.model_rebuild(force=True)
            logger.info(
                "Patched langfuse Project model: 'organization' and 'metadata' "
                "are now Optional for compatibility with older self-hosted servers."
            )
    except ImportError:
        pass  # langfuse not installed yet — will be called again after install
    except Exception as e:
        logger.warning(f"Could not patch langfuse Project model: {e}")


# ---------------------------------------------------------------------------
# Public configuration API
# ---------------------------------------------------------------------------

def get_langfuse_config(
    agent=None,
    _raw_config: dict | None = None,
) -> dict[str, Any]:
    """Get Langfuse configuration via A0's native plugin config scoping.

    Config is resolved by plugins.get_plugin_config() in priority order:
      1. {project_meta}/plugins/langfuse_observability/config.json  (project-scoped)
      2. agents/{profile}/plugins/langfuse_observability/config.json (agent-scoped)
      3. {plugin_dir}/config.json  (global default)

    Flat keys in the resolved config are used directly — no custom profile
    resolution logic.

    Args:
        agent:       Optional A0 Agent instance passed to get_plugin_config for
                     automatic project/agent scope resolution.
                     Pass None (or omit) to always use the global default config.
        _raw_config: Inject raw plugin config dict directly — for unit tests only.
                     Production callers should leave this as None.

    Returns:
        Dict with keys: enabled, public_key, secret_key, host, sample_rate,
        service_name, environment, release, trace_name_template.
    """
    if _raw_config is None:
        from helpers.plugins import get_plugin_config
        _raw_config = get_plugin_config("langfuse_observability", agent) or {}

    public_key = _raw_config.get("langfuse_public_key", "") or os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = _raw_config.get("langfuse_secret_key", "") or os.getenv("LANGFUSE_SECRET_KEY", "")
    host = _raw_config.get("langfuse_host", "") or os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
    enabled = _raw_config.get("langfuse_enabled", False)
    sample_rate = float(_raw_config.get("langfuse_sample_rate", 1.0))
    service_name = _raw_config.get("langfuse_service_name", "") or os.getenv("OTEL_SERVICE_NAME", "agent-zero")
    # Default environment to hostname so multiple instances are distinguishable
    environment = (
        _raw_config.get("langfuse_environment", "")
        or os.getenv("LANGFUSE_ENVIRONMENT", "")
        or os.getenv("HOSTNAME", "")
    )
    # Default release to plugin version if not explicitly set
    release = (
        _raw_config.get("langfuse_release", "")
        or os.getenv("LANGFUSE_RELEASE", "")
        or get_version_info().get("plugin_version", "")
    )
    trace_name_template = _raw_config.get("langfuse_trace_name_template", "")

    # Auto-enable if keys are set via env vars but toggle is off
    if not enabled and public_key and secret_key:
        enabled = True

    return {
        "enabled": enabled,
        "public_key": public_key,
        "secret_key": secret_key,
        "host": host,
        "sample_rate": sample_rate,
        "service_name": service_name,
        "environment": environment,
        "release": release,
        "trace_name_template": trace_name_template,
    }


def get_langfuse_client(agent=None):
    """Get or create the Langfuse client for the resolved config.

    Clients are cached by credential fingerprint (public_key + host).
    Distinct project-scoped or agent-scoped configs each get their own
    client instance when their credentials differ.

    Args:
        agent: Optional A0 Agent instance.  Passed to get_plugin_config for
               automatic project/agent scope resolution.
               Pass None for the global default config.

    Returns:
        Initialised Langfuse client, or None if disabled / not configured.
    """
    global _clients, _clients_initialized

    config = get_langfuse_config(agent)

    if not config["enabled"] or not config["public_key"] or not config["secret_key"]:
        return None

    # Cache key: public_key + host uniquely identifies the Langfuse project/org
    cache_key = f"{config['public_key']}@{config['host']}"

    if cache_key in _clients_initialized and _clients.get(cache_key) is not None:
        return _clients[cache_key]

    _ensure_langfuse_installed()

    try:
        from langfuse import Langfuse
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

        # Build TracerProvider with explicit service.name in Resource.
        # Passing tracer_provider= to Langfuse() bypasses _init_tracer_provider()
        # in the SDK, which otherwise silently returns an already-initialized
        # provider that never received our service.name (race condition with
        # any OTel TracerProvider initialized earlier in the process).
        resource_attrs: dict[str, Any] = {SERVICE_NAME: config["service_name"]}
        if config.get("environment"):
            # OTel semantic convention for deployment environment
            resource_attrs["deployment.environment"] = config["environment"]
        if config.get("release"):
            resource_attrs["service.version"] = config["release"]
        resource = Resource.create(resource_attrs)

        sample_rate = float(config.get("sample_rate", 1.0))
        tracer_provider_kwargs: dict[str, Any] = {"resource": resource}
        if sample_rate < 1.0:
            tracer_provider_kwargs["sampler"] = TraceIdRatioBased(sample_rate)
        tracer_provider = TracerProvider(**tracer_provider_kwargs)

        client_kwargs: dict[str, Any] = {
            "public_key": config["public_key"],
            "secret_key": config["secret_key"],
            "host": config["host"],
            "tracer_provider": tracer_provider,
        }
        if config.get("environment"):
            client_kwargs["environment"] = config["environment"]
        if config.get("release"):
            client_kwargs["release"] = config["release"]

        client = Langfuse(**client_kwargs)
        _clients[cache_key] = client
        _clients_initialized.add(cache_key)
        logger.info(
            f"Langfuse client initialised "
            f"(key={config['public_key'][:8]}\u2026, host={config['host']})"
        )
        return client
    except Exception as e:
        logger.warning(f"Failed to initialise Langfuse client: {e}")
        _clients.pop(cache_key, None)
        _clients_initialized.discard(cache_key)
        return None


def reset_client(profile_name: str | None = None) -> None:
    """Reset and flush all Langfuse clients.

    Args:
        profile_name: Retained for API compatibility — no longer used.
                      All cached clients are always reset regardless of this value.
                      Pass None or omit to reset all (same behaviour).
    """
    global _clients, _clients_initialized
    for client in list(_clients.values()):
        if client:
            try:
                client.flush()
            except Exception:
                pass
    _clients.clear()
    _clients_initialized.clear()


def should_sample(agent=None) -> bool:
    """Check if this interaction should be sampled based on the resolved sample_rate.

    Args:
        agent: Optional A0 Agent instance for config resolution.
    """
    config = get_langfuse_config(agent)
    rate = config.get("sample_rate", 1.0)
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    return random.random() < rate

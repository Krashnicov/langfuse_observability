import asyncio
import os
import sys
import random
import logging
import subprocess
from typing import Any
from contextvars import ContextVar

logger = logging.getLogger(__name__)

# Lazy-loaded singleton
_client = None
_client_initialized = False
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

    # Agent Zero version from settings.json
    try:
        import json as _json
        _settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "usr", "settings.json")
        _settings_path = os.path.normpath(_settings_path)
        with open(_settings_path) as _f:
            _settings = _json.load(_f)
        info["agent_zero"] = str(_settings.get("version", ""))
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
            }
            if cached_tokens:
                usage_details["cache_read"] = cached_tokens
            if reasoning_tokens:
                usage_details["output_reasoning"] = reasoning_tokens

            update_kwargs: dict[str, Any] = {"usage_details": usage_details}
            if cost is not None:
                try:
                    update_kwargs["cost"] = float(cost)
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
                input_preview = f"{input_count} text(s): {first}" + ("…" if len(str(input_data[0])) > 200 else "")
            else:
                input_str = str(input_data)
                input_count = 1
                input_preview = input_str[:200] + ("…" if len(input_str) > 200 else "")

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


def get_langfuse_config() -> dict[str, Any]:
    """Get Langfuse configuration with plugin config > env var > default precedence."""
    from helpers.plugins import get_plugin_config

    config = get_plugin_config("langfuse_observability", None) or {}
    public_key = config.get("langfuse_public_key") or os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = config.get("langfuse_secret_key") or os.getenv("LANGFUSE_SECRET_KEY", "")
    host = config.get("langfuse_host") or os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
    enabled = config.get("langfuse_enabled", False)
    sample_rate = float(config.get("langfuse_sample_rate", 1.0))
    service_name = config.get("langfuse_service_name") or os.getenv("OTEL_SERVICE_NAME", "agent-zero")
    # Default environment to hostname so multiple instances are distinguishable
    environment = config.get("langfuse_environment") or os.getenv("LANGFUSE_ENVIRONMENT", "") or os.getenv("HOSTNAME", "")
    # Default release to plugin version if not explicitly set
    release = config.get("langfuse_release") or os.getenv("LANGFUSE_RELEASE", "") or get_version_info().get("plugin_version", "")
    trace_name_template = config.get("langfuse_trace_name_template", "")

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


def get_langfuse_client():
    """Get or create the Langfuse client singleton. Returns None if disabled or not configured."""
    global _client, _client_initialized

    config = get_langfuse_config()

    if not config["enabled"] or not config["public_key"] or not config["secret_key"]:
        _client = None
        _client_initialized = False
        return None

    # Return cached client if already initialized
    if _client_initialized and _client is not None:
        return _client

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
        _client = Langfuse(**client_kwargs)
        _client_initialized = True
        logger.info("Langfuse client initialized successfully")
        return _client
    except Exception as e:
        logger.warning(f"Failed to initialize Langfuse client: {e}")
        _client = None
        _client_initialized = False
        return None


def reset_client():
    """Reset the client singleton (call when settings change)."""
    global _client, _client_initialized
    if _client:
        try:
            _client.flush()
        except Exception:
            pass
    _client = None
    _client_initialized = False


def should_sample() -> bool:
    """Check if this interaction should be sampled based on sample_rate."""
    config = get_langfuse_config()
    rate = config.get("sample_rate", 1.0)
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    return random.random() < rate

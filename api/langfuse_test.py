import os
import sys

_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PLUGIN_ROOT not in sys.path:
    sys.path.append(_PLUGIN_ROOT)

from helpers.api import ApiHandler, Request, Response
from helpers.plugins import get_plugin_config


# Matches the core PASSWORD_PLACEHOLDER pattern
_SECRET_PLACEHOLDER = "***"


def _resolve_project_info(public_key: str, secret_key: str, host: str) -> dict:
    """Resolve project name and organisation name from GET /api/public/projects.

    Returns dict with "project" and "org" string keys.
    Returns empty strings on any error: empty data array, missing key, request failure.
    """
    try:
        import httpx
        host_url = host.rstrip("/")
        resp = httpx.get(
            f"{host_url}/api/public/projects",
            auth=(public_key, secret_key),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return {"project": "", "org": ""}
        entry = data[0]
        project_name = entry.get("name", "")
        org = entry.get("organization") or {}
        org_name = org.get("name", "") if isinstance(org, dict) else ""
        return {"project": project_name, "org": org_name}
    except Exception:
        return {"project": "", "org": ""}


class LangfuseTest(ApiHandler):

    async def process(self, input: dict, request: Request) -> dict | Response:
        public_key = input.get("public_key", "")
        secret_key = input.get("secret_key", "")
        host = input.get("host", "https://us.cloud.langfuse.com")

        # If frontend sent the masked placeholder, use the real stored key
        if secret_key == _SECRET_PLACEHOLDER:
            config = get_plugin_config("langfuse_observability", None) or {}
            secret_key = config.get("langfuse_secret_key", "")

        if not public_key or not secret_key:
            return {"success": False, "error": "Public key and secret key are required"}

        try:
            from langfuse_helpers.langfuse_helper import _ensure_langfuse_installed
            _ensure_langfuse_installed()
            from langfuse import Langfuse

            client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
            result = client.auth_check()
            client.flush()
            if not result:
                return {"success": False, "error": "Authentication failed"}
        except ImportError:
            return {"success": False, "error": "langfuse package not installed. Could not auto-install."}
        except Exception as e:
            return {"success": False, "error": str(e)}

        # Auth succeeded - resolve project + org from /api/public/projects
        project_info = _resolve_project_info(public_key, secret_key, host)
        return {"success": True, **project_info}

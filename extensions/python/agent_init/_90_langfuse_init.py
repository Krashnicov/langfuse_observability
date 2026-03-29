import os
import sys
import importlib

_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PLUGIN_ROOT not in sys.path:
    sys.path.append(_PLUGIN_ROOT)

# Force-reload langfuse_helper from disk — guards against stale sys.modules in
# long-running processes with symlinked plugins. agent_init is the earliest
# extension point so this reload covers all subsequent extension imports too.
_lf_mod_name = "langfuse_helpers.langfuse_helper"
if _lf_mod_name in sys.modules:
    try:
        importlib.reload(sys.modules[_lf_mod_name])
    except Exception:
        pass

from helpers.extension import Extension
from langfuse_helpers.langfuse_helper import get_langfuse_client, ensure_usage_callback_registered


class LangfuseInit(Extension):

    def execute(self, **kwargs):
        get_langfuse_client()
        ensure_usage_callback_registered()

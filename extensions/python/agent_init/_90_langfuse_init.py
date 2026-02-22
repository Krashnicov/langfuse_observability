import os
import sys

_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)

from python.helpers.extension import Extension
from helpers.langfuse_helper import get_langfuse_client


class LangfuseInit(Extension):

    async def execute(self, **kwargs):
        get_langfuse_client()

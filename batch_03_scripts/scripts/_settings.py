import importlib.util
import os
import sys


_SCRIPTS_DIR = os.path.dirname(__file__)
_CONFIG_PATH = os.path.join(_SCRIPTS_DIR, "config.py")
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_SPEC = importlib.util.spec_from_file_location("_trajectory_transformer_scripts_config", _CONFIG_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

Settings = _MODULE.Settings

"""Settings lookup: .env first, then os.environ, then the Windows user registry.

The registry step matters on Windows: values set with `setx` only reach processes
started after the setx, so an editor or shell opened earlier won't have them.
"""
import os
from pathlib import Path

from dotenv import dotenv_values

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PRICE_USD = 29
LLM_COST_PER_MILLION_TOKENS = 0.10  # blended in+out estimate for Luminal

_dotenv = {k: v for k, v in dotenv_values(BASE_DIR / ".env").items() if v}


def _from_registry(name):
    try:
        import winreg
    except ImportError:
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, kind = winreg.QueryValueEx(key, name)
    except OSError:
        return None
    if kind == winreg.REG_EXPAND_SZ:
        value = winreg.ExpandEnvironmentStrings(value)
    return value


def get(name, default=None):
    """First non-empty value from .env, os.environ, then HKCU\\Environment."""
    for value in (_dotenv.get(name), os.environ.get(name), _from_registry(name)):
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def require(name):
    value = get(name)
    if not value:
        raise RuntimeError(f"{name} is not set (.env, environment, or setx)")
    return value

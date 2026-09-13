"""Overlay package that falls back to the original ToolSafe agent modules."""

from pathlib import Path
import sys

_ORIGINAL_AGENT = Path(__file__).parents[4] / "ToolSafe" / "src" / "agent"
if _ORIGINAL_AGENT.is_dir():
    __path__.append(str(_ORIGINAL_AGENT))
    _ORIGINAL_SRC = str(_ORIGINAL_AGENT.parent)
    if _ORIGINAL_SRC not in sys.path:
        sys.path.append(_ORIGINAL_SRC)

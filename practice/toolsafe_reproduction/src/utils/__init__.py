"""Overlay package that resolves utility modules from the original ToolSafe source."""

from pathlib import Path

_ORIGINAL_UTILS = Path(__file__).parents[4] / "ToolSafe" / "src" / "utils"
if _ORIGINAL_UTILS.is_dir():
    __path__.append(str(_ORIGINAL_UTILS))

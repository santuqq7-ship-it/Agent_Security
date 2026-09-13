"""Overlay package that reuses ToolSafe's unchanged evaluator modules."""

from pathlib import Path

_ORIGINAL_EVALUATOR = Path(__file__).parents[4] / "ToolSafe" / "src" / "guardian_evaluator"
if _ORIGINAL_EVALUATOR.is_dir():
    __path__.append(str(_ORIGINAL_EVALUATOR))

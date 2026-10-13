"""Source-oriented CasU terrain and fluid state testbed.

Keep package discovery lightweight so the file-only research workbench can run
without importing NumPy/Isaac dependencies. Public state exports remain lazy.
"""

__all__ = ["BlockDamage", "CasUGridState", "DecisionStream"]


def __getattr__(name):
    if name in __all__:
        from . import state
        return getattr(state, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

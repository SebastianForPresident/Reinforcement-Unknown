"""Simulator compute-backend selection.

Two named backends implement the same scientific logic:

``reference-python``  the readable NumPy/Python implementation (always available; it is the
                      correctness oracle and the fallback).
``numba``             compiled CPU kernels in :mod:`casu_grid.kernels` for the fluid sweep and
                      the tile part of the ground queries. Requires ``numba`` (bundled with
                      Isaac Sim 4.5's Python as 0.59.1). Kernels are compiled once and cached
                      under ``casu_grid/__pycache__``; the first run in a fresh checkout pays a
                      few seconds of compilation.

Selection: ``CASU_BACKEND=auto|numba|reference-python`` (default ``auto``: numba when it
imports, otherwise reference). ``numba`` fails loudly if unavailable. The active name is
recorded in benchmark results and training-run metadata via :func:`describe`.

Both backends are tested for byte-identical discrete state (fluid sweep vs the literal
oracle ``simulation_step_reference``) and identical query results
(``tests/test_kernels.py``).
"""
from __future__ import annotations

import os

REFERENCE = "reference-python"
NUMBA = "numba"
_VALID = ("auto", NUMBA, REFERENCE)
_kernels = None
_loaded = False
_error: str | None = None


def requested() -> str:
    value = os.environ.get("CASU_BACKEND", "auto").strip().lower()
    if value not in _VALID:
        raise ValueError(f"CASU_BACKEND must be one of {_VALID}, got {value!r}")
    return value


def _load():
    global _kernels, _loaded, _error
    if _loaded:
        return _kernels
    _loaded = True
    try:
        from . import kernels
        _kernels = kernels
    except Exception as error:   # numba missing or kernel compilation failed
        _kernels, _error = None, f"{type(error).__name__}: {error}"
    return _kernels


def kernels_for(name: str | None = None):
    """The kernel module for ``name`` (default: the environment selection) or ``None`` for reference."""
    name = (name or requested()).lower()
    if name == REFERENCE:
        return None
    module = _load()
    if module is None and name == NUMBA:
        raise RuntimeError(f"CASU_BACKEND=numba requested but unavailable ({_error})")
    return module


def active_name(name: str | None = None) -> str:
    return NUMBA if kernels_for(name) is not None else REFERENCE


def describe() -> dict:
    """Machine-readable backend identity for run metadata and benchmark results."""
    module = kernels_for()
    info = {"requested": requested(), "active": NUMBA if module is not None else REFERENCE}
    if module is not None:
        import numba
        info["numba_version"] = numba.__version__
        info["kernel_version"] = module.KERNEL_VERSION
    elif _error:
        info["fallback_reason"] = _error
    return info

"""Send researcher callbacks (action decoder, observation encoder, reward function) to worker processes.

The multi-process environment executes callbacks inside its workers, so they must cross a process
boundary. Plain ``pickle`` only sends functions *by reference* (module + qualified name), which fails for
closures, for functions of a config file loaded by path, and for the wrappers the workbench puts around
user callbacks. This module is a small standard-library serializer (no new dependency) that sends such
functions *by value*: code object (``marshal``), the global names the code references, defaults and
closure cell contents, recursively. Importable top-level functions and classes still travel by reference.

Limits (reported as clear errors before any worker starts): callbacks must only capture picklable
values; classes defined in non-importable modules cannot be sent; self-recursive closures are not
supported; each worker gets its own copy of any captured state, so callbacks must not rely on shared
mutable state. Use ``num_workers=1`` for anything else.
"""
from __future__ import annotations

import builtins
import io
import marshal
import pickle
import sys
import types


def _importable(fn) -> bool:
    """True if ``fn`` can be pickled by reference (found again under module + qualname)."""
    module = sys.modules.get(getattr(fn, "__module__", None))
    obj = module
    try:
        for part in fn.__qualname__.split("."):
            if part == "<locals>":
                return False
            obj = getattr(obj, part)
    except AttributeError:
        return False
    return obj is fn


def _code_names(code, found):
    found.update(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            _code_names(const, found)
    return found


def _rebuild_function(code_bytes, globals_subset, name, defaults, kwdefaults, cells, module, qualname, doc, attrs):
    namespace = {"__builtins__": builtins, "__name__": module}
    namespace.update(globals_subset)
    closure = tuple(types.CellType(c) if c is not _EMPTY else types.CellType() for c in cells)
    fn = types.FunctionType(marshal.loads(code_bytes), namespace, name, defaults, closure or None)
    fn.__kwdefaults__, fn.__module__, fn.__qualname__, fn.__doc__ = kwdefaults, module, qualname, doc
    fn.__dict__.update(attrs)
    return fn


def importlib_module(name):
    import importlib
    return importlib.import_module(name)


class _Empty:
    def __reduce__(self):
        return "_EMPTY"


_EMPTY = _Empty()


class _Pickler(pickle.Pickler):
    def reducer_override(self, obj):
        if isinstance(obj, types.ModuleType):
            return importlib_module, (obj.__name__,)
        if isinstance(obj, types.FunctionType) and not _importable(obj):
            names = _code_names(obj.__code__, set())
            subset = {}
            for key in names:
                if key in obj.__globals__:
                    subset[key] = obj.__globals__[key]
            cells = []
            for cell in obj.__closure__ or ():
                try:
                    cells.append(cell.cell_contents)
                except ValueError:       # empty cell (e.g. a closure over a not yet assigned name)
                    cells.append(_EMPTY)
            return _rebuild_function, (marshal.dumps(obj.__code__), subset, obj.__name__, obj.__defaults__,
                                       obj.__kwdefaults__, tuple(cells), obj.__module__, obj.__qualname__,
                                       obj.__doc__, dict(obj.__dict__))
        return NotImplemented


def dumps(obj) -> bytes:
    buffer = io.BytesIO()
    try:
        _Pickler(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump(obj)
    except (pickle.PicklingError, TypeError, AttributeError, RecursionError, ValueError) as error:
        raise ValueError(
            f"callbacks cannot be sent to worker processes ({type(error).__name__}: {error}); they may only "
            "capture picklable values (top-level functions/classes of importable modules, numbers, strings, "
            "arrays, dataclasses); use num_workers=1 otherwise") from error
    return buffer.getvalue()


def loads(data: bytes):
    return pickle.loads(data)

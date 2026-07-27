import numpy as np


def _build_node(dtype):
    """Precompute the fixed field layout for one structured dtype."""
    if dtype.names is None:
        return ("leaf",)

    fields = []
    for name in dtype.names:
        field_dtype = dtype.fields[name][0]
        base_dtype = field_dtype.subdtype[0] if field_dtype.subdtype else field_dtype
        fields.append((name, _build_node(base_dtype)))

    return ("structured", tuple(fields))


def build_plan(dtype):
    """Build the flattening layout once; the observation schema is fixed."""
    return _build_node(dtype)


def _flatten_node(value, node):
    if node[0] == "leaf":
        values = np.asarray(value, dtype=np.float32)
        return values.reshape(value.shape + (1,))

    parts = []
    parent_shape = value.shape
    for name, child_node in node[1]:
        field = value[name]
        part = _flatten_node(field, child_node)
        # Collapse fixed-size nested arrays into the final per-record axis.
        parts.append(part.reshape(parent_shape + (-1,)))

    return np.concatenate(parts, axis=-1)


def flatten(observation, plan):
    """Flatten one observation using a previously generated layout."""
    return _flatten_node(observation, plan).reshape(-1)

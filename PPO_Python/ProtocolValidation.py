"""Fail-fast runtime invariants for the CB1 Unity/Python contract."""

import numpy as np

import Types


def wire_action(action):
    values = [int(value) for value in action]
    if len(values) != 7:
        raise AssertionError(f"Expected seven policy actions, got {len(values)}")
    return np.asarray(
        [
            values[0] - 1,
            values[1],
            values[2] - 1,
            values[3],
            values[4] - 4,
            values[5] - 5,
            values[6],
        ],
        dtype=np.int8,
    )


def observed_previous_action(obs):
    previous = obs["PreviousAction"]
    return np.asarray(
        [previous[name] for name in Types.POLICY_ACTION_DTYPE.names],
        dtype=np.int8,
    )


def _validate_common(obs, processed):
    version = int(obs["ProtocolVersion"])
    if version != Types.PROTOCOL_VERSION:
        raise RuntimeError(
            f"Protocol mismatch: Unity sent V{version}, Python expects "
            f"V{Types.PROTOCOL_VERSION}"
        )

    width = int(obs["WorldDimensions"]["X"])
    height = int(obs["WorldDimensions"]["Y"])
    x = int(obs["PlayerTilePosition"]["X"])
    y = int(obs["PlayerTilePosition"]["Y"])
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid world dimensions: {width}x{height}")
    if not (0 <= x < width and 0 <= y < height):
        raise RuntimeError(
            f"Absolute player tile {(x, y)} is outside world {width}x{height}"
        )

    for name, encoded in processed.items():
        array = np.asarray(encoded)
        if not np.isfinite(array).all():
            raise RuntimeError(f"Non-finite values in normalized {name} input")
        if array.min(initial=0.0) < -1.00001 or array.max(initial=0.0) > 1.00001:
            raise RuntimeError(
                f"Normalized {name} input escaped [-1, 1]: "
                f"min={array.min()}, max={array.max()}"
            )

    # These values are repaired immediately before every policy observation.
    godmode_minimums = {
        "BrainHealth": 99.0,
        "Consciousness": 99.0,
        "Stamina": 99.0,
        "Energy": 99.0,
        "BloodOxygen": 99.0,
        "ClawHealth": 99.0,
    }
    for name, minimum in godmode_minimums.items():
        value = float(obs[name])
        if value < minimum:
            raise RuntimeError(
                f"Functional godmode invariant failed: {name}={value:.3f}"
            )
    if bool(obs["PlayerDead"]):
        raise RuntimeError("Functional godmode invariant failed: player is dead")

    if int(obs["LayerTimeRemaining"]) < 900_000_000:
        raise RuntimeError(
            "Completion protocol invariant failed: radline deadline is active"
        )


def validate_reset_observation(obs, processed):
    _validate_common(obs, processed)
    ticks = int(obs["MacrostepPhysicsTicks"])
    delta = float(obs["SimulationDeltaTime"])
    if ticks != 0 or abs(delta) > 1e-6:
        raise RuntimeError(
            f"Reset observation must represent zero ticks, got ticks={ticks}, dt={delta}"
        )


def validate_step_observation(obs, processed, action):
    _validate_common(obs, processed)
    ticks = int(obs["MacrostepPhysicsTicks"])
    if ticks != Types.POLICY_PHYSICS_TICKS:
        raise RuntimeError(
            f"Macrostep represented {ticks} physics ticks; expected "
            f"{Types.POLICY_PHYSICS_TICKS}"
        )

    expected_delta = Types.POLICY_PHYSICS_TICKS * 0.02
    actual_delta = float(obs["SimulationDeltaTime"])
    if not np.isclose(actual_delta, expected_delta, rtol=0.0, atol=1e-4):
        raise RuntimeError(
            f"Macrostep simulation delta was {actual_delta:.6f}; expected "
            f"{expected_delta:.6f}"
        )

    expected_action = wire_action(action)
    actual_action = observed_previous_action(obs)
    if not np.array_equal(actual_action, expected_action):
        raise RuntimeError(
            "Previous-action feedback mismatch: "
            f"expected={expected_action.tolist()}, actual={actual_action.tolist()}"
        )

    return {
        "protocol_version": Types.PROTOCOL_VERSION,
        "macrostep_physics_ticks": ticks,
        "simulation_delta_time": actual_delta,
        "normalized_general_min": float(processed["general"].min()),
        "normalized_general_max": float(processed["general"].max()),
        "normalized_spatial_min": float(processed["spatial"].min()),
        "normalized_spatial_max": float(processed["spatial"].max()),
    }

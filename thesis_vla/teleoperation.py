"""Pure mapping helpers for bounded human control of the existing OSC policy space."""
from __future__ import annotations

import math

TRANSLATION_LIMIT_M = 0.05
ROTATION_LIMIT_RAD = 0.5
MIN_POSITION_STEP_M = 0.001
MAX_POSITION_STEP_M = 0.02
MIN_ROTATION_STEP_RAD = math.radians(1.0)
MAX_ROTATION_STEP_RAD = math.radians(8.0)

KEY_AXES = {
    "w": (0, 1), "s": (0, -1), "d": (1, 1), "a": (1, -1),
    "r": (2, 1), "f": (2, -1), "i": (3, 1), "k": (3, -1),
    "l": (4, 1), "j": (4, -1), "u": (5, 1), "m": (5, -1),
}


def map_key(key: str, *, position_step_m: float = 0.005,
            rotation_step_rad: float = math.radians(3.0), gripper: str | None = None):
    """Map one key to one normalized 7D OSC action.

    The configured relative OSC_POSE controller maps translation [-1,1] to
    +/-0.05 m and rotation-vector components to +/-0.5 rad. Translation is in
    the world frame. Panda gripper convention: -1 open, +1 close.
    """
    action = [0.0] * 7
    if key:
        axis = KEY_AXES.get(key.lower())
        if axis is None:
            raise ValueError(f"unsupported teleoperation key: {key!r}")
        index, sign = axis
        if index < 3:
            if not MIN_POSITION_STEP_M <= position_step_m <= MAX_POSITION_STEP_M:
                raise ValueError("position step must be between 1 and 20 mm")
            action[index] = sign * position_step_m / TRANSLATION_LIMIT_M
        else:
            if not MIN_ROTATION_STEP_RAD <= rotation_step_rad <= MAX_ROTATION_STEP_RAD:
                raise ValueError("rotation step must be between 1 and 8 degrees")
            action[index] = sign * rotation_step_rad / ROTATION_LIMIT_RAD
    if gripper is not None:
        if gripper not in {"open", "close"}:
            raise ValueError("gripper must be open or close")
        action[6] = -1.0 if gripper == "open" else 1.0
    if not key and gripper is None:
        raise ValueError("a movement key or gripper command is required")
    return tuple(action)

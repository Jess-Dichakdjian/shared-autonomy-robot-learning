"""Bounded keyboard-to-LIBERO OSC action mapping (no simulator dependency)."""
from __future__ import annotations

import math

TRANSLATION_LIMIT_M = 0.05
ROTATION_LIMIT_RAD = 0.5
MIN_POSITION_STEP_M = 0.001
MAX_POSITION_STEP_M = 0.03
MIN_ROTATION_STEP_RAD = math.radians(1.0)
MAX_ROTATION_STEP_RAD = math.radians(8.0)
MAX_GRIPPER_REPEAT = 5

# Deltas are in the world frame. Keys form opposing pairs for x/y/z, then wrist.
KEY_AXES = {
    "w": (0, 1), "s": (0, -1),
    "d": (1, 1), "a": (1, -1),
    "r": (2, 1), "f": (2, -1),
    "i": (3, 1), "k": (3, -1),
    "l": (4, 1), "j": (4, -1),
    "u": (5, 1), "m": (5, -1),
}


def map_key(key: str, *, position_step_m: float = 0.005,
            rotation_step_rad: float = math.radians(3.0), gripper: str | None = None):
    """Return one finite 7D OSC command; x/y/z deltas then axis-angle then gripper.

    The OSC controller scales normalized translation by 0.05 m and rotation by
    0.5 rad. Panda gripper convention is -1=open, +1=close. A movement key
    generates exactly one action; the browser/server does not queue key repeats.
    """
    if not key and gripper is None:
        raise ValueError("provide a movement key or gripper command")
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
    return tuple(action)


def gripper_repeat_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_GRIPPER_REPEAT:
        raise ValueError(f"gripper repeat count must be an integer from 1 to {MAX_GRIPPER_REPEAT}")
    return value


def camera_plane_basis(camera_rotation):
    """Project MuJoCo camera image axes into the world tabletop plane.

    ``camera_rotation`` is the camera-local-to-world 3x3 rotation from the
    live MuJoCo camera extrinsics. Local +X is image-right and local +Y is
    image-up. The returned directions are display/control metadata only; raw
    observations are never transformed.
    """
    import numpy as np

    rotation = np.asarray(camera_rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("camera rotation must be a finite 3x3 matrix")

    def horizontal_unit(vector, label):
        projected = np.asarray(vector, dtype=float).copy()
        projected[2] = 0.0
        norm = float(np.linalg.norm(projected))
        if not math.isfinite(norm) or norm < 1e-8:
            raise ValueError(f"camera {label} axis has no stable tabletop projection")
        return projected / norm

    right = horizontal_unit(rotation[:, 0], "image-right")
    screen_up = horizontal_unit(rotation[:, 1], "image-up")
    return {
        "right_world": right,
        "forward_world": screen_up,
        "screen_up_world": screen_up,
    }


def world_delta_for_direction(direction, distance_m, *, reference="world",
                              scene_basis=None, wrist_basis=None):
    """Map an operator's direction into bounded world-frame metres.

    World convention is +X forward, +Y right, +Z up. Scene/wrist forward and
    right are derived from each live camera's extrinsics. ``distance_m`` is a
    requested Cartesian displacement, not a guarantee of measured motion.
    """
    import numpy as np

    if direction not in {"forward", "backward", "left", "right", "up", "down"}:
        raise ValueError("unsupported direction")
    try:
        distance = float(distance_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("distance must be a finite positive number") from exc
    if not math.isfinite(distance) or not 0.0 < distance <= TRANSLATION_LIMIT_M:
        raise ValueError("distance must be greater than zero and no more than 0.05 m")
    if direction in {"up", "down"}:
        axis = np.asarray([0.0, 0.0, 1.0])
        sign = 1.0 if direction == "up" else -1.0
    else:
        if reference == "world":
            basis = {"right_world": np.asarray([0.0, 1.0, 0.0]),
                     "forward_world": np.asarray([1.0, 0.0, 0.0])}
        elif reference == "scene":
            basis = scene_basis
        elif reference == "wrist":
            basis = wrist_basis
        else:
            raise ValueError("reference must be world, scene, or wrist")
        if not isinstance(basis, dict) or not {"right_world", "forward_world"}.issubset(basis):
            raise ValueError(f"{reference} camera basis is unavailable")
        axis_name = "forward_world" if direction in {"forward", "backward"} else "right_world"
        axis = np.asarray(basis[axis_name], dtype=float)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            raise ValueError("camera basis must contain finite 3D vectors")
        axis = axis.copy()
        axis[2] = 0.0
        norm = float(np.linalg.norm(axis))
        if norm < 1e-8:
            raise ValueError("camera basis direction has no tabletop projection")
        axis /= norm
        sign = -1.0 if direction in {"backward", "left"} else 1.0
    result = sign * distance * axis
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("computed world displacement is invalid")
    return result


def translation_action(world_delta, *, gripper=0.0):
    """Convert a world-frame metre delta and gripper command to normalized OSC."""
    import numpy as np

    delta = np.asarray(world_delta, dtype=float)
    if delta.shape != (3,) or not np.all(np.isfinite(delta)):
        raise ValueError("world_delta must be a finite three-vector")
    if np.any(np.abs(delta) > TRANSLATION_LIMIT_M + 1e-9):
        raise ValueError("world delta exceeds OSC translation limit of 0.05 m per axis")
    if isinstance(gripper, bool) or gripper not in {-1, 0, 1, -1.0, 0.0, 1.0}:
        raise ValueError("gripper must be -1 (open), 0 (hold), or +1 (close)")
    action = np.zeros(7, dtype=float)
    action[:3] = delta / TRANSLATION_LIMIT_M
    action[6] = float(gripper)
    return action


def rotation_action(axis, sign, angle_rad):
    """Map one bounded world axis-angle component to normalized OSC."""
    import numpy as np

    if isinstance(axis, bool) or not isinstance(axis, int) or axis not in (0, 1, 2):
        raise ValueError("rotation axis must be 0, 1, or 2")
    if isinstance(sign, bool) or sign not in (-1, 1, -1.0, 1.0):
        raise ValueError("rotation sign must be -1 or +1")
    angle = float(angle_rad)
    if not math.isfinite(angle) or not 0.0 < angle <= ROTATION_LIMIT_RAD:
        raise ValueError("rotation angle must be in (0, 0.5] radians")
    action = np.zeros(7, dtype=float)
    action[3 + axis] = float(sign) * angle / ROTATION_LIMIT_RAD
    return action

"""Model-independent policy contracts and safe joint-position action handling.

Policy backends (scripted controllers, GR00T, openpi, or another model) should
adapt their inputs and outputs to these small dataclasses. This module does not
import any particular model package.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Protocol

import numpy as np


@dataclass(frozen=True)
class PolicyObservation:
    """One decision-time observation supplied to any policy implementation."""

    instruction: str
    seed: int | None
    sim_time_s: float
    robot_state: Mapping[str, Any]
    static_rgb: np.ndarray | None = None
    wrist_rgb: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Return JSON-friendly state and metadata; images are saved separately."""
        return {
            "instruction": self.instruction,
            "seed": self.seed,
            "sim_time_s": float(self.sim_time_s),
            "robot_state": _json_value(self.robot_state),
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True)
class RobotAction:
    """Canonical action: seven arm joint-position targets and finger position.

    The finger command is the position of each Panda finger in metres, in the
    MuJoCo model's 0 to 0.04 m range. Policy-specific vectors or action chunks
    must be converted to this form by their adapter before execution.
    """

    arm_joint_position_rad: np.ndarray
    gripper_finger_position_m: float

    def __post_init__(self) -> None:
        joints = np.asarray(self.arm_joint_position_rad, dtype=np.float64).reshape(-1)
        if joints.shape != (7,):
            raise ValueError(f"Expected seven arm joint targets, got shape {joints.shape}")
        object.__setattr__(self, "arm_joint_position_rad", joints.copy())
        object.__setattr__(self, "gripper_finger_position_m", float(self.gripper_finger_position_m))

    def to_record(self) -> dict[str, Any]:
        return {
            "type": "joint_position_targets",
            "arm_joint_position_rad": self.arm_joint_position_rad.tolist(),
            "gripper_finger_position_m": self.gripper_finger_position_m,
        }


class Policy(Protocol):
    """Backend-neutral inference contract; model adapters implement predict."""

    def predict(self, observation: PolicyObservation) -> RobotAction:
        """Produce one canonical robot action for an observation."""


class ScriptedPolicy:
    """Wrap a deterministic controller in the same interface as learned ones."""

    def __init__(self, controller: Callable[[PolicyObservation], RobotAction]):
        self._controller = controller

    def predict(self, observation: PolicyObservation) -> RobotAction:
        return self._controller(observation)


class VLAAdapter:
    """Placeholder VLA adapter with a safe no-op or scripted fallback.

    The adapter accepts the shared observation and an optional explicit
    instruction. A real GR00T/openpi backend is deliberately not loaded here.
    Later, replace the marked fallback section in :meth:`predict` with model
    input preprocessing, the selected model's inference call, and conversion
    of its output to the canonical :class:`RobotAction` format.

    Returned actions still need to pass through :class:`ActionLimits` before
    they are sent to the simulator, just like actions from other policies.
    """

    def __init__(self, fallback_policy: Policy | None = None):
        """Use ``fallback_policy`` when supplied; otherwise hold current pose."""
        self._fallback_policy = fallback_policy

    def predict(
        self,
        observation: PolicyObservation,
        instruction: str | None = None,
    ) -> RobotAction:
        """Accept observation/instruction and return a canonical fallback action."""
        effective_instruction = observation.instruction if instruction is None else instruction
        policy_observation = (
            observation if effective_instruction == observation.instruction
            else replace(observation, instruction=effective_instruction)
        )

        # FUTURE MODEL HOOK: preprocess policy_observation (including camera
        # images and effective_instruction), call the selected GR00T/openpi
        # inference API, then convert its result to RobotAction. No model call
        # or dependency is present in this stub.
        if self._fallback_policy is not None:
            return self._fallback_policy.predict(policy_observation)

        # Safe no-op: hold the observed seven-joint pose and current finger
        # opening. The control loop must still apply ActionLimits to this action.
        try:
            joint_position = np.asarray(
                policy_observation.robot_state["joint_position_rad"], dtype=np.float64
            ).reshape(-1)
        except KeyError as exc:
            raise ValueError(
                "VLAAdapter no-op requires robot_state['joint_position_rad']"
            ) from exc
        if joint_position.shape != (7,) or not np.all(np.isfinite(joint_position)):
            raise ValueError("VLAAdapter no-op requires seven finite joint positions")
        gripper_position = float(
            policy_observation.robot_state.get("gripper_finger_position_m", 0.04)
        )
        if not np.isfinite(gripper_position):
            raise ValueError("VLAAdapter no-op requires a finite gripper position")
        return RobotAction(joint_position, gripper_position)


@dataclass(frozen=True)
class ActionLimits:
    """Absolute and per-decision safety limits for the canonical action."""

    joint_lower_rad: np.ndarray
    joint_upper_rad: np.ndarray
    max_joint_speed_rad_s: np.ndarray
    gripper_lower_m: float = 0.0
    gripper_upper_m: float = 0.04
    max_gripper_speed_m_s: float = 0.08

    def __post_init__(self) -> None:
        for name in ("joint_lower_rad", "joint_upper_rad", "max_joint_speed_rad_s"):
            values = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if values.shape != (7,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain seven finite values")
            object.__setattr__(self, name, values.copy())
        if np.any(self.joint_lower_rad >= self.joint_upper_rad):
            raise ValueError("Every joint lower limit must be below its upper limit")
        if np.any(self.max_joint_speed_rad_s <= 0):
            raise ValueError("Joint step limits must be positive")
        if self.gripper_lower_m >= self.gripper_upper_m or self.max_gripper_speed_m_s <= 0:
            raise ValueError("Invalid gripper bounds")

    @classmethod
    def from_mujoco(
        cls,
        model: Any,
        *,
        max_joint_speed_rad_s: float = 0.35,
        max_gripper_speed_m_s: float = 0.08,
    ) -> "ActionLimits":
        """Build absolute bounds from the first seven arm actuator ctrlranges."""
        if model.nu < 8:
            raise ValueError(f"Expected at least eight Panda actuators, got {model.nu}")
        ctrlrange = np.asarray(model.actuator_ctrlrange[:7], dtype=np.float64)
        if not np.all(np.isfinite(ctrlrange)) or np.any(ctrlrange[:, 0] >= ctrlrange[:, 1]):
            raise ValueError("Panda arm actuators must have finite, increasing ctrlranges")
        if max_joint_speed_rad_s <= 0 or max_gripper_speed_m_s <= 0:
            raise ValueError("Maximum action speeds must be positive")
        return cls(
            joint_lower_rad=ctrlrange[:, 0],
            joint_upper_rad=ctrlrange[:, 1],
            max_joint_speed_rad_s=np.full(7, max_joint_speed_rad_s),
            max_gripper_speed_m_s=max_gripper_speed_m_s,
        )

    def clamp(
        self,
        action: RobotAction,
        previous_joint_target_rad: np.ndarray,
        dt_s: float,
        previous_gripper_target_m: float = 0.04,
    ) -> RobotAction:
        """Clamp absolute bounds and target changes from the prior command."""
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be a positive finite duration")
        previous = np.asarray(previous_joint_target_rad, dtype=np.float64).reshape(-1)
        if previous.shape != (7,) or not np.all(np.isfinite(previous)):
            raise ValueError("Previous arm joint target must contain seven finite values")
        if not np.isfinite(previous_gripper_target_m):
            raise ValueError("Previous gripper target must be finite")
        requested_joints = action.arm_joint_position_rad
        if not np.all(np.isfinite(requested_joints)) or not np.isfinite(action.gripper_finger_position_m):
            raise ValueError("Policy action contains NaN or infinite values")

        per_step_joint_delta = self.max_joint_speed_rad_s * dt_s
        slew_limited = np.clip(requested_joints, previous - per_step_joint_delta,
                               previous + per_step_joint_delta)
        safe_joints = np.clip(slew_limited, self.joint_lower_rad, self.joint_upper_rad)

        gripper_delta = self.max_gripper_speed_m_s * dt_s
        slew_limited_gripper = np.clip(
            action.gripper_finger_position_m,
            previous_gripper_target_m - gripper_delta,
            previous_gripper_target_m + gripper_delta,
        )
        safe_gripper = float(np.clip(slew_limited_gripper, self.gripper_lower_m, self.gripper_upper_m))
        return RobotAction(safe_joints, safe_gripper)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value

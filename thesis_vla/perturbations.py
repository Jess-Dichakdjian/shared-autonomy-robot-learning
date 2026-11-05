"""Physical, reproducible XY scene perturbations for LIBERO-Spatial task 1."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
TASK_ID = 1
OBJECT_JOINTS = {
    "bowl": ("akita_black_bowl_1", "akita_black_bowl_1_joint0"),
    "plate": ("plate_1", "plate_1_joint0"),
}
SETTLE_STEPS = 10
TABLE_MARGIN_M = 0.04
CONTACT_PENETRATION_TOLERANCE_M = 1e-3


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_scenario(spec: dict) -> dict:
    if not isinstance(spec, dict):
        raise ValueError("scene perturbation must be a mapping")
    if spec.get("task_id") != TASK_ID:
        raise ValueError("controlled perturbations currently support LIBERO-Spatial task 1 only")
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported perturbation schema version")
    for key in ("bowl_translation_xy_m", "plate_translation_xy_m"):
        value = spec.get(key, [0.0, 0.0])
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (2,) or not np.isfinite(array).all():
            raise ValueError(f"{key} must contain two finite XY offsets in metres")
        if np.any(np.abs(array) > 0.08 + 1e-12):
            raise ValueError(f"{key} exceeds the pre-registered 8 cm range")
    if spec.get("split") not in {"discovery", "validation", "held_out"}:
        raise ValueError("split must be discovery, validation, or held_out")
    if not spec.get("scenario_id"):
        raise ValueError("scenario_id is required")
    return spec


def _task_env(lerobot_env):
    wrapped = lerobot_env._env
    task_env = wrapped.env
    while not hasattr(task_env, "objects_dict") and hasattr(task_env, "env"):
        task_env = task_env.env
    if not hasattr(task_env, "objects_dict") or not hasattr(task_env, "object_states_dict"):
        raise RuntimeError("Unrecognized LIBERO task environment; refusing to guess object layout")
    return wrapped, task_env


def _free_joint(sim, object_model, joint_name):
    if joint_name not in object_model.joints:
        raise RuntimeError(f"Expected joint {joint_name!r} is absent from the identified object")
    joint_id = sim.model.joint_name2id(joint_name)
    if int(sim.model.jnt_type[joint_id]) != 0:  # MuJoCo mjJNT_FREE
        raise RuntimeError(f"Expected a MuJoCo free joint for {joint_name}")
    return int(sim.model.jnt_qposadr[joint_id]), int(sim.model.jnt_dofadr[joint_id])


def _contact_pairs(sim):
    minimum_distance = {}
    deep_overlaps = []
    for contact in sim.data.contact[: sim.data.ncon]:
        body_1 = sim.model.body(int(sim.model.geom_bodyid[contact.geom1])).name
        body_2 = sim.model.body(int(sim.model.geom_bodyid[contact.geom2])).name
        pair = tuple(sorted((body_1, body_2)))
        distance = float(contact.dist)
        minimum_distance[pair] = min(distance, minimum_distance.get(pair, float("inf")))
        if distance < -CONTACT_PENETRATION_TOLERANCE_M:
            deep_overlaps.append((pair, distance))
    return minimum_distance, deep_overlaps


def apply_scene_perturbation(lerobot_env, spec: dict) -> tuple[dict, dict]:
    """Apply pose offsets after saved-state reset, settle consistently, and reobserve.

    Offsets translate actual MuJoCo free-joint roots in world X/Y; Z and quaternion
    are preserved. All scenarios, including the zero-offset reference, receive
    the same ten no-op physics steps after the standard LIBERO reset settling.
    """
    spec = validate_scenario(spec)
    wrapped, task_env = _task_env(lerobot_env)
    sim = task_env.sim
    objects = task_env.objects_dict
    before_pairs, before_overlaps = _contact_pairs(sim)
    initial_robot_qpos = np.asarray(sim.data.qpos[:9]).copy()
    pose_records = {}
    moved_body_names = set()

    for key, (object_name, joint_name) in OBJECT_JOINTS.items():
        translation = np.asarray(
            spec.get(f"{key}_translation_xy_m", [0.0, 0.0]), dtype=np.float64
        )
        if object_name not in objects:
            raise RuntimeError(f"Identified task object {object_name!r} is missing")
        qadr, dadr = _free_joint(sim, objects[object_name], joint_name)
        old_pose = np.asarray(sim.data.qpos[qadr : qadr + 7]).copy()
        new_pose = old_pose.copy()
        new_pose[:2] += translation
        sim.data.qpos[qadr : qadr + 7] = new_pose
        sim.data.qvel[dadr : dadr + 6] = 0.0
        pose_records[key] = {
            "object_name": object_name,
            "joint_name": joint_name,
            "original_pose_xyz_wxyz": old_pose.tolist(),
            "requested_translation_xy_m": translation.tolist(),
            "requested_pose_xyz_wxyz": new_pose.tolist(),
        }
        if np.any(translation):
            moved_body_names.add(objects[object_name].root_body)

    sim.forward()
    if not np.isfinite(sim.data.qpos).all() or not np.isfinite(sim.data.qvel).all():
        raise ValueError("perturbed MuJoCo state contains non-finite values")

    # The task goal is object-on-object, verified from the loaded task definition.
    goal = task_env.parsed_problem["goal_state"]
    goal_text = json.dumps(goal).lower()
    if "akita_black_bowl_1" not in goal_text or "plate_1" not in goal_text or "on" not in goal_text:
        raise RuntimeError(f"Task success predicate differs from audited bowl-on-plate goal: {goal!r}")

    # Confirm the task's On predicate refers to the live object states and not the
    # fixed plate_region site. This guards against a later task-definition drift.
    if task_env.object_states_dict["plate_1"] is task_env.object_states_dict.get("main_table_plate_region"):
        raise RuntimeError("Task target resolves to a fixed region, not the moved plate object")

    # Bound object centers to the actual collision table's XY extent with margin.
    table_ids = [i for i in range(sim.model.ngeom) if sim.model.geom(i).name == "table_collision"]
    if len(table_ids) != 1:
        raise RuntimeError("Cannot uniquely identify the table collision geometry")
    table_id = table_ids[0]
    table_center = np.asarray(sim.data.geom_xpos[table_id])[:2]
    half_xy = np.asarray(sim.model.geom_size[table_id])[:2]
    for key, (object_name, _) in OBJECT_JOINTS.items():
        pos = np.asarray(task_env.object_states_dict[object_name].get_geom_state()["pos"])
        if not np.isfinite(pos).all() or np.any(np.abs(pos[:2] - table_center) > half_xy - TABLE_MARGIN_M):
            raise ValueError(f"{key} pose leaves the table safety margin: {pos.tolist()}")
        pose_records[key]["settled_pose_xyz_wxyz"] = None

    # Ten steps matches LIBERO's own reset settle count. It is applied equally to
    # reference and perturbed conditions; robot drift is recorded for audit.
    from lerobot.envs.libero import get_libero_dummy_action
    for _ in range(SETTLE_STEPS):
        wrapped.step(get_libero_dummy_action())
    sim.forward()

    after_pairs, after_overlaps = _contact_pairs(sim)
    invalid_new_overlaps = [
        item for item in after_overlaps
        if any(body in item[0] for body in moved_body_names)
        and before_pairs.get(item[0], 0.0) >= -CONTACT_PENETRATION_TOLERANCE_M
    ]
    if invalid_new_overlaps:
        raise ValueError(f"new object interpenetration after perturbation: {invalid_new_overlaps[:8]}")

    for key, (object_name, joint_name) in OBJECT_JOINTS.items():
        qadr, _ = _free_joint(sim, objects[object_name], joint_name)
        pose_records[key]["settled_pose_xyz_wxyz"] = sim.data.qpos[qadr : qadr + 7].tolist()
        actual_delta = np.asarray(pose_records[key]["settled_pose_xyz_wxyz"][:2]) - np.asarray(
            pose_records[key]["original_pose_xyz_wxyz"][:2]
        )
        pose_records[key]["actual_translation_xy_m"] = actual_delta.tolist()

    robot_drift = float(np.max(np.abs(np.asarray(sim.data.qpos[:9]) - initial_robot_qpos)))
    raw_observation = wrapped.env._get_observations()
    observation = lerobot_env._format_raw_obs(raw_observation)
    for camera, image in observation["pixels"].items():
        if np.asarray(image).shape != (lerobot_env.observation_height, lerobot_env.observation_width, 3):
            raise ValueError(f"unexpected camera observation shape for {camera}")
    if task_env._check_success():
        raise ValueError("A scene reset unexpectedly satisfies the bowl-on-plate task before policy action")

    scene_record = {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": spec["scenario_id"],
        "split": spec["split"],
        "scenario_sha256": canonical_hash(spec),
        "task_id": spec["task_id"],
        "seed": spec["seed"],
        "saved_init_state_index": spec["saved_init_state_index"],
        "saved_init_state_sha256": spec.get("saved_init_state_sha256"),
        "settle_steps_after_perturbation": SETTLE_STEPS,
        "robot_qpos_max_drift": robot_drift,
        "pose_records": pose_records,
        "camera_sha256": {
            key: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
            for key, value in observation["pixels"].items()
        },
        "task_goal_predicate": goal,
        "task_success_at_policy_start": bool(task_env._check_success()),
        "contact_pairs_before": len(before_pairs),
        "contact_pairs_after": len(after_pairs),
        "preexisting_deep_overlaps": len(before_overlaps),
        "new_deep_overlaps": len(invalid_new_overlaps),
        "physical_validity": "passed",
    }
    return observation, scene_record

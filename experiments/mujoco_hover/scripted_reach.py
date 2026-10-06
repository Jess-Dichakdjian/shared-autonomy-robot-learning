"""Bounded differential-IK reach above the red cube, with paired image logging."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from trajectory_logger import EpisodeLogger
from policy_interface import ActionLimits, PolicyObservation, RobotAction, ScriptedPolicy, VLAAdapter


INITIAL_ARM_Q = np.array([0.0, -0.58, 0.0, -1.68, 0.0, 1.13, 0.8])
HAND_TCP_LOCAL = np.array([0.0, 0.0, 0.105])
HOVER_ABOVE_CUBE_TOP = 0.10
LOG_PERIOD = 0.1
CONTROL_PERIOD = 0.02
SUCCESS_TOLERANCE = 0.02
SUCCESS_HOLD_TIME = 0.5
INSTRUCTION = "Move the open gripper to hover 10 cm above the red cube."
POSITION_GAIN = 2.0
ORIENTATION_GAIN = 1.0
MAX_JOINT_SPEED = 0.35
DLS_DAMPING = 0.04
SAFE_CUBE_X_RANGE = (0.45, 0.55)
SAFE_CUBE_Y_RANGE = (-0.23, -0.17)


def tcp_pose(model: mujoco.MjModel, data: mujoco.MjData, hand_id: int):
    rotation = data.xmat[hand_id].reshape(3, 3)
    return data.xpos[hand_id] + rotation @ HAND_TCP_LOCAL, rotation


def orientation_error(target_rotation: np.ndarray, current_rotation: np.ndarray) -> np.ndarray:
    target_quat = np.empty(4)
    current_quat = np.empty(4)
    inverse_current = np.empty(4)
    error_quat = np.empty(4)
    error = np.empty(3)
    mujoco.mju_mat2Quat(target_quat, target_rotation.ravel())
    mujoco.mju_mat2Quat(current_quat, current_rotation.ravel())
    mujoco.mju_negQuat(inverse_current, current_quat)
    mujoco.mju_mulQuat(error_quat, target_quat, inverse_current)
    mujoco.mju_quat2Vel(error, error_quat, 1.0)
    return error


def differential_ik(model, data, hand_id, target_position, target_rotation):
    position, rotation = tcp_pose(model, data, hand_id)
    twist = np.r_[POSITION_GAIN * (target_position - position),
                  ORIENTATION_GAIN * orientation_error(target_rotation, rotation)]
    jac_position = np.zeros((3, model.nv))
    jac_rotation = np.zeros((3, model.nv))
    # MuJoCo expects this Jacobian point in world coordinates, attached to hand_id.
    mujoco.mj_jac(model, data, jac_position, jac_rotation, position, hand_id)
    jacobian = np.vstack((jac_position, jac_rotation))[:, :7]
    gram = jacobian @ jacobian.T
    gram.flat[::7] += DLS_DAMPING**2
    joint_velocity = jacobian.T @ np.linalg.solve(gram, twist)
    return np.clip(joint_velocity, -MAX_JOINT_SPEED, MAX_JOINT_SPEED), position


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="run without viewer or camera windows")
    parser.add_argument("--timeout", type=float, default=10.0, help="task timeout in simulated seconds")
    parser.add_argument("--settle", type=float, default=1.0, help="initial object settling time in seconds")
    parser.add_argument("--log-dir", type=Path, default=Path("recordings"))
    parser.add_argument("--scene", type=Path, help="MuJoCo scene XML; defaults to the scene beside this script or in the current directory")
    parser.add_argument("--randomize-cube", action="store_true", help="sample the red cube position in a small tested-area neighborhood")
    parser.add_argument("--seed", type=int, default=42, help="repeatable random seed used with --randomize-cube")
    parser.add_argument(
        "--policy-mode",
        choices=("scripted", "vla-stub-scripted-fallback"),
        default="scripted",
        help="select the scripted baseline or VLAAdapter with its scripted fallback",
    )
    args = parser.parse_args()
    if args.timeout <= 0 or args.settle < 0:
        parser.error("--timeout must be positive and --settle cannot be negative")

    script_dir = Path(__file__).resolve().parent
    scene_candidates = [script_dir / "franka_emika_panda" / "scene.xml",
                        Path.cwd() / "franka_emika_panda" / "scene.xml"]
    scene_path = args.scene.expanduser().resolve() if args.scene else next(
        (candidate.resolve() for candidate in scene_candidates if candidate.is_file()), None
    )
    if scene_path is None or not scene_path.is_file():
        raise FileNotFoundError(
            "MuJoCo scene not found. Pass --scene or run from a directory containing "
            "franka_emika_panda/scene.xml."
        )
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    if model.nu != 8 or model.nv < 7:
        raise RuntimeError(f"Expected the single Panda model (nu=8), got nu={model.nu}, nv={model.nv}")

    data.qpos[:7] = INITIAL_ARM_Q
    data.qpos[7:9] = 0.04
    sampled_cube_xy = None
    if args.randomize_cube:
        random = np.random.default_rng(args.seed)
        cube_id_for_reset = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block2")
        if cube_id_for_reset < 0 or model.body_jntnum[cube_id_for_reset] != 1:
            raise RuntimeError("Expected block2 to have one free joint")
        cube_joint_id = model.body_jntadr[cube_id_for_reset]
        if model.jnt_type[cube_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError("Expected block2 to use a free joint")
        cube_qpos = model.jnt_qposadr[cube_joint_id]
        sampled_cube_xy = np.array([
            random.uniform(*SAFE_CUBE_X_RANGE),
            random.uniform(*SAFE_CUBE_Y_RANGE),
        ])
        data.qpos[cube_qpos:cube_qpos + 2] = sampled_cube_xy
    data.ctrl[:7] = INITIAL_ARM_Q
    data.ctrl[7] = 255.0
    mujoco.mj_forward(model, data)

    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block2")
    static_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "static_cam")
    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ee_cam")
    ids = {"hand": hand_id, "block2": cube_id, "static_cam": static_id, "ee_cam": wrist_id}
    missing = [name for name, index in ids.items() if index < 0]
    if missing:
        raise RuntimeError(f"Scene is missing required names: {missing}")

    frame_width = frame_height = 400
    renderer_static = mujoco.Renderer(model, height=frame_height, width=frame_width)
    renderer_wrist = mujoco.Renderer(model, height=frame_height, width=frame_width)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    episode_dir = (script_dir / args.log_dir / f"reach_red_{timestamp}").resolve()
    episode_dir.parent.mkdir(parents=True, exist_ok=True)
    logger = EpisodeLogger(episode_dir)

    control_steps = max(1, round(CONTROL_PERIOD / model.opt.timestep))
    log_steps = max(control_steps, round(LOG_PERIOD / model.opt.timestep))
    settle_steps = round(args.settle / model.opt.timestep)
    max_steps = round(args.timeout / model.opt.timestep)
    initial_tcp, target_rotation = tcp_pose(model, data, hand_id)
    start_tcp = initial_tcp.copy()

    try:
        def run(viewer=None):
            nonlocal initial_tcp, target_rotation, start_tcp
            for _ in range(settle_steps):
                mujoco.mj_step(model, data)
                if viewer is not None:
                    viewer.sync()
            cube_start = data.xpos[cube_id].copy()
            final_target_position = cube_start + np.array([0.0, 0.0, 0.03 + HOVER_ABOVE_CUBE_TOP])
            initial_tcp, target_rotation = tcp_pose(model, data, hand_id)
            start_tcp = initial_tcp.copy()
            transfer_height = max(initial_tcp[2], final_target_position[2] + 0.15)
            waypoints = [
                initial_tcp.copy(),
                np.array([final_target_position[0], final_target_position[1], transfer_height]),
                final_target_position,
            ]
            waypoint_names = ["hold_start_pose", "transfer_above_cube", "hover_above_cube"]
            waypoint_index = 0
            waypoint_hold = 0.0
            qpos_addresses = model.jnt_qposadr[:7]
            dof_addresses = model.jnt_dofadr[:7]
            joint_target = data.qpos[qpos_addresses].copy()
            action_limits = ActionLimits.from_mujoco(
                model, max_joint_speed_rad_s=MAX_JOINT_SPEED
            )

            def scripted_controller(observation: PolicyObservation) -> RobotAction:
                nonlocal joint_target
                qdot, _ = differential_ik(
                    model,
                    data,
                    hand_id,
                    waypoints[waypoint_index],
                    target_rotation,
                )
                joint_target = joint_target + qdot * CONTROL_PERIOD
                return RobotAction(joint_target, gripper_finger_position_m=0.04)

            scripted_policy = ScriptedPolicy(scripted_controller)
            if args.policy_mode == "vla-stub-scripted-fallback":
                policy = VLAAdapter(fallback_policy=scripted_policy)
                policy_name = "VLAAdapter"
                policy_mode = "scripted_fallback_no_model"
            else:
                policy = scripted_policy
                policy_name = "ScriptedPolicy"
                policy_mode = "scripted_baseline"

            def request_action(observation: PolicyObservation) -> RobotAction:
                if isinstance(policy, VLAAdapter):
                    return policy.predict(observation, instruction=INSTRUCTION)
                return policy.predict(observation)

            last_requested_action = RobotAction(joint_target, 0.04)
            last_safe_action = last_requested_action
            success_elapsed = 0.0
            result = "timeout"
            reason = "target tolerance was not held before timeout"

            for step in range(max_steps):
                if viewer is not None and not viewer.is_running():
                    result, reason = "aborted", "viewer closed by user"
                    break
                if step % control_steps == 0:
                    tcp_position, _ = tcp_pose(model, data, hand_id)
                    observation = PolicyObservation(
                        instruction=INSTRUCTION,
                        seed=args.seed if args.randomize_cube else None,
                        sim_time_s=float(data.time),
                        robot_state={
                            "joint_position_rad": data.qpos[qpos_addresses].copy(),
                            "joint_velocity_rad_s": data.qvel[dof_addresses].copy(),
                            "gripper_finger_position_m": float(np.mean(data.qpos[7:9])),
                            "tcp_position_m": tcp_position.copy(),
                            "cube_position_m": data.xpos[cube_id].copy(),
                            "active_waypoint_m": waypoints[waypoint_index].copy(),
                        },
                    )
                    last_requested_action = request_action(observation)
                    previous_safe_action = last_safe_action
                    last_safe_action = action_limits.clamp(
                        last_requested_action,
                        previous_joint_target_rad=previous_safe_action.arm_joint_position_rad,
                        previous_gripper_target_m=previous_safe_action.gripper_finger_position_m,
                        dt_s=control_steps * model.opt.timestep,
                    )
                    joint_target = last_safe_action.arm_joint_position_rad.copy()
                    data.ctrl[:7] = last_safe_action.arm_joint_position_rad
                    data.ctrl[7] = last_safe_action.gripper_finger_position_m / 0.04 * 255.0
                    position_error = float(np.linalg.norm(waypoints[waypoint_index] - tcp_position))
                    if waypoint_index < len(waypoints) - 1:
                        waypoint_hold = waypoint_hold + control_steps * model.opt.timestep if position_error <= 0.035 else 0.0
                        if waypoint_hold >= 0.15:
                            waypoint_index += 1
                            waypoint_hold = 0.0
                    elif position_error <= SUCCESS_TOLERANCE:
                        success_elapsed += control_steps * model.opt.timestep
                    else:
                        success_elapsed = 0.0
                    if waypoint_index == len(waypoints) - 1 and success_elapsed >= SUCCESS_HOLD_TIME:
                        result, reason = "success", "TCP stayed within tolerance for the required hold time"

                tcp_position, _ = tcp_pose(model, data, hand_id)
                if step % log_steps == 0:
                    renderer_static.update_scene(data, camera=static_id)
                    renderer_wrist.update_scene(data, camera=wrist_id)
                    static_rgb = renderer_static.render()
                    wrist_rgb = renderer_wrist.render()
                    q_current = data.qpos[qpos_addresses].copy()
                    q_velocity = data.qvel[dof_addresses].copy()
                    current_error = float(np.linalg.norm(final_target_position - tcp_position))
                    logged_observation = PolicyObservation(
                        instruction=INSTRUCTION,
                        seed=args.seed if args.randomize_cube else None,
                        sim_time_s=float(data.time),
                        robot_state={
                            "joint_position_rad": q_current,
                            "joint_velocity_rad_s": q_velocity,
                            "gripper_finger_position_m": float(np.mean(data.qpos[7:9])),
                            "tcp_position_m": tcp_position,
                            "tcp_quaternion_wxyz": data.xquat[hand_id].copy(),
                            "cube_position_m": data.xpos[cube_id].copy(),
                            "target_position_m": final_target_position,
                            "active_waypoint_m": waypoints[waypoint_index],
                        },
                        static_rgb=static_rgb,
                        wrist_rgb=wrist_rgb,
                        metadata={"controller_mode": waypoint_names[waypoint_index]},
                    )
                    logger.log({
                        "sim_time_s": float(data.time),
                        "instruction": INSTRUCTION,
                        "seed": args.seed if args.randomize_cube else None,
                        "policy": policy_name,
                        "policy_mode": policy_mode,
                        "episode_outcome": "running",
                        "observation": logged_observation.to_record(),
                        "requested_action": last_requested_action.to_record(),
                        "clamped_action": last_safe_action.to_record(),
                        "action": last_safe_action.to_record(),
                        "controller_mode": waypoint_names[waypoint_index],
                        "joint_position_rad": q_current.tolist(),
                        "joint_velocity_rad_s": q_velocity.tolist(),
                        "joint_position_target_rad": joint_target.tolist(),
                        "gripper_command": float(data.ctrl[7]),
                        "tcp_position_m": tcp_position.tolist(),
                        "tcp_quaternion_wxyz": data.xquat[hand_id].tolist(),
                        "cube_position_m": data.xpos[cube_id].tolist(),
                        "target_position_m": final_target_position.tolist(),
                        "active_waypoint_m": waypoints[waypoint_index].tolist(),
                        "target_error_m": current_error,
                    }, static_rgb, wrist_rgb)
                    if viewer is not None:
                        cv2.imshow("Static Camera", cv2.cvtColor(static_rgb, cv2.COLOR_RGB2BGR))
                        cv2.imshow("End Effector Camera", cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)

                mujoco.mj_step(model, data)
                if viewer is not None:
                    viewer.sync()
                if result == "success":
                    break

            return {
                "result": result,
                "reason": reason,
                "instruction": INSTRUCTION,
                "policy": policy_name,
                "policy_mode": policy_mode,
                "model_inference_called": False,
                "scene_xml": str(scene_path),
                "scene_commit": "708bfc8c67ecdd232e3130736055cf0856f1685d",
                "seed": args.seed if args.randomize_cube else None,
                "cube_randomized_xy_m": sampled_cube_xy.tolist() if sampled_cube_xy is not None else None,
                "cube_randomization_bounds_m": {
                    "x": list(SAFE_CUBE_X_RANGE),
                    "y": list(SAFE_CUBE_Y_RANGE),
                } if args.randomize_cube else None,
                "mujoco_version": mujoco.__version__,
                "numpy_version": np.__version__,
                "opencv_version": cv2.__version__,
                "physics_timestep_s": float(model.opt.timestep),
                "logging_period_s": float(log_steps * model.opt.timestep),
                "settle_time_s": args.settle,
                "timeout_s": args.timeout,
                "success_tolerance_m": SUCCESS_TOLERANCE,
                "success_hold_time_s": SUCCESS_HOLD_TIME,
                "tcp_local_offset_m": HAND_TCP_LOCAL.tolist(),
                "initial_tcp_position_m": start_tcp.tolist(),
                "cube_initial_position_m": cube_start.tolist(),
                "target_position_m": final_target_position.tolist(),
                "waypoints_m": [point.tolist() for point in waypoints],
                "final_tcp_position_m": tcp_pose(model, data, hand_id)[0].tolist(),
                "final_target_error_m": float(np.linalg.norm(final_target_position - tcp_pose(model, data, hand_id)[0])),
                "logged_frames": logger.frame_id,
                "ended_utc": datetime.now(timezone.utc).isoformat(),
            }

        if args.headless:
            manifest = run()
        else:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                manifest = run(viewer)
        (episode_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        print(f"Episode saved to: {episode_dir}")
    finally:
        logger.close()
        renderer_static.close()
        renderer_wrist.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

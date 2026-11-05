import math
import numpy as np
import pytest

from thesis_vla.teleoperation import (
    MAX_POSITION_STEP_M, MIN_POSITION_STEP_M, MAX_ROTATION_STEP_RAD,
    MIN_ROTATION_STEP_RAD, camera_plane_basis, map_key, translation_action,
    world_delta_for_direction,
)
from scripts.teleop_libero import TeleopSession


@pytest.mark.parametrize("key,index,sign", [
    ("w", 0, 1), ("s", 0, -1), ("d", 1, 1), ("a", 1, -1),
    ("r", 2, 1), ("f", 2, -1), ("i", 3, 1), ("k", 3, -1),
    ("l", 4, 1), ("j", 4, -1), ("u", 5, 1), ("m", 5, -1),
])
def test_each_key_maps_only_to_expected_axis(key, index, sign):
    action = map_key(key)
    assert len(action) == 7
    assert action[index] == pytest.approx(sign * (0.005 / 0.05 if index < 3 else math.radians(3) / 0.5))
    assert sum(v != 0 for v in action) == 1
    assert max(abs(v) for v in action) <= 1


def test_translation_step_limits_map_to_controller_limits():
    assert map_key("w", position_step_m=MIN_POSITION_STEP_M)[0] == pytest.approx(.02)
    assert map_key("w", position_step_m=MAX_POSITION_STEP_M)[0] == pytest.approx(.6)
    for invalid in (0, .0009, .0301, float("nan")):
        with pytest.raises(ValueError):
            map_key("w", position_step_m=invalid)


def test_rotation_step_limits_and_signs():
    assert map_key("i", rotation_step_rad=MIN_ROTATION_STEP_RAD)[3] == pytest.approx(math.radians(1)/.5)
    assert map_key("k", rotation_step_rad=MAX_ROTATION_STEP_RAD)[3] == pytest.approx(-math.radians(8)/.5)
    with pytest.raises(ValueError):
        map_key("i", rotation_step_rad=0)


@pytest.mark.parametrize("gripper,expected", [("open", -1.0), ("close", 1.0)])
def test_panda_gripper_signs(gripper, expected):
    action = map_key("", gripper=gripper)
    assert action == (0, 0, 0, 0, 0, 0, expected)


def test_invalid_mapping_requests_are_rejected():
    for args in (("x",), ("",)):
        with pytest.raises(ValueError):
            map_key(*args)
    with pytest.raises(ValueError):
        map_key("", gripper="invalid")


def test_camera_extrinsics_produce_opposite_view_axes():
    scene_r = np.asarray([[-.0000, -.6283, .7780], [1.0, 0.0, 0.0], [0.0, .7780, .6283]])
    wrist_r = np.asarray([[.0016, .9984, .0558], [-1.0, .0016, -.0004], [-.0005, -.0558, .9984]])
    scene, wrist = camera_plane_basis(scene_r), camera_plane_basis(wrist_r)
    assert scene["right_world"] == pytest.approx([0, 1, 0])
    assert scene["forward_world"] == pytest.approx([-1, 0, 0])
    assert wrist["right_world"] == pytest.approx([0, -1, 0], abs=0.002)
    assert wrist["forward_world"] == pytest.approx([1, 0, 0], abs=0.002)


def test_camera_relative_direction_maps_to_extrinsic_world_vector():
    scene = camera_plane_basis(np.asarray([[0, -.6283, .778], [1, 0, 0], [0, .778, .6283]]))
    wrist = camera_plane_basis(np.asarray([[.0016, .9984, .0558], [-1, .0016, -.0004], [-.0005, -.0558, .9984]]))
    assert world_delta_for_direction("forward", .008, reference="scene", scene_basis=scene,
                                     wrist_basis=wrist) == pytest.approx([-.008, 0, 0])
    assert world_delta_for_direction("right", .008, reference="wrist", scene_basis=scene,
                                     wrist_basis=wrist) == pytest.approx([0, -.008, 0], abs=2e-5)
    assert world_delta_for_direction("up", .008, reference="scene", scene_basis=scene,
                                     wrist_basis=wrist) == pytest.approx([0, 0, .008])


def test_translation_action_uses_normalized_world_delta_and_gripper_hold():
    action = translation_action(np.asarray([.01, -.005, .002]), gripper=0)
    assert action == pytest.approx([.2, -.1, .04, 0, 0, 0, 0])
    assert translation_action(np.zeros(3), gripper=-1)[6] == -1
    assert translation_action(np.zeros(3), gripper=1)[6] == 1
    with pytest.raises(ValueError):
        translation_action(np.asarray([.051, 0, 0]))


def _test_camera_bases():
    scene = camera_plane_basis(np.asarray([[0, -.6283, .778], [1, 0, 0], [0, .778, .6283]]))
    wrist = camera_plane_basis(np.asarray([[.0016, .9984, .0558], [-1, .0016, -.0004], [-.0005, -.0558, .9984]]))
    return scene, wrist


def test_wrist_180_display_rotation_flips_both_planar_axes_only():
    scene, wrist = _test_camera_bases()
    original_forward = world_delta_for_direction(
        "forward", .002, reference="wrist", wrist_basis=wrist, wrist_display_rotation_deg=0)
    rotated_forward = world_delta_for_direction(
        "forward", .002, reference="wrist", wrist_basis=wrist, wrist_display_rotation_deg=180)
    original_right = world_delta_for_direction(
        "right", .002, reference="wrist", wrist_basis=wrist, wrist_display_rotation_deg=0)
    rotated_right = world_delta_for_direction(
        "right", .002, reference="wrist", wrist_basis=wrist, wrist_display_rotation_deg=180)
    assert rotated_forward == pytest.approx(-original_forward)
    assert rotated_right == pytest.approx(-original_right)
    assert world_delta_for_direction(
        "forward", .002, reference="scene", scene_basis=scene, wrist_basis=wrist,
        wrist_display_rotation_deg=180) == pytest.approx(
            world_delta_for_direction("forward", .002, reference="scene",
                                      scene_basis=scene, wrist_basis=wrist,
                                      wrist_display_rotation_deg=0))


def test_world_mapping_is_independent_of_wrist_display_rotation():
    _, wrist = _test_camera_bases()
    original = world_delta_for_direction("right", .005, reference="world", wrist_basis=wrist,
                                         wrist_display_rotation_deg=0)
    rotated = world_delta_for_direction("right", .005, reference="world", wrist_basis=wrist,
                                        wrist_display_rotation_deg=180)
    assert original == pytest.approx([0, .005, 0])
    assert rotated == pytest.approx(original)
    assert world_delta_for_direction("up", .005, reference="wrist", wrist_basis=wrist,
                                     wrist_display_rotation_deg=180) == pytest.approx([0, 0, .005])


def test_wrist_rotation_mapping_uses_current_camera_extrinsics():
    _, wrist_initial = _test_camera_bases()
    yaw = math.radians(37)
    yaw_world = np.asarray([[math.cos(yaw), -math.sin(yaw), 0],
                            [math.sin(yaw), math.cos(yaw), 0],
                            [0, 0, 1]])
    initial_rotation = np.asarray([[.0016, .9984, .0558], [-1, .0016, -.0004], [-.0005, -.0558, .9984]])
    wrist_after_rotation = camera_plane_basis(yaw_world @ initial_rotation)
    requested = world_delta_for_direction("forward", .002, reference="wrist",
                                          wrist_basis=wrist_after_rotation,
                                          wrist_display_rotation_deg=180)
    assert requested == pytest.approx(-.002 * wrist_after_rotation["forward_world"])
    assert requested != pytest.approx(-.002 * wrist_initial["forward_world"])


def test_operator_view_changes_are_recorded_in_session_metadata():
    import threading

    session = TeleopSession.__new__(TeleopSession)
    session.lock = threading.Lock()
    session.hold_lock = threading.Lock()
    session.active_hold = None
    session.mode = "HUMAN"
    session.epoch = 0
    session.step = 0
    session.run_id = "synthetic-view-test"
    session.control_reference = "scene"
    session.wrist_display_rotation_deg = 180
    session.camera_axes = {}
    session.operator_view_history = []
    records = []
    session._log = records.append
    session._write_config = lambda: None
    session.snapshot = lambda: session._operator_view()

    session.set_reference("wrist")
    assert session._operator_view()["display_transform_applied_to_control"] is True
    session.set_wrist_display_rotation(0)
    assert session._operator_view()["display_transform_applied_to_control"] is False
    assert len(records) == len(session.operator_view_history) == 2
    assert [record["wrist_display_rotation_deg"] for record in records] == [180, 0]
    with pytest.raises(ValueError):
        session.set_wrist_display_rotation(90)


def test_action_log_preserves_command_state_view_and_camera_hashes():
    session = TeleopSession.__new__(TeleopSession)
    session.env = type("Env", (), {"step": lambda _self, action: (
        {"robot_state": {"eef": {"pos": [.101, .2, .3]}}}, .25, False, False, {"is_success": False})})()
    session.observation = {"robot_state": {"eef": {"pos": [.1, .2, .3]}}}
    session.reward = 0.0
    session.terminated = session.truncated = False
    session.last_info = {}
    session.step = 0
    session.run_id = "synthetic-log-test"
    session.started_wall = 0
    session.camera_axes = {"wrist": {"rotation_camera_to_world": np.eye(3)}}
    session.control_reference = "wrist"
    session.wrist_display_rotation_deg = 180
    session._update_camera_axes = lambda: None
    expected_hash = "a" * 64
    session._save_raw_cameras = lambda _obs, _step: {"wrist": {"sha256": expected_hash}}
    session._write_video_frame = lambda _obs: None
    records = []
    session._log = records.append

    action = np.asarray([.02, 0, 0, 0, 0, 0, 0])
    session._execute_env_step(action, command={"kind": "move"}, step_index=1,
                              planned_steps=1, input_method="keyboard",
                              before=session._state(), started_mono=0.0)
    row = records[0]
    assert row["executed_action_7d"] == pytest.approx(action.tolist())
    assert row["operator_view"] == {
        "control_reference": "wrist", "wrist_display_rotation_deg": 180,
        "display_transform_applied_to_control": True}
    assert row["camera_files"]["wrist"]["sha256"] == expected_hash
    assert row["robot_state_before"]["eef"]["pos"] == pytest.approx([.1, .2, .3])
    assert row["robot_state_after"]["eef"]["pos"] == pytest.approx([.101, .2, .3])


def test_canonical_camera_png_and_hash_are_independent_of_view_rotation(tmp_path):
    from PIL import Image

    session = TeleopSession.__new__(TeleopSession)
    session.output_dir = tmp_path
    session.camera_dir = tmp_path / "camera_raw"
    (session.camera_dir / "scene").mkdir(parents=True)
    (session.camera_dir / "wrist").mkdir(parents=True)
    scene = np.arange(256 * 256 * 3, dtype=np.uint32).reshape(256, 256, 3).astype(np.uint8)
    wrist = np.flip(scene, axis=1).copy()
    saved = session._save_raw_cameras({"pixels": {"image": scene, "image2": wrist}}, 7)
    for label, expected in (("scene", scene), ("wrist", wrist)):
        frame_path = tmp_path / saved[label]["path"]
        decoded = np.asarray(Image.open(frame_path))
        assert np.array_equal(decoded, expected)
        assert saved[label]["sha256"] == __import__("hashlib").sha256(expected.tobytes()).hexdigest()


def test_heartbeat_and_emergency_stop_cancel_active_input():
    import threading
    import time

    session = TeleopSession.__new__(TeleopSession)
    session.lock = threading.Lock()
    session.hold_lock = threading.Lock()
    session.active_hold = None
    session.mode = "HUMAN"
    session.epoch = 0
    session.step = 2
    session.run_id = "synthetic-safety-test"
    session.stop_hold = TeleopSession.stop_hold.__get__(session)
    session.snapshot = lambda: session.mode
    session._log = lambda _record: None
    session.begin_hold("hold-1")
    assert session.heartbeat("hold-1") is True
    assert session._lease_valid("hold-1") is True
    session.active_hold["last_heartbeat"] = time.monotonic() - 2.0
    assert session._lease_valid("hold-1") is False
    session.begin_hold("hold-2")
    assert session.heartbeat("hold-2") is True
    assert session.transition("ESTOP") == "ESTOP"
    assert session.epoch == 1
    assert session.heartbeat("hold-2") is False
    assert session._lease_valid("hold-2") is False

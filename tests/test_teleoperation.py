import math
import numpy as np
import pytest

from thesis_vla.teleoperation import (
    MAX_POSITION_STEP_M, MIN_POSITION_STEP_M, MAX_ROTATION_STEP_RAD,
    MIN_ROTATION_STEP_RAD, camera_plane_basis, map_key, translation_action,
    world_delta_for_direction,
)


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

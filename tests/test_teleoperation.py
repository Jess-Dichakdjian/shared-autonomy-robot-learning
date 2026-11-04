import math
import pytest

from thesis_vla.teleoperation import (
    KEY_AXES, MAX_POSITION_STEP_M, MIN_POSITION_STEP_M,
    MAX_ROTATION_STEP_RAD, MIN_ROTATION_STEP_RAD, map_key,
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
    assert map_key("w", position_step_m=MAX_POSITION_STEP_M)[0] == pytest.approx(.4)
    for invalid in (0, .0009, .0201, float("nan")):
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
        map_key("", gripper="hold")

"""Scenario specification and physical-scene validity tests."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from thesis_vla.perturbations import canonical_hash, validate_scenario


def test_scenario_schema_is_finite_bounded_and_task_specific():
    valid={"schema_version":1,"scenario_id":"unit","split":"discovery","task_id":1,
           "seed":1101,"saved_init_state_index":0,"bowl_translation_xy_m":[0.02,0.0],
           "plate_translation_xy_m":[0.0,0.0]}
    assert validate_scenario(valid) is valid
    for key,value in (("bowl_translation_xy_m",[float("nan"),0]),
                      ("plate_translation_xy_m",[0.081,0]),
                      ("bowl_translation_xy_m",[1,2,3])):
        with pytest.raises(ValueError):
            validate_scenario({**valid,key:value})
    with pytest.raises(ValueError,match="task 1"):
        validate_scenario({**valid,"task_id":0})


def test_canonical_scenario_hash_is_order_independent():
    assert canonical_hash({"a":1,"b":[2,3]})==canonical_hash({"b":[2,3],"a":1})


def test_preregistered_split_counts_are_disjoint_and_test_stays_closed():
    path=Path(__file__).parents[1]/"configs/libero_task1_scene_scenarios_v1.json"
    plan=json.loads(path.read_text())
    assert plan["final_test_status"]=="reserved_not_run"
    ids=[item["scenario_id"] for item in plan["scenarios"]]
    assert len(ids)==len(set(ids))==24
    assert {split:sum(item["split"]==split for item in plan["scenarios"])
            for split in ("discovery","validation","held_out")}=={
                "discovery":12,"validation":6,"held_out":6}
    hashes=[canonical_hash(item) for item in plan["scenarios"]]
    assert len(hashes)==len(set(hashes))


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIBERO_INTEGRATION") != "1",
    reason="set RUN_LIBERO_INTEGRATION=1 for physical LIBERO scene tests",
)


def test_live_bowl_and_plate_perturbations_change_physics_and_camera():
    from scripts.libero_smoke import prepare_config_and_assets
    from thesis_vla.libero import LiberoTaskConfig, LiberoVectorAdapter, create_vector_env
    prepare_config_and_assets()
    plan=json.loads((Path(__file__).parents[1]/"configs/libero_task1_scene_scenarios_v1.json").read_text())
    scenarios=[next(s for s in plan["scenarios"] if s["scenario_id"]==key)
               for key in ("D00_reference","D01_bowl_x_plus_2cm","D10_plate_x_plus_2cm")]
    env=LiberoVectorAdapter(create_vector_env(LiberoTaskConfig(suite="libero_spatial",task_id=1),gl=os.environ.get("MUJOCO_GL","osmesa")))
    records=[]
    try:
        for spec in scenarios:
            obs,info=env.reset(seed=spec["seed"],options={
                "saved_init_state_index":spec["saved_init_state_index"],"scene_perturbation":spec})
            record=info["scene_perturbation"]
            assert record["physical_validity"]=="passed"
            assert record["task_success_at_policy_start"] is False
            assert set(record["camera_sha256"])=={"image","image2"}
            assert all(np.asarray(v).shape==(256,256,3) for v in obs["pixels"].values())
            records.append(record)
        assert records[0]["scenario_sha256"]!=records[1]["scenario_sha256"]
        assert records[0]["camera_sha256"]!=records[1]["camera_sha256"]
        assert records[1]["pose_records"]["bowl"]["actual_translation_xy_m"][0]==pytest.approx(0.02,abs=0.005)
        assert records[2]["pose_records"]["plate"]["actual_translation_xy_m"][0]==pytest.approx(0.02,abs=0.005)
        assert records[1]["pose_records"]["plate"]["actual_translation_xy_m"]==pytest.approx([0,0],abs=1e-6)
    finally:
        env.close()

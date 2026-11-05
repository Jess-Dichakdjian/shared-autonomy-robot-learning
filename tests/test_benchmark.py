import hashlib
import json
import numpy as np
import pytest
import torch
from thesis_vla.benchmark import run_benchmark, validate_episode, completed_episodes, wilson
from thesis_vla.dataset import export_dataset
from thesis_vla.pi05 import Pi05PolicyAdapter


class SceneEnv:
    def __init__(self, fail_at=None):
        self.resets=[]; self.index=99; self.step_index=0; self.fail_at=fail_at
    def reset(self, *, seed, options):
        self.index=options["saved_init_state_index"]; self.step_index=0
        self.resets.append((seed,self.index))
        info = {"saved_init_state_index":self.index,
            "saved_init_state_sha256":hashlib.sha256(str(self.index).encode()).hexdigest(), "is_success":False}
        if "scene_perturbation" in options:
            scene = options["scene_perturbation"]
            info["scene_perturbation"] = {"scenario_id":scene["scenario_id"],
                "scenario_sha256":scene["scenario_sha256"]}
        return self.obs(), info
    def obs(self):
        return {"state":np.array([self.index,self.step_index,0,0,0,0,0,0],np.float32),
            "robot_state":{"index":self.index,"step":self.step_index}}
    def step(self, action):
        self.step_index+=1
        if self.index==self.fail_at and self.step_index==2:
            raise RuntimeError("simulated interruption")
        done=self.step_index==3; success=done and self.index%2==0
        return self.obs(),float(success),bool(success),bool(done and not success),{"is_success":success}


def policy():
    capture={}
    class Inference:
        def __init__(self): self.resets=0
        def reset(self): self.resets+=1
        def predict_action_chunk(self,batch):
            a=torch.zeros(1,2,7)
            a[0,:,0]=float(batch["state"][1])
            return a
    def prepare(obs, task):
        capture["physical"]=obs["state"].copy()
        capture["vla_normalized"]=obs["state"].copy()/10
        return {"state":capture["vla_normalized"]}
    p=Pi05PolicyAdapter(Inference(),prepare,task="fixture",execute_steps=2)
    p.prediction_state_capture=capture
    return p


def run(env,p,path,**kwargs):
    return run_benchmark(env,p,path,configuration={"policy":"test"},task="fixture",record_video=False,**kwargs)


def test_five_episode_pilot_resumes_to_twenty_without_repeating_completed(tmp_path):
    # A prior gate can have advanced the environment arbitrarily; explicit indices win.
    env=SceneEnv();p=policy()
    r=run(env,p,tmp_path,stop_after=5)
    assert r["completed"]==5
    assert (tmp_path/"pilot_report.json").exists()
    r=run(env,p,tmp_path)
    assert r["completed"]==20 and r["successes"]==10 and r["failures"]==10
    assert env.resets==[(i,i) for i in range(20)]
    assert p.policy.resets==20
    schema=export_dataset(tmp_path)
    assert schema["samples"]==40 and schema["step_samples"]==60
    with np.load(tmp_path/"dataset/gp_chunk_starts.npz",allow_pickle=False) as d:
        assert d["X"].shape==(40,8) and d["Y"].shape==(40,7)
        assert set(d["step"])=={0,2}
        np.testing.assert_array_equal(d["Y"][:,0],d["X"][:,1])
    rows=[json.loads(s) for s in (tmp_path/"dataset/full_step_trajectories.jsonl").read_text().splitlines()]
    assert rows[1]["prediction_step"]==0
    assert rows[1]["prediction_state"]==rows[0]["prediction_state"]
    assert rows[1]["robot_state"]!=rows[0]["robot_state"]


def test_interruption_archives_only_incomplete_episode_and_recovers(tmp_path):
    with pytest.raises(RuntimeError,match="interruption"):
        run(SceneEnv(fail_at=2),policy(),tmp_path,total=5)
    assert len(completed_episodes(tmp_path))==2
    env=SceneEnv()
    r=run(env,policy(),tmp_path,total=5)
    assert r["completed"]==5 and env.resets==[(i,i) for i in range(2,5)]
    assert len(list((tmp_path/"interrupted_attempts").iterdir()))==1
    with pytest.raises(ValueError,match="Resume protocol"):
        run(env,policy(),tmp_path,total=5,base_seed=99)
    path=tmp_path/"episodes/episode_000/steps.jsonl"
    path.write_text(path.read_text()+"\n")
    with pytest.raises(ValueError,match="Committed evidence changed"):
        completed_episodes(tmp_path)


def test_validation_rejects_queued_action_repaired_to_later_state(tmp_path):
    run(SceneEnv(),policy(),tmp_path,total=1,pilot=1)
    path=tmp_path/"episodes/episode_000/steps.jsonl"
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    rows[1]["prediction_state"]["vla_normalized"][1]=0.1
    path.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
    with pytest.raises(ValueError,match="incorrectly paired"):
        validate_episode(path.parent,require_video=False)


def test_wilson_boundary_results():
    assert wilson(0,0)==[None,None]
    assert wilson(0,20)[0]==0
    assert wilson(20,20)[1]==1
    assert wilson(10,20)==pytest.approx([0.2992980082,0.7007019918])


def test_scenario_plan_allows_distinct_scenes_from_same_saved_state(tmp_path):
    from thesis_vla.perturbations import canonical_hash
    scenes=[]
    for i,shift in enumerate((0.02,-0.02)):
        scene={"scenario_id":f"D{i}","scenario_sha256":canonical_hash({"shift":shift}),
               "bowl_translation_xy_m":[shift,0.0]}
        scenes.append({"episode_id":i,"seed":1101,"saved_init_state_index":0,
                       "scenario_id":scene["scenario_id"],"scene_perturbation":scene})
    env=SceneEnv()
    result=run(env,policy(),tmp_path,total=2,pilot=2,scenario_specifications=scenes)
    assert result["completed"]==2
    assert env.resets==[(1101,0),(1101,0)]
    records=completed_episodes(tmp_path)
    assert {r["scenario_id"] for r in records}=={"D0","D1"}
    assert len({r["scenario_sha256"] for r in records})==2

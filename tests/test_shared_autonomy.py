import pytest
from thesis_vla.shared_autonomy import SharedAutonomyProtocol

class Policy:
    def __init__(self): self.resets=0
    def reset(self): self.resets += 1

class Proposal:
    def __init__(self, boundary, request=None, chunk_id=3, chunk_index=0):
        self.is_chunk_start=boundary
        self.assistance_request=request
        self.chunk_id=chunk_id
        self.chunk_index=chunk_index

def test_manual_handoff_waits_for_chunk_boundary_and_discards_queue():
    p=Policy(); c=SharedAutonomyProtocol(p)
    c.start_autonomy()
    assert p.resets==1
    assert c.request_human()["pending_chunk_boundary"]
    assert c.at_chunk_boundary(Proposal(False)) is None
    event=c.at_chunk_boundary(Proposal(True))
    assert event.source=="manual_operator_request"
    assert c.state==c.REQUESTED and p.resets==2
    c.accept_human(); c.resume_autonomy()
    assert c.state==c.AUTONOMY and p.resets==3

def test_gp_trigger_only_hands_off_when_policy_marks_chunk_start():
    p=Policy(); c=SharedAutonomyProtocol(p); c.start_autonomy()
    proposal=Proposal(True,{"requested":True,"request":{"request_id":"ep:chunk3"}})
    event=c.at_chunk_boundary(proposal)
    assert event.source=="gp_trigger"
    assert event.request["request_id"]=="ep:chunk3"
    assert c.state==c.REQUESTED

def test_no_unrequested_or_nonboundary_handoff():
    p=Policy(); c=SharedAutonomyProtocol(p); c.start_autonomy()
    assert c.at_chunk_boundary(Proposal(False,{"requested":True})) is None
    assert c.at_chunk_boundary(Proposal(True,{"requested":False})) is None
    assert c.state==c.AUTONOMY

def test_invalid_transitions_and_estop():
    p=Policy(); c=SharedAutonomyProtocol(p)
    with pytest.raises(RuntimeError): c.accept_human()
    c.start_autonomy(); c.emergency_stop()
    assert c.state==c.ESTOP
    with pytest.raises(RuntimeError): c.resume_autonomy()


def _mock_integrated_session(proposal):
    import threading
    from types import SimpleNamespace
    import numpy as np
    from scripts.teleop_shared_autonomy import SharedSession
    class Policy:
        def __init__(self): self.resets=0
        def reset(self): self.resets+=1
        def act(self, observation): return proposal
    class Env:
        def __init__(self): self.actions=[]
        def step(self, action):
            self.actions.append(np.asarray(action).copy())
            return {"robot_state":{"eef":{"pos":[0.1,0.2,0.3]}},
                    "pixels":{"image":np.zeros((256,256,3),dtype=np.uint8),
                              "image2":np.zeros((256,256,3),dtype=np.uint8)}},0.0,False,False,{}
    policy=Policy(); session=SharedSession.__new__(SharedSession)
    session.policy=policy;session.mode="AUTONOMY";session.protocol=SharedAutonomyProtocol(policy)
    session.protocol.state=SharedAutonomyProtocol.AUTONOMY
    blank=np.zeros((256,256,3),dtype=np.uint8)
    session.lock=threading.Lock();session.observation={"robot_state":{"eef":{"pos":[0.,0.,0.]}},"pixels":{"image":blank,"image2":blank.copy()}}
    session.env=Env();session.run_id="test";session.step=0;session.epoch=0;session.reward=0.0
    session.terminated=False;session.truncated=False;session.last_info={}
    session.camera_axes={};session.last_action=None;session.fresh_resume_pending=False
    session._state=lambda:session.observation["robot_state"]
    session._update_camera_axes=lambda:None
    session._save_raw_cameras=lambda obs,step:{"scene":{"sha256":"a"*64},"wrist":{"sha256":"b"*64}}
    session._write_video_frame=lambda obs:None
    session.records=[];session._log=lambda rec:session.records.append(rec)
    return session


def test_integrated_mock_step_executes_and_logs_vla_source():
    import numpy as np
    from types import SimpleNamespace
    proposal=SimpleNamespace(predicted_action=np.zeros(7),executed_action=np.full(7,.1),
        chunk_id=1,chunk_index=2,is_chunk_start=False,passive_gp_prediction=None,assistance_request=None)
    session=_mock_integrated_session(proposal)
    result=session.autonomous_step()
    assert result["step"]==1
    np.testing.assert_array_equal(session.env.actions[0],np.full(7,.1))
    record=next(r for r in session.records if r["record_type"]=="vla_action_step")
    assert record["action_source"]=="pi05" and record["owner"]=="AUTONOMY"
    assert len(record["predicted_raw_normalized_action_7d"])==7
    assert record["action_executed"] is True


def test_integrated_gp_handoff_preempts_action_and_clears_queue():
    import numpy as np
    from types import SimpleNamespace
    proposal=SimpleNamespace(predicted_action=np.zeros(7),executed_action=np.full(7,.1),
        chunk_id=2,chunk_index=0,is_chunk_start=True,passive_gp_prediction={"std":[.2]*7},
        assistance_request={"requested":True,"request":{"request_id":"episode:chunk2","signal_name":"gp_uncertainty"}})
    session=_mock_integrated_session(proposal)
    result=session.autonomous_step()
    assert result["handoff"] is True and session.mode=="ASSISTANCE_REQUESTED"
    assert not session.env.actions
    assert session.policy.resets==1
    record=next(r for r in session.records if r["record_type"]=="assistance_handoff_request")
    assert record["candidate_executed"] is False
    assert record["queued_pi05_actions_cleared"] is True
    assert record["request_source"]=="gp_trigger"


def test_duplicate_manual_requests_are_idempotent_and_keep_origin():
    p=Policy(); c=SharedAutonomyProtocol(p); c.start_autonomy()
    first=c.request_human("run:manual:one",origin="integration_preflight")
    second=c.request_human("run:manual:two",origin="human_operator")
    assert first["request_id"]==second["request_id"]=="run:manual:one"
    assert second["already_pending"] is True
    event=c.at_chunk_boundary(Proposal(True))
    assert event.request["request_origin"]=="integration_preflight"


def test_fresh_policy_chunk_after_correction_is_logged():
    import numpy as np
    from types import SimpleNamespace
    proposal=SimpleNamespace(predicted_action=np.zeros(7),executed_action=np.full(7,.1),
        chunk_id=4,chunk_index=0,is_chunk_start=True,passive_gp_prediction=None,
        assistance_request={"requested":False})
    session=_mock_integrated_session(proposal)
    session.fresh_resume_pending=True
    session.autonomous_step()
    record=next(r for r in session.records if r["record_type"]=="vla_action_step")
    assert record["fresh_post_correction_inference"] is True
    assert record["policy_reset_before_inference"] is True


def test_human_correction_dataset_records_source_actions_and_camera_provenance(tmp_path):
    import numpy as np
    from scripts.teleop_shared_autonomy import SharedSession
    proposal=type("P",(),{})()
    session=_mock_integrated_session(proposal)
    session.output_dir=tmp_path
    session.reward=0.0
    session.terminated=session.truncated=False
    session.last_info={}
    session._operator_view=lambda **kwargs:{"control_reference":"scene"}
    session.manual_request_id="test:manual:1"
    session.current_intervention_id="test:handoff:1"
    before=session._state()
    SharedSession._execute_env_step(session,np.zeros(7,dtype=np.float32),command={"kind":"move"},
        step_index=1,planned_steps=1,input_method="button",before=before,started_mono=1.0)
    record=__import__("json").loads((tmp_path/"human_corrections.jsonl").read_text().splitlines()[0])
    assert record["source"]=="human_button"
    assert record["intervention_id"]=="test:handoff:1"
    assert len(record["executed_action_7d"])==7
    assert set(record["observation_before"]["camera_frames"])=={"scene","wrist"}
    assert all(len(v["sha256"])==64 for v in record["observation_before"]["camera_frames"].values())
    assert set(record["observation_after"]["camera_frames"])=={"scene","wrist"}

import numpy as np
import pytest
from thesis_vla.gp.monitoring import (
    AssistanceRequest, CorrectionBudget, MonitoringSignals, arbitrate_action,
    compute_monitoring_signals, decide_assistance, random_matched_schedule,
)


def test_monitor_scores_use_train_scalers_and_current_state():
    signals=compute_monitoring_signals(
        np.zeros(8),np.ones(7),np.zeros(7),np.full(7,0.5),
        target_scale=np.ones(7),input_mean=np.zeros(8),input_scale=np.ones(8),
        train_states_scaled=np.vstack([np.zeros(8),np.ones(8)]))
    assert signals.gp_uncertainty == pytest.approx(0.5)
    assert signals.gp_vla_disagreement == pytest.approx(1.0)
    assert signals.state_distance == pytest.approx(0.0)


def test_monitor_rejects_contract_errors():
    with pytest.raises(ValueError,match="8-vector"):
        compute_monitoring_signals(np.zeros(7),np.ones(7),np.zeros(7),np.ones(7),
            target_scale=np.ones(7),input_mean=np.zeros(8),input_scale=np.ones(8),
            train_states_scaled=np.zeros((1,8)))


def test_trigger_threshold_is_inclusive_and_only_at_chunk_boundary():
    s=MonitoringSignals(.8,.9,1.2); thresholds={"gp_uncertainty":.8}
    d=decide_assistance("gp_uncertainty",s,thresholds,episode_id="e",boundary_index=4)
    assert d.requested and d.request.score==.8
    queued=decide_assistance("gp_uncertainty",s,thresholds,episode_id="e",boundary_index=5,is_chunk_boundary=False)
    assert not queued.requested and queued.reason=="queued_action_not_boundary"
    assert not decide_assistance("passive_monitoring",s,{},episode_id="e",boundary_index=0).requested
    assert not decide_assistance("autonomy_only",s,{},episode_id="e",boundary_index=0).requested


def test_random_matched_schedule_is_deterministic_and_exact_budget():
    a=random_matched_schedule(max_boundaries=20,budget=4,seed=9)
    assert a==random_matched_schedule(max_boundaries=20,budget=4,seed=9)
    assert len(a)==4 and all(0<=x<20 for x in a)


class FakeProvider:
    name="synthetic-test-provider"
    def provide_correction(self,request,current_observation):
        assert request.request_id=="ep:chunk2:gp_uncertainty"
        assert current_observation=={"current":"only"}
        return np.full(7,.25)


def _request():
    req=AssistanceRequest("ep",2,"gp_uncertainty",.9,.8,"ep:chunk2:gp_uncertainty")
    return type("Decision",(),{"request":req,"requested":True})()


def test_passive_never_changes_action_even_if_provider_is_supplied():
    action=np.full(7,.1,dtype=np.float32);budget=CorrectionBudget(1)
    result=arbitrate_action(action,_request(),provider=FakeProvider(),
        current_observation={"current":"only"},budget=budget,active_enabled=False)
    np.testing.assert_array_equal(result.action,action)
    assert result.source=="pi05" and not result.correction_executed and budget.used==0


def test_active_arbitration_requires_provider_and_budget():
    with pytest.raises(RuntimeError,match="provider"):
        arbitrate_action(np.zeros(7),_request(),active_enabled=True,budget=CorrectionBudget(1))
    with pytest.raises(RuntimeError,match="budget"):
        arbitrate_action(np.zeros(7),_request(),active_enabled=True,provider=FakeProvider())


def test_active_arbitration_records_action_source_and_enforces_budget():
    budget=CorrectionBudget(1)
    result=arbitrate_action(np.zeros(7),_request(),active_enabled=True,provider=FakeProvider(),
        current_observation={"current":"only"},budget=budget)
    np.testing.assert_array_equal(result.action,np.full(7,.25))
    assert result.correction_executed and result.source=="synthetic-test-provider" and budget.used==1
    again=arbitrate_action(np.zeros(7),_request(),active_enabled=True,provider=FakeProvider(),
        current_observation={"current":"only"},budget=budget)
    assert not again.correction_executed and again.source=="pi05"
    assert again.reason=="correction_budget_exhausted"


def test_invalid_correction_is_rejected_without_spending_budget():
    class Invalid(FakeProvider):
        def provide_correction(self,*args):return np.full(7,np.nan)
    budget=CorrectionBudget(1)
    with pytest.raises(ValueError,match="correction provider"):
        arbitrate_action(np.zeros(7),_request(),active_enabled=True,provider=Invalid(),
            current_observation={},budget=budget)
    assert budget.used==0


def test_random_schedule_factory_is_deterministic_per_episode():
    factory=lambda episode_id: random_matched_schedule(
        max_boundaries=28,budget=2,seed=20261104+sum(episode_id.encode("utf-8")))
    signals=MonitoringSignals(0.1,0.2,0.3)
    schedule=factory("task4:seed1202")
    first=decide_assistance("random_matched_budget",signals,{},episode_id="task4:seed1202",
                            boundary_index=min(schedule),random_schedule=factory)
    assert first.requested
    assert first.request.episode_id=="task4:seed1202"
    repeated=decide_assistance("random_matched_budget",signals,{},episode_id="task4:seed1202",
                               boundary_index=min(schedule),random_schedule=factory)
    assert repeated==first

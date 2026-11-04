"""Current-chunk monitoring and explicit assistance arbitration primitives.

Signals are diagnostic only. They do not change actions unless an experiment
explicitly enables arbitration and supplies a correction provider.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Any
import numpy as np


MODES = {"autonomy_only", "passive_monitoring", "gp_uncertainty",
         "gp_vla_disagreement", "state_distance", "random_matched_budget"}


@dataclass(frozen=True)
class MonitoringSignals:
    gp_uncertainty: float
    gp_vla_disagreement: float
    state_distance: float

    def as_dict(self) -> dict[str, float]:
        return {"gp_uncertainty_standardized_rms": self.gp_uncertainty,
                "gp_vla_disagreement_standardized_rms": self.gp_vla_disagreement,
                "nearest_training_state_distance": self.state_distance}


def compute_monitoring_signals(state, vla_action, gp_mean, gp_std, *,
                               target_scale, input_mean, input_scale,
                               train_states_scaled) -> MonitoringSignals:
    """Compute three current-boundary scores with train-fitted scaling only."""
    x=np.asarray(state,dtype=np.float64)
    a=np.asarray(vla_action,dtype=np.float64)
    mu=np.asarray(gp_mean,dtype=np.float64)
    std=np.asarray(gp_std,dtype=np.float64)
    ys=np.asarray(target_scale,dtype=np.float64)
    xm=np.asarray(input_mean,dtype=np.float64)
    xs=np.asarray(input_scale,dtype=np.float64)
    train=np.asarray(train_states_scaled,dtype=np.float64)
    if x.shape!=(8,) or not np.isfinite(x).all():raise ValueError("state must be a finite normalized 8-vector")
    if a.shape!=(7,) or mu.shape!=(7,) or std.shape!=(7,):raise ValueError("action, GP mean and std must be 7-vectors")
    if not all(np.isfinite(v).all() for v in (a,mu,std,ys,xm,xs,train)):raise ValueError("monitor inputs must be finite")
    if np.any(std<=0) or ys.shape!=(7,) or np.any(ys<=0):raise ValueError("GP std and target scales must be positive 7-vectors")
    if xm.shape!=(8,) or xs.shape!=(8,) or np.any(xs<=0):raise ValueError("input scaler must have positive eight-feature scales")
    if train.ndim!=2 or train.shape[1]!=8 or train.shape[0]==0:raise ValueError("training states must have shape (N,8)")
    uncertainty=float(np.sqrt(np.mean(np.square(std/ys))))
    disagreement=float(np.sqrt(np.mean(np.square((mu-a)/ys))))
    scaled=(x-xm)/xs
    distance=float(np.linalg.norm(train-scaled[None,:],axis=1).min())
    return MonitoringSignals(uncertainty,disagreement,distance)


@dataclass(frozen=True)
class AssistanceRequest:
    episode_id: str
    boundary_index: int
    signal_name: str
    score: float | None
    threshold: float | None
    request_id: str


@dataclass(frozen=True)
class TriggerDecision:
    mode: str
    requested: bool
    reason: str
    request: AssistanceRequest | None


def decide_assistance(mode: str, signals: MonitoringSignals, thresholds: dict[str,float], *,
                      episode_id: str, boundary_index: int, is_chunk_boundary: bool = True,
                      random_schedule: frozenset[int] | set[int] | None = None) -> TriggerDecision:
    """Decide from this boundary's signals only; no outcome/future fields are accepted."""
    if mode not in MODES:raise ValueError(f"unsupported assistance mode: {mode}")
    if boundary_index<0:raise ValueError("boundary_index must be non-negative")
    if not is_chunk_boundary:return TriggerDecision(mode,False,"queued_action_not_boundary",None)
    selected=None;threshold=None;score=None
    if mode=="autonomy_only":return TriggerDecision(mode,False,"autonomy_only",None)
    if mode=="passive_monitoring":return TriggerDecision(mode,False,"passive_only",None)
    if mode=="random_matched_budget":
        selected="random_matched_budget"
        if callable(random_schedule):
            random_schedule = random_schedule(str(episode_id))
        requested=random_schedule is not None and boundary_index in random_schedule
        reason="scheduled_random_request" if requested else "not_selected_by_random_schedule"
    else:
        key={"gp_uncertainty":"gp_uncertainty","gp_vla_disagreement":"gp_vla_disagreement",
             "state_distance":"state_distance"}[mode]
        if key not in thresholds:raise ValueError(f"missing validation-frozen threshold: {key}")
        selected=key;threshold=float(thresholds[key]);score=float(getattr(signals,key))
        requested=score>=threshold
        reason="score_at_or_above_threshold" if requested else "below_threshold"
    request=None
    if requested:
        request=AssistanceRequest(str(episode_id),int(boundary_index),selected,score,threshold,
            f"{episode_id}:chunk{boundary_index}:{selected}")
    return TriggerDecision(mode,bool(requested),reason,request)


def random_matched_schedule(*, max_boundaries: int, budget: int, seed: int) -> frozenset[int]:
    """Precompute exactly budget distinct boundary indices without outcome access."""
    if max_boundaries<1 or not 0<=budget<=max_boundaries:raise ValueError("invalid boundary count or budget")
    if budget==0:return frozenset()
    return frozenset(int(x) for x in np.random.default_rng(seed).choice(max_boundaries,size=budget,replace=False))


class CorrectionProvider(Protocol):
    """Provider receives the request and current observation snapshot only."""
    name: str
    def provide_correction(self, request: AssistanceRequest, current_observation: Any) -> np.ndarray: ...


@dataclass
class CorrectionBudget:
    maximum: int
    used: int = 0
    def __post_init__(self):
        if self.maximum<0 or self.used<0 or self.used>self.maximum:raise ValueError("invalid correction budget")
    @property
    def remaining(self):return self.maximum-self.used
    def consume(self):
        if self.remaining<=0:return False
        self.used+=1
        return True


@dataclass(frozen=True)
class ActionSelection:
    action: np.ndarray
    source: str
    correction_executed: bool
    request_id: str | None
    reason: str


def arbitrate_action(vla_action, decision: TriggerDecision, *, active_enabled: bool = False,
                     provider: CorrectionProvider | None = None, current_observation: Any = None,
                     budget: CorrectionBudget | None = None) -> ActionSelection:
    """Keep VLA control unless active mode, a request, provider and budget all exist."""
    vla=np.asarray(vla_action,dtype=np.float32)
    if vla.shape!=(7,) or not np.isfinite(vla).all() or np.any(np.abs(vla)>1):
        raise ValueError("VLA action must be a finite in-range seven-vector")
    req=decision.request
    if not active_enabled or not decision.requested:
        return ActionSelection(vla.copy(),"pi05",False,req.request_id if req else None,
                               "assistance_disabled" if not active_enabled else "no_request")
    if provider is None:raise RuntimeError("active assistance requires an explicit correction provider")
    if budget is None:raise RuntimeError("active assistance requires an explicit correction budget")
    if budget.remaining<=0:
        return ActionSelection(vla.copy(),"pi05",False,req.request_id,"correction_budget_exhausted")
    corrected=np.asarray(provider.provide_correction(req,current_observation),dtype=np.float32)
    if corrected.shape!=(7,) or not np.isfinite(corrected).all() or np.any(np.abs(corrected)>1):
        raise ValueError("correction provider must return a finite in-range seven-vector")
    if not budget.consume():return ActionSelection(vla.copy(),"pi05",False,req.request_id,"correction_budget_exhausted")
    return ActionSelection(corrected.copy(),str(provider.name),True,req.request_id,"correction_applied")

"""Safe chunk-boundary arbitration between pi0.5 and human teleoperation."""
from __future__ import annotations
from dataclasses import dataclass
import threading
import uuid

@dataclass(frozen=True)
class HandoffEvent:
    source: str
    request: dict | None
    chunk_id: int
    chunk_index: int

class SharedAutonomyProtocol:
    """State machine; simulator calls remain owned by one thread."""
    HUMAN="HUMAN"
    AUTONOMY="AUTONOMY"
    REQUESTED="ASSISTANCE_REQUESTED"
    ESTOP="ESTOP"
    FINISHED="FINISHED"

    def __init__(self,policy):
        self.policy=policy
        self.state=self.HUMAN
        self._manual_pending=False
        self._manual_request_id=None
        self._manual_origin=None
        self._lock=threading.RLock()
        self.transition_index=0

    def start_autonomy(self):
        with self._lock:
            if self.state!=self.HUMAN: raise RuntimeError(f"cannot start autonomy from {self.state}")
            self.policy.reset()
            self.state=self.AUTONOMY
            self._manual_pending=False
            self._manual_request_id=self._manual_origin=None
            self.transition_index+=1
            return self.state

    def request_human(self,request_id=None,origin="human_operator"):
        with self._lock:
            if self.state!=self.AUTONOMY: raise RuntimeError(f"cannot request human control from {self.state}")
            if origin not in {"human_operator","integration_preflight","integration_test"}:
                raise ValueError("unsupported request origin")
            if self._manual_pending:
                return {"accepted":True,"pending_chunk_boundary":True,"already_pending":True,
                        "request_id":self._manual_request_id,"request_origin":self._manual_origin}
            self._manual_request_id=str(request_id or uuid.uuid4())
            self._manual_origin=origin
            self._manual_pending=True
            return {"accepted":True,"pending_chunk_boundary":True,"already_pending":False,
                    "request_id":self._manual_request_id,"request_origin":origin}

    def at_chunk_boundary(self,proposal):
        with self._lock:
            if self.state!=self.AUTONOMY or not bool(proposal.is_chunk_start): return None
            decision=proposal.assistance_request or {}
            gp_request=decision.get("request") if decision.get("requested") else None
            if not gp_request and not self._manual_pending: return None
            if gp_request:
                source="gp_trigger";request=gp_request
            else:
                source="manual_operator_request"
                request={"request_id":self._manual_request_id,"source":source,"request_origin":self._manual_origin}
            self.policy.reset()
            self._manual_pending=False
            self._manual_request_id=self._manual_origin=None
            self.state=self.REQUESTED
            self.transition_index+=1
            return HandoffEvent(source,request,int(proposal.chunk_id),int(proposal.chunk_index))

    def accept_human(self):
        with self._lock:
            if self.state!=self.REQUESTED: raise RuntimeError(f"cannot accept human control from {self.state}")
            self.state=self.HUMAN
            self.transition_index+=1
            return self.state

    def resume_autonomy(self):
        with self._lock:
            if self.state!=self.HUMAN: raise RuntimeError(f"cannot resume autonomy from {self.state}")
            self.policy.reset()
            self.state=self.AUTONOMY
            self._manual_pending=False
            self._manual_request_id=self._manual_origin=None
            self.transition_index+=1
            return self.state

    def finish(self):
        with self._lock:
            self.policy.reset()
            self._manual_pending=False
            self.state=self.FINISHED
            self.transition_index+=1
            return self.state

    def emergency_stop(self):
        with self._lock:
            self.policy.reset()
            self._manual_pending=False
            self.state=self.ESTOP
            self.transition_index+=1
            return self.state

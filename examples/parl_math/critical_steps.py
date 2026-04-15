"""Critical-path token-step tracker for nested / multi-stage agent rollouts.

K2.5 PARL defines:
    CriticalSteps = Σ_t (S_main^(t) + max_i S_sub,i^(t))

which is the wall-clock-equivalent token count along the longest path of a DAG
of generation calls: parallel calls within a stage take the max, sequential
stages sum. This module generalizes the 2-stage hardcoded form so the same
tracker handles arbitrary fan-out, multi-round orchestration, and dynamic
subagent counts without changes to the agent loop.

Usage:
    tracker = CriticalPathTracker()
    tracker.begin_stage()
    for s in solvers:
        tracker.record(s.response_length)
    tracker.begin_stage()
    tracker.record(orch.response_length)
    total, per_stage = tracker.finalize()
"""

from __future__ import annotations


class CriticalPathTracker:
    __slots__ = ("_stages", "_open")

    def __init__(self) -> None:
        self._stages: list[list[int]] = []
        self._open: list[int] | None = None

    def begin_stage(self) -> None:
        if self._open is not None:
            self._stages.append(self._open)
        self._open = []

    def record(self, response_length: int) -> None:
        if self._open is None:
            self._open = []
        self._open.append(max(0, int(response_length)))

    def finalize(self) -> tuple[int, list[int]]:
        if self._open is not None:
            self._stages.append(self._open)
            self._open = None
        per_stage = [(max(s) if s else 0) for s in self._stages]
        return sum(per_stage), per_stage

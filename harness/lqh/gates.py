"""Gate registry (harness/config/gates.json): stable gate ids, definition versions, thresholds, prerequisites and
evidence ids. A status may only be recorded against a gate id of the registry; a PASS from another numbering (plan 5's
G2-G6, a review's G2-G6) is refused, never mapped onto the v6 gate of the same number."""
from __future__ import annotations

from .util import harness_root, read_json


class GateError(ValueError):
    pass


def load() -> dict:
    return read_json(harness_root() / "config" / "gates.json")


def record(gate_id: str, status: str, evidence_ids: list[str], registry: dict | None = None) -> dict:
    registry = registry or load()
    if gate_id in registry.get("rejected_numberings", []) or gate_id.split(".")[0] in ("plan5", "review2", "p5"):
        raise GateError(f"{gate_id}: another numbering; its status cannot be imported into the v6 gates")
    gate = next((g for g in registry["gates"] if g["gate_id"] == gate_id), None)
    if gate is None:
        raise GateError(f"{gate_id}: not in the registry (a new gate needs a new definition_version)")
    if status in ("PASS", "MET") and not evidence_ids:
        raise GateError(f"{gate_id}: a pass needs evidence ids")
    unmet = [p for p in gate.get("prerequisites", []) if p.startswith("v6.")
             and next((g for g in registry["gates"] if g["gate_id"] == p), {}).get("status") not in ("PASS", "MET")]
    if status in ("PASS", "MET") and unmet:
        raise GateError(f"{gate_id}: prerequisites not passed: {unmet}")
    return {**gate, "status": status, "evidence_ids": evidence_ids}

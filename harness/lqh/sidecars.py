"""Figure sidecar rules (prompt rev. 3, section 16): every plotted point carries its value, unit, source row,
originating experiment, candidate, hardware provenance, measurement boundary, comparison class, repetition count
(where it applies), evidence kind and derivation; error-type metrics state their norm; an unmeasured point is
listed, never plotted (and never as zero)."""
from __future__ import annotations

REQUIRED = ("figure_id", "candidate_id", "originating_experiment", "hardware", "measurement_boundary",
            "comparison_class", "evidence_kind", "derivation")


def sidecar_problems(sidecar: dict) -> list[str]:
    problems = []
    if not sidecar.get("generator"):
        problems.append("no generator version")
    if not sidecar.get("data_sha256"):
        problems.append("no data hash")
    for index, point in enumerate(sidecar.get("points", [])):
        where = f"point {index}"
        for key in REQUIRED:
            if point.get(key) in (None, ""):
                problems.append(f"{where}: missing {key}")
        if not (point.get("run_id") or point.get("source_row_id")):
            problems.append(f"{where}: no source row")
        if not any(k.endswith("_unit") for k in point):
            problems.append(f"{where}: no unit")
        axis_values = [point[k] for k in point if k.startswith(("x_", "y_")) and not k.endswith("_unit")]
        plotted = point.get("plotted")
        if plotted and any(v is None for v in axis_values):
            problems.append(f"{where}: plotted with a null value")
        if plotted is False and point.get("plotted_as_zero"):
            problems.append(f"{where}: an unmeasured point plotted as zero")
        if "error" in str(point.get("metric", "")).lower() and not point.get("error_norm"):
            problems.append(f"{where}: error metric without a stated norm")
    return problems

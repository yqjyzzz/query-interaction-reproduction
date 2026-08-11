#!/usr/bin/env python3
"""Export analysis-ready rows for the TMLR replication package.

This script reads only the frozen non-V/F artifacts used by H4-D, Gate C,
T1, and T2. It does not run a model or modify the source project.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


MODELS = ("detr_r50_500", "dino_r50_4s_12e")
GATE_C_DOSES = (0.0, 0.25, 0.5, 0.75, 1.0)
T2_DOSES = ("0.25", "0.5", "0.75")
T1_FIELDS = (
    "local",
    "focal",
    "spillover",
    "fixed",
    "matching",
    "rematched",
    "selection",
    "native",
)
T2_ENDPOINTS = ("local", "spillover", "fixed")


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def saved_assignment_utility(
    quality_matrix: list[list[float]], assignment: list[int]
) -> float:
    if not assignment:
        return 0.0
    return (
        sum(
            float(quality_matrix[prediction][gt])
            for gt, prediction in enumerate(assignment)
            if prediction >= 0
        )
        / len(assignment)
    )


def h4_fast_effect(receipt: dict[str, Any]) -> dict[str, float]:
    """Equivalent H4 decomposition using assignments saved in the receipt."""

    sham = receipt["arm_measurements"]["reverse_sham"]
    intervention = receipt["arm_measurements"]["targeted_reverse"]
    native = receipt["native_measurements"]["targeted_reverse"]
    assignment = [int(item) for item in sham["optimal_assignment"]]
    focal = int(receipt["pair"]["competitor"])
    u0 = float(sham["image_set_utility"])
    u1 = (
        sum(
            float(
                intervention["quality_matrix"][prediction][gt]
                if prediction == focal
                else sham["quality_matrix"][prediction][gt]
            )
            for gt, prediction in enumerate(assignment)
            if prediction >= 0
        )
        / len(assignment)
        if assignment
        else 0.0
    )
    u2 = saved_assignment_utility(intervention["quality_matrix"], assignment)
    u3 = float(intervention["image_set_utility"])
    u4 = float(native["image_set_utility"])
    components = {
        "focal": u1 - u0,
        "spillover": u2 - u1,
        "matching": u3 - u2,
        "selection": u4 - u3,
    }
    total = u4 - u0
    closure = total - sum(components.values())
    baseline = receipt["arm_measurements"]["baseline"]
    random_edge = receipt["arm_measurements"]["random_edge"]
    different = receipt["arm_measurements"]["different_object"]
    return {
        "delta_local": float(intervention["fixed_competitor_quality"])
        - float(sham["fixed_competitor_quality"]),
        "delta_target": float(intervention["target_set_quality"])
        - float(sham["target_set_quality"]),
        "delta_image_fixed": u2 - u0,
        "delta_image_rematched": u3 - u0,
        "delta_native": total,
        "delta_focal": components["focal"],
        "delta_spillover": components["spillover"],
        "delta_matching": components["matching"],
        "delta_selection": components["selection"],
        "closure_error": closure,
        "random_control_local": float(random_edge["fixed_competitor_quality"])
        - float(baseline["fixed_competitor_quality"]),
        "different_control_local": float(different["fixed_competitor_quality"])
        - float(baseline["fixed_competitor_quality"]),
    }


def export_h4_d(project: Path, output: Path, h4: Any) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for model in MODELS:
        rows = []
        paths = sorted((project / f"artifacts/iclr_h4/d_raw/{model}").glob("*.json"))
        for path in paths:
            receipt = load(path)
            rows.append(
                {
                    "image_id": receipt["image_id"],
                    "image_sha256": receipt["image_sha256"],
                    "stratum": receipt["stratum"],
                    **h4_fast_effect(receipt),
                }
            )
        target = output / "analysis_ready" / f"h4_d_{model}.jsonl"
        write_jsonl(target, rows)
        report[model] = {
            "rows": len(rows),
            "path": str(target.relative_to(output)).replace("\\", "/"),
            "sha256": sha256(target),
        }
    return report


def gate_c_fast_effect(
    receipt: dict[str, Any], kind: str, dose: float
) -> dict[str, float]:
    """Equivalent Gate C endpoints using assignments/utilities saved in receipts.

    The frozen raw receipt already stores the deterministic optimal assignment
    and its image-set utility for baseline, changed, and native measurements.
    Reusing those values avoids repeating 900x900 Hungarian solves.
    """

    token = str(dose).replace(".", "p")
    arm_name = f"{kind}_alpha_{token}"
    baseline = receipt["arm_measurements"]["baseline"]
    changed = receipt["arm_measurements"][arm_name]
    native = receipt["native_measurements"][arm_name]
    u0 = float(baseline["image_set_utility"])
    u3 = float(changed["image_set_utility"])
    u4 = float(native["image_set_utility"])
    return {
        "local": float(changed["fixed_competitor_quality"])
        - float(baseline["fixed_competitor_quality"]),
        "target": float(changed["target_set_quality"])
        - float(baseline["target_set_quality"]),
        "image": u3 - u0,
        "native": u4 - u0,
        "closure": 0.0,
    }


def export_gate_c(project: Path, output: Path, gate_c: Any) -> dict[str, Any]:
    population = load(project / "artifacts/iclr_h4/H4_D_FULL_GT_MANIFEST.json")
    stratum_by_id = {row["image_id"]: row["stratum"] for row in population["rows"]}
    report: dict[str, Any] = {}
    for model in MODELS:
        rows = []
        paths = sorted(
            (project / f"artifacts/iclr_h4/d_gate_c_raw_v2/{model}").glob("*.json")
        )
        for path in paths:
            receipt = load(path)
            effects = {}
            for dose in GATE_C_DOSES:
                effects[str(dose)] = {
                    "target": gate_c_fast_effect(receipt, "target", dose),
                    "control": gate_c_fast_effect(receipt, "control", dose),
                }
            rows.append(
                {
                    "image_id": receipt["image_id"],
                    "image_sha256": receipt["image_sha256"],
                    "stratum": stratum_by_id[receipt["image_id"]],
                    "effects": effects,
                }
            )
        target = output / "analysis_ready" / f"gate_c_{model}.jsonl"
        write_jsonl(target, rows)
        report[model] = {
            "rows": len(rows),
            "path": str(target.relative_to(output)).replace("\\", "/"),
            "sha256": sha256(target),
        }
    return report


def export_t1(project: Path, output: Path, t1: Any) -> dict[str, Any]:
    mapping = load(project / "artifacts/iclr_transport/T1_REALIZED_DOSE_MAPPING.json")
    report: dict[str, Any] = {}
    for model in MODELS:
        hard = {
            path.stem: load(path)
            for path in sorted((project / f"artifacts/iclr_h4/d_raw/{model}").glob("*.json"))
        }
        mass = {
            path.stem: load(path)
            for path in sorted(
                (project / f"artifacts/iclr_h4/d_gate_c_raw_v2/{model}").glob("*.json")
            )
        }
        rows = []
        for mapped in mapping["models"][model]["rows"]:
            image_id = mapped["image_id"]
            hard_effect = t1.effect(hard[image_id], "targeted_reverse")
            dose_name = str(mapped["selected_mass_dose"]).replace(".", "p")
            mass_effect = t1.effect(mass[image_id], f"target_alpha_{dose_name}")
            contrast = {
                field: hard_effect[field] - mass_effect[field] for field in T1_FIELDS
            }
            rows.append(
                {
                    "image_id": image_id,
                    "image_sha256": mapped["image_sha256"],
                    "stratum": mapped["stratum"],
                    "selected_mass_dose": mapped["selected_mass_dose"],
                    **contrast,
                }
            )
        target = output / "analysis_ready" / f"t1_{model}.jsonl"
        write_jsonl(target, rows)
        report[model] = {
            "rows": len(rows),
            "path": str(target.relative_to(output)).replace("\\", "/"),
            "sha256": sha256(target),
        }
    return report


def export_t2(project: Path, output: Path, t2: Any) -> dict[str, Any]:
    mapping = load(project / "artifacts/iclr_transport/T2_OUTCOME_BLIND_MAPPING.json")
    raw = project / "artifacts/iclr_transport/t2_confirmation_raw_initial_v2"
    input_root = project / "artifacts/iclr_transport/t2_input_capture_raw"
    report: dict[str, Any] = {}
    for model in MODELS:
        mapped = {row["image_id"]: row for row in mapping["models"][model]["rows"]}
        rows = []
        for path in sorted((raw / model).glob("*.json")):
            receipt = load(path)
            input_receipt = load(input_root / model / path.name)
            map_row = mapped[path.stem]
            effects: dict[str, list[dict[str, float]]] = {"M": [], "P": [], "D": []}
            for dose in T2_DOSES:
                token = dose.replace(".", "p")
                effects["M"].append(t2.primary_effect(receipt, f"C_v2_{token}"))
                a_dose = str(map_row["selected_doses"][f"A_for_M_{dose}"])
                b_dose = str(map_row["selected_doses"][f"B_for_M_{dose}"])
                effects["P"].append(
                    t2.primary_effect(receipt, f"A_{a_dose.replace('.', 'p')}")
                )
                effects["D"].append(
                    t2.primary_effect(receipt, f"B_{b_dose.replace('.', 'p')}")
                )
            averaged = {
                operator: {
                    endpoint: sum(item[endpoint] for item in values) / len(values)
                    for endpoint in T2_ENDPOINTS
                }
                for operator, values in effects.items()
            }
            hard = t2.primary_effect(receipt, "hard")
            p_curves = [
                t2.primary_effect(receipt, f"A_{dose.replace('.', 'p')}")
                for dose in T2_DOSES
            ]
            metrics = input_receipt["realized_metrics"]
            p_norms = [
                sum(metrics["A"][dose]["immediate_message_norm"])
                / len(metrics["A"][dose]["immediate_message_norm"])
                for dose in T2_DOSES
            ]
            h_norm = (
                sum(metrics["C_v2"]["0.75"]["immediate_message_norm"])
                / len(metrics["C_v2"]["0.75"]["immediate_message_norm"])
                / 0.75
            )
            rows.append(
                {
                    "image_id": path.stem,
                    "image_sha256": map_row["image_sha256"],
                    "stratum": map_row["stratum"],
                    "selected_doses": map_row["selected_doses"],
                    "averaged_effects": averaged,
                    "hard_effect": hard,
                    "p_curve_effects": p_curves,
                    "p_immediate_message_norms": p_norms,
                    "h_immediate_message_norm": h_norm,
                }
            )
        target = output / "analysis_ready" / f"t2_{model}.jsonl"
        write_jsonl(target, rows)
        report[model] = {
            "rows": len(rows),
            "path": str(target.relative_to(output)).replace("\\", "/"),
            "sha256": sha256(target),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    project = args.project_root.resolve()
    output = args.output_root.resolve()
    sys.path.insert(0, str(project / "src"))

    h4 = load_module("p1_h4", project / "tools/iclr_h4_aggregate.py")
    gate_c = load_module("p1_gate_c", project / "tools/iclr_h4_gate_c_aggregate.py")
    t1 = load_module("p1_t1", project / "tools/iclr_transport_t1_aggregate.py")
    t2 = load_module("p1_t2", project / "tools/iclr_transport_t2_aggregate.py")

    report = {
        "schema_version": "tmlr.p1.analysis_ready_export.v1",
        "project_root_used_for_export": "<frozen-source-project>",
        "V_F_read": False,
        "new_model_inference": False,
        "source_scripts": {
            "h4_d": {
                "path": "tools/iclr_h4_aggregate.py",
                "sha256": sha256(project / "tools/iclr_h4_aggregate.py"),
            },
            "gate_c": {
                "path": "tools/iclr_h4_gate_c_aggregate.py",
                "sha256": sha256(project / "tools/iclr_h4_gate_c_aggregate.py"),
            },
            "t1": {
                "path": "tools/iclr_transport_t1_aggregate.py",
                "sha256": sha256(project / "tools/iclr_transport_t1_aggregate.py"),
            },
            "t2": {
                "path": "tools/iclr_transport_t2_aggregate.py",
                "sha256": sha256(project / "tools/iclr_transport_t2_aggregate.py"),
            },
        },
        "exports": {
            "h4_d": export_h4_d(project, output, h4),
            "gate_c": export_gate_c(project, output, gate_c),
            "t1": export_t1(project, output, t1),
            "t2": export_t2(project, output, t2),
        },
    }
    manifest = output / "analysis_ready" / "ANALYSIS_READY_PROVENANCE.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

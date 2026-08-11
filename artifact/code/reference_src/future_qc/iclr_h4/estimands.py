"""Pure H4 estimands.

This module does not load manifests, model outputs, or receipts.  It converts
already-computed prediction-by-GT quality matrices into the preregistered
fixed-assignment, rematched, and telescoping image-set measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


class H4EstimandError(RuntimeError):
    """Raised when a measurement cannot satisfy the frozen H4 contract."""


def _matrix(value: Sequence[Sequence[float]], *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2:
        raise H4EstimandError(f"{name} must be a rank-2 matrix")
    if result.shape[1] < 1:
        raise H4EstimandError(f"{name} must contain at least one GT column")
    if not np.isfinite(result).all():
        raise H4EstimandError(f"{name} contains non-finite values")
    if (result < 0.0).any() or (result > 1.0).any():
        raise H4EstimandError(f"{name} must remain on the frozen [0,1] scale")
    return result


@dataclass(frozen=True)
class Assignment:
    """Injective prediction-to-GT assignment; missing GTs use prediction -1."""

    prediction_for_gt: tuple[int, ...]


def optimal_assignment(quality: Sequence[Sequence[float]]) -> Assignment:
    """Return the deterministic maximum-sum injective assignment.

    Zero-valued dummy predictions are added when predictions are fewer than
    GT objects.  A tiny index-only tie break makes exact ties reproducible
    without changing any non-tied scientific value.
    """

    matrix = _matrix(quality, name="quality")
    prediction_count, gt_count = matrix.shape
    size = max(prediction_count, gt_count)
    padded = np.zeros((size, size), dtype=np.float64)
    padded[:prediction_count, :gt_count] = matrix
    row_ids = np.arange(size, dtype=np.float64)[:, None]
    col_ids = np.arange(size, dtype=np.float64)[None, :]
    tie_break = (row_ids * size + col_ids) * np.finfo(np.float64).eps
    rows, columns = linear_sum_assignment(-(padded - tie_break))
    by_gt = [-1] * gt_count
    for row, column in zip(rows.tolist(), columns.tolist()):
        if column < gt_count and row < prediction_count:
            by_gt[column] = row
    return Assignment(tuple(by_gt))


def assignment_utility(
    quality: Sequence[Sequence[float]],
    assignment: Assignment,
) -> float:
    """Mean p(class)*IoU over GTs, with unmatched GT contribution fixed at 0."""

    matrix = _matrix(quality, name="quality")
    if len(assignment.prediction_for_gt) != matrix.shape[1]:
        raise H4EstimandError("assignment GT count mismatch")
    used: set[int] = set()
    total = 0.0
    for gt_id, prediction_id in enumerate(assignment.prediction_for_gt):
        if prediction_id == -1:
            continue
        if not 0 <= prediction_id < matrix.shape[0]:
            raise H4EstimandError("assignment prediction index outside matrix")
        if prediction_id in used:
            raise H4EstimandError("assignment is not injective")
        used.add(prediction_id)
        total += float(matrix[prediction_id, gt_id])
    return total / matrix.shape[1]


def _replace_focal(
    sham: np.ndarray,
    intervention: np.ndarray,
    focal_prediction: int,
) -> np.ndarray:
    if sham.shape != intervention.shape:
        raise H4EstimandError("sham/intervention matrix shape mismatch")
    if not 0 <= focal_prediction < sham.shape[0]:
        raise H4EstimandError("focal prediction outside matrix")
    result = sham.copy()
    result[focal_prediction, :] = intervention[focal_prediction, :]
    return result


def image_set_decomposition(
    *,
    sham_quality: Sequence[Sequence[float]],
    intervention_quality: Sequence[Sequence[float]],
    focal_prediction: int,
    native_intervention_quality: Sequence[Sequence[float]] | None = None,
    closure_tolerance: float = 1e-12,
) -> dict[str, object]:
    """Compute the frozen U0..U4 H4 telescoping decomposition."""

    sham = _matrix(sham_quality, name="sham_quality")
    intervention = _matrix(intervention_quality, name="intervention_quality")
    focal_only = _replace_focal(sham, intervention, focal_prediction)
    sham_assignment = optimal_assignment(sham)
    intervention_assignment = optimal_assignment(intervention)
    native = (
        intervention
        if native_intervention_quality is None
        else _matrix(native_intervention_quality, name="native_intervention_quality")
    )
    if native.shape[1] != sham.shape[1]:
        raise H4EstimandError("native intervention GT count mismatch")
    native_assignment = optimal_assignment(native)

    u0 = assignment_utility(sham, sham_assignment)
    u1 = assignment_utility(focal_only, sham_assignment)
    u2 = assignment_utility(intervention, sham_assignment)
    u3 = assignment_utility(intervention, intervention_assignment)
    u4 = assignment_utility(native, native_assignment)
    components = {
        "focal": u1 - u0,
        "spillover": u2 - u1,
        "matching": u3 - u2,
        "selection": u4 - u3,
    }
    total = u4 - u0
    closure_error = total - sum(components.values())
    if abs(closure_error) > closure_tolerance:
        raise H4EstimandError(
            f"telescoping closure failed: {closure_error} > {closure_tolerance}"
        )
    return {
        "utilities": {"u0": u0, "u1": u1, "u2": u2, "u3": u3, "u4": u4},
        "components": components,
        "total": total,
        "closure_error": closure_error,
        "sham_assignment": list(sham_assignment.prediction_for_gt),
        "intervention_assignment": list(intervention_assignment.prediction_for_gt),
        "native_assignment": list(native_assignment.prediction_for_gt),
    }

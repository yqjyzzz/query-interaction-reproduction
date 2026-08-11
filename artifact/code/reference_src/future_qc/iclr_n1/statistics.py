"""Frozen N1 paired estimands, bootstrap, and robustness summaries."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from typing import Any


class N1StatisticsError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise N1StatisticsError(message)


def percentile(values: list[float], probability: float) -> float:
    require(values and 0.0 <= probability <= 1.0, "invalid percentile input")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def trimmed_mean(values: list[float], fraction_each_tail: float) -> float:
    require(values and 0.0 <= fraction_each_tail < 0.5, "invalid trimmed mean")
    ordered = sorted(float(value) for value in values)
    trim = math.floor(len(ordered) * fraction_each_tail)
    kept = ordered[trim : len(ordered) - trim] if trim else ordered
    require(kept, "trim removed all observations")
    return statistics.fmean(kept)


def winsorized_mean(values: list[float], lower: float, upper: float) -> float:
    require(values and 0.0 <= lower < upper <= 1.0, "invalid winsorization")
    floor_value, ceiling_value = percentile(values, lower), percentile(values, upper)
    return statistics.fmean(min(max(float(value), floor_value), ceiling_value) for value in values)


def direction(value: float, *, tolerance: float = 0.0) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def robustness_summary(
    values: list[float],
    *,
    expected_direction: int,
    trim_fraction: float = 0.1,
    winsor_limits: tuple[float, float] = (0.01, 0.99),
    minimum_same_direction_fraction: float = 0.55,
) -> dict[str, Any]:
    require(expected_direction in {-1, 1}, "expected direction must be signed")
    require(values and all(math.isfinite(float(value)) for value in values), "invalid robustness values")
    raw_mean = statistics.fmean(values)
    median = statistics.median(values)
    trimmed = trimmed_mean(values, trim_fraction)
    winsorized = winsorized_mean(values, *winsor_limits)
    leave_one_out = (
        [statistics.fmean(values[:index] + values[index + 1 :]) for index in range(len(values))]
        if len(values) > 1
        else [raw_mean]
    )
    same_direction = sum(direction(float(value)) == expected_direction for value in values) / len(values)
    absolute = sorted((abs(float(value)) for value in values), reverse=True)
    top_count = max(1, math.ceil(len(values) * 0.01))
    total_absolute = sum(absolute)
    top_share = sum(absolute[:top_count]) / total_absolute if total_absolute else 0.0
    signs = {
        "raw_mean": direction(raw_mean),
        "trimmed_mean": direction(trimmed),
        "winsorized_mean": direction(winsorized),
    }
    leave_one_out_stable = all(direction(value) == expected_direction for value in leave_one_out)
    return {
        "n": len(values),
        "raw_mean": raw_mean,
        "median": median,
        "trimmed_mean_10pct_each_tail": trimmed,
        "winsorized_mean_1_99": winsorized,
        "same_direction_image_fraction": same_direction,
        "leave_one_image_out_min": min(leave_one_out),
        "leave_one_image_out_max": max(leave_one_out),
        "leave_one_image_out_direction_stable": leave_one_out_stable,
        "top_one_percent_absolute_contribution_share": top_share,
        "expected_direction": expected_direction,
        "pass": (
            all(value == expected_direction for value in signs.values())
            and leave_one_out_stable
            and same_direction >= minimum_same_direction_fraction
        ),
    }


def stratified_bootstrap_mean(
    values: list[float],
    strata: list[str],
    *,
    draws: int,
    seed: str,
    confidence_level: float,
    expected_direction: int,
) -> dict[str, Any]:
    require(len(values) == len(strata) and values, "bootstrap input mismatch")
    require(draws >= 100 and 0.0 < confidence_level < 1.0, "invalid bootstrap settings")
    groups: dict[str, list[float]] = {}
    for value, stratum in zip(values, strata):
        require(math.isfinite(float(value)), "bootstrap non-finite value")
        groups.setdefault(str(stratum), []).append(float(value))
    seed_value = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")
    rng = random.Random(seed_value)
    means = []
    for _ in range(draws):
        sampled = [
            rng.choice(group)
            for stratum in sorted(groups)
            for group in [groups[stratum]]
            for _ in range(len(group))
        ]
        means.append(statistics.fmean(sampled))
    alpha = (1.0 - confidence_level) / 2.0
    probability = sum(direction(value) == expected_direction for value in means) / draws
    return {
        "kind": "stratified_image_percentile",
        "draws": draws,
        "seed": seed,
        "confidence_level": confidence_level,
        "lower": percentile(means, alpha),
        "upper": percentile(means, 1.0 - alpha),
        "expected_direction": expected_direction,
        "direction_probability": probability,
    }


def native_quality(arm: dict[str, Any]) -> float:
    value = arm["native_postprocess_target_summary"].get("set_best")
    return 0.0 if value is None else float(value["target_quality"])


def receipt_estimands(receipt: dict[str, Any]) -> dict[str, float]:
    """Compute only the preregistered paired contrasts for one receipt."""
    arms = receipt["arm_instrumentation"]
    reverse, reverse_sham = arms["targeted_reverse"], arms["reverse_sham"]
    direct, direct_sham = arms["targeted_direct"], arms["matched_sham"]
    reverse_fixed = (
        float(reverse["scalar_readout"]["fixed_competitor_quality"])
        - float(reverse_sham["scalar_readout"]["fixed_competitor_quality"])
    )
    reverse_set = (
        float(reverse["scalar_readout"]["set_quality"])
        - float(reverse_sham["scalar_readout"]["set_quality"])
    )
    return {
        "reverse_fixed": reverse_fixed,
        "reverse_set": reverse_set,
        "reverse_recovery": reverse_set - reverse_fixed,
        "reverse_owner_switch": float(reverse["owner_switch_from_baseline"])
        - float(reverse_sham["owner_switch_from_baseline"]),
        "reverse_competitor_rank_improvement": float(reverse_sham["rank_and_extinction"]["competitor_rank_one_based"])
        - float(reverse["rank_and_extinction"]["competitor_rank_one_based"]),
        "reverse_competitor_extinction": float(reverse["rank_and_extinction"]["competitor_extinct"])
        - float(reverse_sham["rank_and_extinction"]["competitor_extinct"]),
        "reverse_duplicate_pairs": float(reverse["duplicate_predictions"]["primary"]["duplicate_pair_count"])
        - float(reverse_sham["duplicate_predictions"]["primary"]["duplicate_pair_count"]),
        "reverse_native_postprocess": native_quality(reverse) - native_quality(reverse_sham),
        "direct_fixed": float(direct["scalar_readout"]["fixed_leader_quality"])
        - float(direct_sham["scalar_readout"]["fixed_leader_quality"]),
        "direct_set": float(direct["scalar_readout"]["set_quality"])
        - float(direct_sham["scalar_readout"]["set_quality"]),
        "random_control_fixed_vs_baseline": float(arms["random_edge"]["scalar_readout"]["fixed_leader_quality"])
        - float(arms["baseline"]["scalar_readout"]["fixed_leader_quality"]),
        "different_object_control_fixed_vs_baseline": float(
            arms["different_object"]["scalar_readout"]["fixed_leader_quality"]
        )
        - float(arms["baseline"]["scalar_readout"]["fixed_leader_quality"]),
    }

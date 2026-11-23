"""Statistics for paired run blocks, quantile uncertainty and paired quality comparisons. Standard library only.

The unit of uncertainty is an independent paired run block (baseline and candidate run back to back in balanced
order). Tokens or requests inside one run are NOT independent trials; reusing a table is not a repetition.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Sequence


# ---- Student t quantile (regularised incomplete beta, continued fraction) ------------------------------------
def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


def t_quantile(p: float, df: float) -> float:
    """Quantile of Student's t by bisection on the exact CDF."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def normal_quantile(p: float) -> float:
    """Acklam's rational approximation (abs error < 1.2e-9)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    low = 0.02425
    if p < low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - low:
        return -normal_quantile(1 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# ---- paired blocks -----------------------------------------------------------------------------------------
def paired_ratio(baseline: Sequence[float], candidate: Sequence[float], confidence: float = 0.95,
                 estimator: str = "geometric_mean") -> dict:
    """Per-block ratios candidate/baseline and a confidence interval.

    geometric_mean: t interval on log ratios (the headline estimator, symmetric for speedups and slowdowns).
    mean_of_ratios / ratio_of_means are reported alongside so they can never be confused.
    """
    if len(baseline) != len(candidate):
        raise ValueError("paired data must have the same number of blocks")
    pairs = [(b, c) for b, c in zip(baseline, candidate) if b is not None and c is not None]
    n = len(pairs)
    out: dict = {"n_blocks": n, "estimator": estimator, "confidence": confidence,
                 "ratios": [c / b for b, c in pairs] if n else []}
    if n == 0 or any(b <= 0 or c <= 0 for b, c in pairs):
        out.update(value=None, low=None, high=None, null_reason="no_positive_pairs" if n else "no_pairs")
        return out
    logs = [math.log(c / b) for b, c in pairs]
    mean_log = sum(logs) / n
    out["geometric_mean"] = math.exp(mean_log)
    out["mean_of_ratios"] = sum(out["ratios"]) / n
    out["ratio_of_means"] = (sum(c for _, c in pairs) / n) / (sum(b for b, _ in pairs) / n)
    out["median_ratio"] = sorted(out["ratios"])[n // 2] if n % 2 else \
        (sorted(out["ratios"])[n // 2 - 1] + sorted(out["ratios"])[n // 2]) / 2.0
    out["value"] = out[{"geometric_mean": "geometric_mean", "mean_of_ratios": "mean_of_ratios",
                        "ratio_of_means": "ratio_of_means", "median": "median_ratio"}[estimator]]
    if n < 2:
        out.update(low=None, high=None, interval_null_reason="fewer_than_two_blocks")
        return out
    sd = math.sqrt(sum((x - mean_log) ** 2 for x in logs) / (n - 1))
    half = t_quantile(0.5 + confidence / 2.0, n - 1) * sd / math.sqrt(n)
    out["low"], out["high"] = math.exp(mean_log - half), math.exp(mean_log + half)
    out["interval_method"] = f"t interval on log ratios, df={n - 1}"
    out["width_ratio"] = out["high"] / out["low"]
    return out


def bootstrap_ci(values: Sequence[float], statistic: Callable[[Sequence[float]], float], confidence: float = 0.95,
                 resamples: int = 10000, seed: int = 20261114) -> tuple[float | None, float | None]:
    """Percentile bootstrap over blocks (a cross-check for the t interval; deterministic seed)."""
    if len(values) < 2:
        return None, None
    rng = random.Random(seed)
    stats = sorted(statistic([rng.choice(values) for _ in values]) for _ in range(resamples))
    lo = stats[int(math.floor((1 - confidence) / 2 * resamples))]
    hi = stats[min(resamples - 1, int(math.ceil((1 + confidence) / 2 * resamples)) - 1)]
    return lo, hi


def quantile_interval(sorted_values: Sequence[float], p: float, confidence: float = 0.95) -> dict:
    """Distribution-free interval for the p-quantile from order statistics (binomial). WITHIN one run: requests and
    token gaps are not independent, so this shows sampling resolution, not between-run uncertainty."""
    n = len(sorted_values)
    if n == 0:
        return {"low": None, "high": None, "null_reason": "no_observations"}
    q = p / 100.0 if p > 1 else p
    z = normal_quantile(0.5 + confidence / 2.0)
    centre = n * q
    spread = z * math.sqrt(n * q * (1 - q))
    lo_rank = max(0, int(math.floor(centre - spread)) - 1)
    hi_rank = min(n - 1, int(math.ceil(centre + spread)))
    return {"low": sorted_values[lo_rank], "high": sorted_values[hi_rank], "n": n,
            "method": "order-statistic normal approximation, within-run",
            "coverage_note": "not a between-run interval"}


# ---- proportions and paired quality comparisons -------------------------------------------------------------
def wilson(successes: int, count: int, confidence: float = 0.95) -> tuple[float | None, float | None]:
    if count <= 0:
        return None, None
    z = normal_quantile(0.5 + confidence / 2.0)
    p = successes / count
    denom = 1 + z * z / count
    centre = (p + z * z / (2 * count)) / denom
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe_paired(both: int, only_candidate: int, only_baseline: int, neither: int,
                    confidence: float = 0.95) -> dict:
    """Interval for (candidate accuracy - baseline accuracy) on paired items: Newcombe's method 10 (hybrid score
    with the phi correction). With zero discordant pairs it does NOT collapse to 0..0, so a small sample stays
    inconclusive instead of passing."""
    n = both + only_candidate + only_baseline + neither
    if n == 0:
        return {"difference": None, "low": None, "high": None, "null_reason": "no_items"}
    p1 = (both + only_candidate) / n
    p2 = (both + only_baseline) / n
    l1, u1 = wilson(both + only_candidate, n, confidence)
    l2, u2 = wilson(both + only_baseline, n, confidence)
    a, b, c, d = both, only_candidate, only_baseline, neither
    denominator = (a + b) * (c + d) * (a + c) * (b + d)
    if denominator == 0:
        phi = 0.0
    else:
        numerator = a * d - b * c
        if numerator > 0:
            numerator = max(numerator - n / 2.0, 0.0)
        phi = numerator / math.sqrt(denominator)
    diff = p1 - p2
    dl = math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0.0))
    du = math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0.0))
    return {"difference": diff, "low": max(-1.0, diff - dl), "high": min(1.0, diff + du), "n": n,
            "discordant": b + c, "method": "Newcombe hybrid score (method 10) with phi correction"}


def noninferiority(low: float | None, high: float | None, margin: float) -> str:
    """PASS when the lower bound of (candidate - baseline) is above -margin; FAIL when the upper bound is below
    -margin; otherwise INCONCLUSIVE. "No significant difference" is never a pass."""
    if low is None or high is None:
        return "INCONCLUSIVE"
    if low > -margin:
        return "PASS"
    if high < -margin:
        return "FAIL"
    return "INCONCLUSIVE"


def required_blocks(cv_of_log_ratio: float, half_width_target: float, confidence: float = 0.95,
                    minimum: int = 5, maximum: int = 40) -> int:
    """Smallest number of paired blocks whose t interval half-width on the log ratio meets the target, given the
    pilot's standard deviation of log ratios. Bounded by the declared minimum and the budget maximum."""
    for n in range(max(2, minimum), maximum + 1):
        if t_quantile(0.5 + confidence / 2.0, n - 1) * cv_of_log_ratio / math.sqrt(n) <= half_width_target:
            return n
    return maximum

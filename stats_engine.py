"""
Stats engine for the Experiment Readout agent.

All the math happens here, in plain Python, so the numbers are exact and
repeatable. The AI model never calculates statistics - it only explains
the results this file produces.

What it checks for every A/B test:
  1. Sample ratio mismatch (SRM) - did traffic split the way it should?
  2. Conversion rates and relative lift
  3. Statistical significance (two-sided, two-proportion z-test)
  4. Confidence interval for the lift
  5. Statistical power - was the test big enough to detect a real effect?
Then it gives one verdict: Winner, Loser, Flat, Underpowered, or Invalid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

# ---------------------------------------------------------------------------
# Small math helpers (no scipy needed)
# ---------------------------------------------------------------------------


def normal_cdf(z: float) -> float:
    """Probability that a standard normal value is below z."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normal_ppf(p: float) -> float:
    """Inverse of normal_cdf (the z-score for a given probability)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be between 0 and 1")
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def chi_square_1df_p_value(x: float) -> float:
    """p-value for a chi-square statistic with 1 degree of freedom."""
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

WINNER = "Winner"
LOSER = "Loser"
FLAT = "Flat (no meaningful difference)"
UNDERPOWERED = "Underpowered - keep running"
INVALID = "Invalid - traffic split problem"

VERDICT_EMOJI = {
    WINNER: "🟢",
    LOSER: "🔴",
    FLAT: "⚪",
    UNDERPOWERED: "🟡",
    INVALID: "⛔",
}

# A traffic split is flagged only when it is extremely unlikely to be chance.
# p < 0.001 is the common industry threshold for sample ratio mismatch.
SRM_ALPHA = 0.001


@dataclass
class TestInput:
    test_name: str
    control_visitors: int
    control_conversions: int
    variant_visitors: int
    variant_conversions: int
    hypothesis: str = ""
    metric: str = "Conversion rate"
    page: str = ""
    expected_split: float = 0.5  # share of traffic meant for control


@dataclass
class TestResult:
    test_name: str
    hypothesis: str
    metric: str
    page: str
    control_visitors: int
    control_conversions: int
    variant_visitors: int
    variant_conversions: int
    control_rate: float
    variant_rate: float
    absolute_diff: float
    relative_lift: float | None
    p_value: float
    confidence_level: float
    is_significant: bool
    lift_ci_low: float | None
    lift_ci_high: float | None
    srm_p_value: float
    has_srm: bool
    required_visitors_per_arm: int | None
    mde_relative: float
    has_enough_traffic: bool
    verdict: str
    verdict_reason: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------


def two_proportion_z_test(c_n: int, c_x: int, v_n: int, v_x: int) -> float:
    """Two-sided p-value for the difference between two conversion rates."""
    pooled = (c_x + v_x) / (c_n + v_n)
    se = math.sqrt(pooled * (1 - pooled) * (1 / c_n + 1 / v_n))
    if se == 0:
        return 1.0
    z = (v_x / v_n - c_x / c_n) / se
    return 2 * (1 - normal_cdf(abs(z)))


def sample_ratio_mismatch_p_value(c_n: int, v_n: int, expected_split: float = 0.5) -> float:
    """Chi-square test: does the visitor split match the planned split?"""
    total = c_n + v_n
    exp_c = total * expected_split
    exp_v = total * (1 - expected_split)
    if exp_c == 0 or exp_v == 0:
        return 1.0
    chi2 = (c_n - exp_c) ** 2 / exp_c + (v_n - exp_v) ** 2 / exp_v
    return chi_square_1df_p_value(chi2)


def lift_confidence_interval(
    c_n: int, c_x: int, v_n: int, v_x: int, confidence: float
) -> tuple[float | None, float | None]:
    """
    Confidence interval for the RELATIVE lift (e.g. +12%), using the
    delta method on the log of the rate ratio.
    """
    if c_x == 0 or v_x == 0:
        return None, None
    p_c, p_v = c_x / c_n, v_x / v_n
    se_log = math.sqrt((1 - p_c) / c_x + (1 - p_v) / v_x)
    z = normal_ppf(1 - (1 - confidence) / 2)
    log_ratio = math.log(p_v / p_c)
    return math.exp(log_ratio - z * se_log) - 1, math.exp(log_ratio + z * se_log) - 1


def required_sample_size_per_arm(
    baseline_rate: float, mde_relative: float, confidence: float, power: float
) -> int | None:
    """
    Visitors needed in EACH group to reliably detect a lift of
    `mde_relative` (e.g. 0.10 = a 10% relative lift) over the baseline.
    """
    if baseline_rate <= 0 or baseline_rate >= 1 or mde_relative <= 0:
        return None
    p1 = baseline_rate
    p2 = min(p1 * (1 + mde_relative), 0.9999)
    p_bar = (p1 + p2) / 2
    z_a = normal_ppf(1 - (1 - confidence) / 2)
    z_b = normal_ppf(power)
    numerator = (
        z_a * math.sqrt(2 * p_bar * (1 - p_bar))
        + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return math.ceil(numerator / (p2 - p1) ** 2)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def analyze_test(
    t: TestInput,
    confidence: float = 0.95,
    power: float = 0.80,
    mde_relative: float = 0.10,
) -> TestResult:
    c_n, c_x = t.control_visitors, t.control_conversions
    v_n, v_x = t.variant_visitors, t.variant_conversions

    p_c, p_v = c_x / c_n, v_x / v_n
    abs_diff = p_v - p_c
    rel_lift = (p_v - p_c) / p_c if p_c > 0 else None

    p_value = two_proportion_z_test(c_n, c_x, v_n, v_x)
    alpha = 1 - confidence
    significant = p_value < alpha
    ci_low, ci_high = lift_confidence_interval(c_n, c_x, v_n, v_x, confidence)

    srm_p = sample_ratio_mismatch_p_value(c_n, v_n, t.expected_split)
    has_srm = srm_p < SRM_ALPHA

    req_n = required_sample_size_per_arm(p_c if p_c > 0 else p_v, mde_relative, confidence, power)
    enough = req_n is not None and min(c_n, v_n) >= req_n

    warnings: list[str] = []
    if min(c_x, v_x) < 30:
        warnings.append(
            "Fewer than 30 conversions in at least one group - results are fragile."
        )
    if significant and not enough:
        warnings.append(
            "Significant, but the test is smaller than planned. Small tests that "
            "'win' often exaggerate the true lift - consider a confirmation run."
        )

    conf_pct = f"{confidence:.0%}"
    lift_txt = f"{rel_lift:+.1%}" if rel_lift is not None else "an undefined % (control had 0 conversions)"

    if has_srm:
        actual_share = c_n / (c_n + v_n)
        verdict = INVALID
        reason = (
            f"Control got {actual_share:.1%} of traffic instead of "
            f"{t.expected_split:.0%} (SRM p = {srm_p:.2g}). Something in the "
            "setup (redirects, bots, tracking, targeting) is skewing who sees "
            "each version, so the result can't be trusted. Fix the split and re-run."
        )
    elif significant and abs_diff > 0:
        verdict = WINNER
        reason = (
            f"The variant beat control by {lift_txt} and the result is "
            f"statistically significant at {conf_pct} confidence (p = {p_value:.3f})."
        )
    elif significant and abs_diff < 0:
        verdict = LOSER
        reason = (
            f"The variant changed {t.metric.lower()} by {lift_txt} vs. control, and the "
            f"drop is statistically significant at {conf_pct} confidence "
            f"(p = {p_value:.3f}). Keep control."
        )
    elif enough:
        verdict = FLAT
        reason = (
            f"The test had enough traffic to detect a {mde_relative:.0%} lift and "
            f"didn't find one (p = {p_value:.2f}). Any real difference is likely "
            "too small to matter. Ship whichever is simpler and move on."
        )
    else:
        need = f"{req_n:,}" if req_n else "more"
        reason = (
            f"Observed lift of {lift_txt} is not significant (p = {p_value:.2f}), "
            f"but the test is too small to say there's no effect. It needs about "
            f"{need} visitors per group to detect a {mde_relative:.0%} lift; it has "
            f"{min(c_n, v_n):,}. Don't call this a winner yet."
        )
        verdict = UNDERPOWERED

    return TestResult(
        test_name=t.test_name,
        hypothesis=t.hypothesis,
        metric=t.metric,
        page=t.page,
        control_visitors=c_n,
        control_conversions=c_x,
        variant_visitors=v_n,
        variant_conversions=v_x,
        control_rate=p_c,
        variant_rate=p_v,
        absolute_diff=abs_diff,
        relative_lift=rel_lift,
        p_value=p_value,
        confidence_level=confidence,
        is_significant=significant,
        lift_ci_low=ci_low,
        lift_ci_high=ci_high,
        srm_p_value=srm_p,
        has_srm=has_srm,
        required_visitors_per_arm=req_n,
        mde_relative=mde_relative,
        has_enough_traffic=enough,
        verdict=verdict,
        verdict_reason=reason,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Input cleaning (works on plain dicts, e.g. rows from a CSV)
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "test_name",
    "control_visitors",
    "control_conversions",
    "variant_visitors",
    "variant_conversions",
]

COLUMN_ALIASES = {
    "name": "test_name",
    "test": "test_name",
    "experiment": "test_name",
    "experiment_name": "test_name",
    "control_users": "control_visitors",
    "control_sessions": "control_visitors",
    "variant_users": "variant_visitors",
    "variant_sessions": "variant_visitors",
    "treatment_visitors": "variant_visitors",
    "treatment_conversions": "variant_conversions",
}

MAX_TESTS = 15


def normalize_column(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    return COLUMN_ALIASES.get(key, key)


def _to_int(value) -> int:
    if isinstance(value, str):
        value = value.replace(",", "").strip()
    number = float(value)
    if math.isnan(number) or number != int(number):
        raise ValueError
    return int(number)


def parse_rows(rows: list[dict]) -> tuple[list[TestInput], list[str]]:
    """
    Turn raw rows into validated TestInput objects.
    Returns (valid_tests, error_messages). Bad rows are skipped, not fatal.
    """
    errors: list[str] = []
    tests: list[TestInput] = []

    if not rows:
        return [], ["No rows found."]

    normalized = [{normalize_column(k): v for k, v in r.items()} for r in rows]
    missing = [c for c in REQUIRED_COLUMNS if c not in normalized[0]]
    if missing:
        return [], [
            "Missing required column(s): " + ", ".join(missing)
            + ". Download the CSV template to see the expected format."
        ]

    if len(normalized) > MAX_TESTS:
        errors.append(f"Only the first {MAX_TESTS} tests were analyzed.")
        normalized = normalized[:MAX_TESTS]

    for i, r in enumerate(normalized, start=1):
        raw_name = r.get("test_name")
        if raw_name is None or (isinstance(raw_name, float) and math.isnan(raw_name)):
            raw_name = ""
        name = str(raw_name).strip() or f"Test {i}"
        try:
            c_n = _to_int(r["control_visitors"])
            c_x = _to_int(r["control_conversions"])
            v_n = _to_int(r["variant_visitors"])
            v_x = _to_int(r["variant_conversions"])
        except (ValueError, TypeError):
            errors.append(f"Row {i} ({name}): visitors and conversions must be whole numbers.")
            continue
        if min(c_n, c_x, v_n, v_x) < 0:
            errors.append(f"Row {i} ({name}): numbers can't be negative.")
            continue
        if c_n == 0 or v_n == 0:
            errors.append(f"Row {i} ({name}): each group needs at least one visitor.")
            continue
        if c_x > c_n or v_x > v_n:
            errors.append(f"Row {i} ({name}): conversions can't be higher than visitors.")
            continue
        if c_x == 0 and v_x == 0:
            errors.append(f"Row {i} ({name}): no conversions in either group, nothing to compare.")
            continue

        def text(key: str, default: str = "") -> str:
            val = r.get(key)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return default
            return str(val).strip() or default

        tests.append(
            TestInput(
                test_name=name,
                control_visitors=c_n,
                control_conversions=c_x,
                variant_visitors=v_n,
                variant_conversions=v_x,
                hypothesis=text("hypothesis"),
                metric=text("metric", "Conversion rate"),
                page=text("page"),
            )
        )

    return tests, errors

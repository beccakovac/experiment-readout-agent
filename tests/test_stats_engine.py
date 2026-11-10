"""
Run with:  python -m pytest -q

These tests check the stats engine against the sample data, where the
right answer for every test is known in advance (see README answer key).
"""

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import stats_engine as se  # noqa: E402


def load_sample():
    with open(ROOT / "sample_data" / "sample_tests.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    tests, errors = se.parse_rows(rows)
    assert errors == []
    return {t.test_name: se.analyze_test(t) for t in tests}


# --- Answer key ------------------------------------------------------------

def test_clear_winner():
    r = load_sample()["Annual plan pre-selected"]
    assert r.verdict == se.WINNER
    assert math.isclose(r.relative_lift, 0.20, rel_tol=1e-9)
    assert r.p_value < 0.01


def test_flat_result_with_enough_traffic():
    r = load_sample()["New hero headline"]
    assert r.verdict == se.FLAT
    assert r.has_enough_traffic


def test_small_test_with_big_lift_is_underpowered_not_a_winner():
    r = load_sample()["Remove phone field"]
    assert r.relative_lift > 0.25          # looks exciting...
    assert r.verdict == se.UNDERPOWERED    # ...but can't be trusted yet
    assert not r.is_significant


def test_sample_ratio_mismatch_is_invalid():
    r = load_sample()["Trust badges at checkout"]
    assert r.has_srm
    assert r.verdict == se.INVALID


def test_clear_loser():
    r = load_sample()["Skip product tour"]
    assert r.verdict == se.LOSER
    assert r.relative_lift < 0


# --- Math checks against known reference values ----------------------------

def test_normal_functions():
    assert math.isclose(se.normal_cdf(1.959964), 0.975, abs_tol=1e-6)
    assert math.isclose(se.normal_ppf(0.975), 1.959964, abs_tol=1e-5)
    assert math.isclose(se.normal_ppf(0.80), 0.841621, abs_tol=1e-5)


def test_z_test_reference_value():
    # 4.0% vs 4.8% on 10k each -> z ~ 2.76, p ~ 0.0058
    p = se.two_proportion_z_test(10000, 400, 10000, 480)
    assert math.isclose(p, 0.0058, abs_tol=0.0005)


def test_sample_size_reference_value():
    # 5% baseline, 10% relative MDE, 95% / 80% -> ~31,200 per arm
    n = se.required_sample_size_per_arm(0.05, 0.10, 0.95, 0.80)
    assert 31000 <= n <= 31500


def test_even_split_has_no_srm():
    assert se.sample_ratio_mismatch_p_value(10000, 10050) > 0.5


def test_ci_contains_observed_lift():
    r = load_sample()["Annual plan pre-selected"]
    assert r.lift_ci_low < r.relative_lift < r.lift_ci_high
    assert r.lift_ci_low > 0  # significant winner -> CI above zero


# --- Input validation -------------------------------------------------------

def test_missing_columns():
    tests, errors = se.parse_rows([{"test_name": "x", "control_visitors": 10}])
    assert tests == []
    assert "Missing required column" in errors[0]


def test_bad_rows_are_skipped_not_fatal():
    rows = [
        {"Test Name": "ok", "Control Visitors": "1,000", "Control Conversions": "50",
         "Variant Visitors": "1000", "Variant Conversions": "60"},
        {"Test Name": "too many conv", "Control Visitors": "100", "Control Conversions": "200",
         "Variant Visitors": "100", "Variant Conversions": "5"},
        {"Test Name": "text", "Control Visitors": "abc", "Control Conversions": "1",
         "Variant Visitors": "100", "Variant Conversions": "5"},
        {"Test Name": "zero", "Control Visitors": "0", "Control Conversions": "0",
         "Variant Visitors": "100", "Variant Conversions": "5"},
    ]
    tests, errors = se.parse_rows(rows)
    assert [t.test_name for t in tests] == ["ok"]
    assert len(errors) == 3


def test_row_limit():
    row = {"test_name": "t", "control_visitors": 100, "control_conversions": 5,
           "variant_visitors": 100, "variant_conversions": 6}
    tests, errors = se.parse_rows([row] * 20)
    assert len(tests) == se.MAX_TESTS
    assert any("first" in e for e in errors)


def test_zero_conversions_in_control_does_not_crash():
    t = se.TestInput("edge", 500, 0, 500, 12)
    r = se.analyze_test(t)
    assert r.relative_lift is None
    assert r.verdict in {se.WINNER, se.UNDERPOWERED}

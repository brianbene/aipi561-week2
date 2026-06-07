"""
test_monitoring.py
Tests for compute_metrics.py and detect_drift.py functions.
Run with: python -m pytest week4/scripts/test_monitoring.py -v
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from compute_metrics import compute_psi, alert_level, run_metrics
from detect_drift import detect_drift

BASELINE_PATH = "week4/data/demand_enriched_baseline.parquet"
WEEK4_PATH    = "week4/data/demand_enriched_week4.parquet"

@pytest.fixture(scope="module")
def baseline():
    return pd.read_parquet(BASELINE_PATH)

@pytest.fixture(scope="module")
def feb(baseline):
    week4 = pd.read_parquet(WEEK4_PATH)
    return week4[(week4['time_bucket'] >= '2026-02-02') &
                 (week4['time_bucket'] <= '2026-02-28 23:45:00')].copy()

# ── PSI function tests ────────────────────────────────────────────────

def test_psi_identical_distributions():
    """PSI of a series against itself should be 0."""
    s = pd.Series([1, 2, 3, 4, 5, 2, 3, 4, 1, 2])
    assert compute_psi(s, s) == pytest.approx(0.0, abs=0.01)

def test_psi_very_different_distributions():
    """PSI of two very different distributions should exceed 0.25."""
    low  = pd.Series(np.random.uniform(0, 1, 1000))
    high = pd.Series(np.random.uniform(10, 20, 1000))
    assert compute_psi(low, high) > 0.25

def test_psi_returns_float():
    """compute_psi should always return a float."""
    s1 = pd.Series([1.0, 2.0, 3.0])
    s2 = pd.Series([1.5, 2.5, 3.5])
    result = compute_psi(s1, s2)
    assert isinstance(result, float)

def test_psi_constant_series_returns_zero():
    """A constant series has no bins — PSI should return 0.0 gracefully."""
    s = pd.Series([5.0] * 100)
    assert compute_psi(s, s) == 0.0

# ── alert_level tests ─────────────────────────────────────────────────

def test_alert_level_psi_critical():
    assert alert_level(psi=0.30) == "CRITICAL"

def test_alert_level_psi_warning():
    assert alert_level(psi=0.15) == "WARNING"

def test_alert_level_psi_ok():
    assert alert_level(psi=0.05) == "ok"

def test_alert_level_pvalue_critical():
    assert alert_level(p_value=0.001) == "CRITICAL"

def test_alert_level_pvalue_ok():
    assert alert_level(p_value=0.10) == "ok"

def test_alert_level_rate_change_critical():
    assert alert_level(rate_change=1.0) == "CRITICAL"

def test_alert_level_rate_change_ok():
    assert alert_level(rate_change=0.05) == "ok"

# ── run_metrics integration tests ─────────────────────────────────────

def test_run_metrics_returns_eight_results(baseline, feb):
    """run_metrics should return exactly 8 metric results."""
    results = run_metrics(baseline, feb)
    assert len(results) == 8

def test_run_metrics_all_have_required_keys(baseline, feb):
    """Every metric result must have the 5 required keys."""
    results = run_metrics(baseline, feb)
    required = {'metric', 'category', 'baseline', 'current', 'value', 'alert'}
    for r in results:
        assert required.issubset(r.keys()), f"Missing keys in: {r}"

def test_run_metrics_trip_count_psi_critical(baseline, feb):
    """trip_count PSI should be CRITICAL given the 12x demand shift."""
    results = run_metrics(baseline, feb)
    tc_psi = next(r for r in results if r['metric'] == 'trip_count PSI')
    assert tc_psi['alert'] == 'CRITICAL'

def test_run_metrics_cbd_critical(baseline, feb):
    """cbd_pricing_active rate change should be CRITICAL (0->1 flip)."""
    results = run_metrics(baseline, feb)
    cbd = next(r for r in results if r['metric'] == 'cbd_pricing_active rate')
    assert cbd['alert'] == 'CRITICAL'

# ── detect_drift integration tests ───────────────────────────────────

def test_detect_drift_finds_four_patterns(baseline, feb):
    """detect_drift should identify exactly 4 drift patterns."""
    findings = detect_drift(baseline, feb)
    assert len(findings) == 4

def test_detect_drift_all_patterns_detected(baseline, feb):
    """All 4 drift patterns should be detected in Feb data."""
    findings = detect_drift(baseline, feb)
    for f in findings:
        assert f['detected'], f"Pattern {f['id']} ({f['name']}) not detected"

def test_detect_drift_patterns_have_required_keys(baseline, feb):
    """Each drift finding must include id, name, type, detected, evidence, impact, segment."""
    findings = detect_drift(baseline, feb)
    required = {'id', 'name', 'type', 'detected', 'evidence', 'impact', 'segment'}
    for f in findings:
        assert required.issubset(f.keys()), f"Missing keys in finding: {f}"

def test_detect_drift_baseline_vs_itself(baseline):
    """Running drift detection on baseline vs itself should detect no patterns."""
    findings = detect_drift(baseline, baseline)
    assert not any(f['detected'] for f in findings), \
        "False positive: drift detected when comparing baseline to itself"

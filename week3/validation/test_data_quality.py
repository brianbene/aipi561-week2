import pandas as pd
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from week3.validation.check_data_quality import (
    validate_data,
    check_schema,
    check_duplicates,
    check_trip_count_range,
    check_holiday_rate,
    check_lag_contamination,
    apply_graceful_degradation,
    EXPECTED_COLUMNS,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


@pytest.fixture(scope="module")
def baseline_df():
    return pd.read_parquet(os.path.join(DATA_DIR, 'demand_enriched_baseline.parquet'))


@pytest.fixture(scope="module")
def corrupted_df():
    df = pd.read_parquet(os.path.join(DATA_DIR, 'demand_enriched_corrupted.parquet'))
    return df[df['time_bucket'] >= '2026-01-16'].copy()


@pytest.fixture(scope="module")
def clean_df(baseline_df):
    """A minimal clean dataframe built from baseline stats for injection tests."""
    n = 200
    df = pd.DataFrame({col: 0 for col in EXPECTED_COLUMNS}, index=range(n))
    df['PULocationID'] = list(range(1, n + 1))
    df['time_bucket'] = pd.date_range('2026-01-16', periods=n, freq='15min')
    df['trip_count'] = np.random.randint(0, 50, n)
    df['is_holiday'] = 0
    df['lag_1week'] = df['trip_count'] + np.random.normal(0, 2, n)
    df['zone_slot_baseline'] = df['trip_count'].mean()
    return df


# ── Baseline should pass all checks ──────────────────────────────────────────

def test_baseline_passes_schema(baseline_df):
    result = check_schema(baseline_df)
    assert result['passed'], f"Schema failed on baseline: {result['detail']}"


def test_baseline_passes_trip_count(baseline_df):
    result = check_trip_count_range(baseline_df)
    assert result['passed'], f"Trip count failed on baseline: {result['detail']}"


def test_baseline_passes_holiday_rate(baseline_df):
    result = check_holiday_rate(baseline_df)
    assert result['passed'], f"Holiday rate failed on baseline: {result['detail']}"


def test_baseline_passes_duplicates(baseline_df):
    result = check_duplicates(baseline_df)
    assert result['passed'], f"Duplicates found in baseline: {result['detail']}"


# ── Corrupted data should fail all four checks ────────────────────────────────

def test_corrupted_fails_overall(corrupted_df, baseline_df):
    result = validate_data(corrupted_df, baseline_df)
    assert not result['is_valid'], "Corrupted data should fail overall validation"
    assert result['passed_checks'] < result['total_checks']


def test_corrupted_detects_duplicates(corrupted_df):
    result = check_duplicates(corrupted_df)
    assert not result['passed'], "Should detect duplicates in corrupted data"
    assert result['duplicate_rows'] > 0
    assert set([4, 43, 87, 107, 229]).issubset(set(result['affected_zones']))


def test_corrupted_detects_trip_count_outliers(corrupted_df):
    result = check_trip_count_range(corrupted_df)
    assert not result['passed'], "Should detect out-of-range trip_count"
    assert result['bad_rows'] >= 600
    assert any(v < 0 for v in result['bad_values_sample'])
    assert any(v > 1000 for v in result['bad_values_sample'])


def test_corrupted_detects_holiday_drift(corrupted_df):
    result = check_holiday_rate(corrupted_df)
    assert not result['passed'], "Should detect holiday rate drift"
    assert result['fully_flagged_days'] > 0


def test_corrupted_detects_lag_contamination(corrupted_df):
    result = check_lag_contamination(corrupted_df)
    assert not result['passed'], "Should detect lag contamination"
    assert set([161, 162, 186]).issubset(set(result['contaminated_zones']))
    for zone, corr in result['zone_correlations'].items():
        assert abs(corr) < 0.10, f"Zone {zone} corr {corr} should be near zero"


# ── Individual injection tests ────────────────────────────────────────────────

def test_injected_duplicates_detected(clean_df):
    df = clean_df.copy()
    # Duplicate first 50 rows
    df = pd.concat([df, df.iloc[:50]], ignore_index=True)
    result = check_duplicates(df)
    assert not result['passed']
    assert result['duplicate_rows'] == 100  # 50 originals + 50 copies = 100 in mask


def test_injected_negative_trip_count_detected(clean_df):
    df = clean_df.copy()
    df.loc[0:9, 'trip_count'] = -5
    result = check_trip_count_range(df)
    assert not result['passed']
    assert result['bad_rows'] == 10


def test_injected_extreme_trip_count_detected(clean_df):
    df = clean_df.copy()
    df.loc[0:4, 'trip_count'] = 99999
    result = check_trip_count_range(df)
    assert not result['passed']
    assert result['bad_rows'] == 5


def test_injected_holiday_100pct_detected(clean_df):
    df = clean_df.copy()
    df['is_holiday'] = 1
    result = check_holiday_rate(df)
    assert not result['passed']
    assert result['fully_flagged_days'] > 0


# ── Graceful degradation tests ────────────────────────────────────────────────

def test_degradation_removes_duplicates(clean_df):
    df = pd.concat([clean_df, clean_df.iloc[:20]], ignore_index=True)
    result = validate_data(df)
    issues = result['issues']
    cleaned = apply_graceful_degradation(df, issues)
    assert cleaned.duplicated(subset=['PULocationID', 'time_bucket']).sum() == 0


def test_degradation_fixes_trip_count(clean_df):
    df = clean_df.copy()
    df.loc[0:9, 'trip_count'] = -99
    result = validate_data(df)
    cleaned = apply_graceful_degradation(df, result['issues'])
    assert (cleaned['trip_count'] < 0).sum() == 0


def test_api_does_not_crash_on_bad_data(corrupted_df, baseline_df):
    """Graceful degradation should return a dataframe, never raise."""
    result = validate_data(corrupted_df, baseline_df)
    try:
        cleaned = apply_graceful_degradation(corrupted_df, result['issues'])
        assert isinstance(cleaned, pd.DataFrame)
        assert len(cleaned) > 0
    except Exception as e:
        pytest.fail(f"apply_graceful_degradation raised an exception: {e}")

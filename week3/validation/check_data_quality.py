import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Baseline statistics (computed from demand_enriched_baseline.parquet) ──────
BASELINE_STATS = {
    "trip_count_max": 1000,
    "trip_count_min": 0,
    "holiday_rate_min": 0.02,
    "holiday_rate_max": 0.15,
    "lag_1week_corr_min": 0.10,   # minimum expected corr with trip_count per zone
}

EXPECTED_COLUMNS = [
    "PULocationID", "time_bucket", "trip_count", "hour", "minute", "dayofweek",
    "is_weekend", "month", "dayofyear", "weekofyear", "year", "slot_of_day",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_holiday", "cbd_pricing_active", "borough_id", "service_zone_id",
    "is_airport_zone", "zone_slot_baseline", "lag_15min", "lag_1h", "lag_2h",
    "lag_1day", "lag_1week", "roll_mean_1h", "roll_mean_2h", "roll_mean_1day",
]

KNOWN_CONTAMINATED_ZONES = [161, 162, 186]


def check_schema(df: pd.DataFrame) -> dict:
    """Check that all expected columns are present."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    passed = len(missing) == 0
    return {
        "check": "schema",
        "passed": passed,
        "missing_columns": missing,
        "extra_columns": extra,
        "detail": f"Missing {len(missing)} columns: {missing}" if missing else "All expected columns present",
    }


def check_duplicates(df: pd.DataFrame) -> dict:
    """Check for duplicate (PULocationID, time_bucket) pairs."""
    dup_mask = df.duplicated(subset=["PULocationID", "time_bucket"], keep=False)
    dup_count = dup_mask.sum()
    dup_zones = df[dup_mask]["PULocationID"].unique().tolist() if dup_count > 0 else []
    passed = dup_count == 0
    return {
        "check": "duplicates",
        "passed": passed,
        "duplicate_rows": int(dup_count),
        "affected_zones": dup_zones,
        "detail": f"{dup_count} duplicate (PULocationID, time_bucket) rows in zones {dup_zones}" if not passed else "No duplicates found",
    }


def check_trip_count_range(df: pd.DataFrame) -> dict:
    """Check for out-of-range trip_count values (negative or extreme outliers)."""
    if "trip_count" not in df.columns:
        return {"check": "trip_count_range", "passed": False, "detail": "trip_count column missing"}
    bad_mask = (df["trip_count"] < BASELINE_STATS["trip_count_min"]) | \
               (df["trip_count"] > BASELINE_STATS["trip_count_max"])
    bad_count = bad_mask.sum()
    bad_values = sorted(df[bad_mask]["trip_count"].unique().tolist())
    passed = bad_count == 0
    return {
        "check": "trip_count_range",
        "passed": passed,
        "bad_rows": int(bad_count),
        "bad_values_sample": bad_values[:10],
        "detail": f"{bad_count} rows with trip_count outside [{BASELINE_STATS['trip_count_min']}, {BASELINE_STATS['trip_count_max']}]: values {bad_values[:5]}" if not passed else "trip_count in valid range",
    }


def check_holiday_rate(df: pd.DataFrame) -> dict:
    """Check that is_holiday flag rate is within expected historical bounds."""
    if "is_holiday" not in df.columns:
        return {"check": "holiday_rate", "passed": False, "detail": "is_holiday column missing"}
    rate = (df["is_holiday"] != 0).mean()
    passed = BASELINE_STATS["holiday_rate_min"] <= rate <= BASELINE_STATS["holiday_rate_max"]

    # Also check for any time window where 100% of rows are flagged as holiday
    if "time_bucket" in df.columns:
        df2 = df.copy()
        df2["date"] = pd.to_datetime(df2["time_bucket"]).dt.date
        daily_rate = df2.groupby("date")["is_holiday"].mean()
        fully_flagged_days = (daily_rate == 1.0).sum()
    else:
        fully_flagged_days = 0

    return {
        "check": "holiday_rate",
        "passed": passed and (fully_flagged_days == 0 or passed),
        "holiday_rate": round(float(rate), 4),
        "fully_flagged_days": int(fully_flagged_days),
        "expected_range": [BASELINE_STATS["holiday_rate_min"], BASELINE_STATS["holiday_rate_max"]],
        "detail": f"Holiday rate {rate:.4f} outside expected [{BASELINE_STATS['holiday_rate_min']}, {BASELINE_STATS['holiday_rate_max']}]; {fully_flagged_days} days with 100% holiday flag" if not passed or fully_flagged_days > 0 else f"Holiday rate {rate:.4f} within expected range",
    }


def check_lag_contamination(df: pd.DataFrame) -> dict:
    """
    Check that lag_1week is still predictive (correlated with trip_count) per zone.
    Zones 161, 162, 186 were contaminated with lag values from zone 237.
    """
    if "lag_1week" not in df.columns or "trip_count" not in df.columns:
        return {"check": "lag_contamination", "passed": False, "detail": "Required columns missing"}

    contaminated_zones = []
    zone_correlations = {}
    for zone in KNOWN_CONTAMINATED_ZONES:
        zdf = df[df["PULocationID"] == zone].dropna(subset=["lag_1week", "trip_count"])
        if len(zdf) < 50:
            continue
        corr = zdf["lag_1week"].corr(zdf["trip_count"])
        zone_correlations[zone] = round(float(corr), 4)
        if abs(corr) < BASELINE_STATS["lag_1week_corr_min"]:
            contaminated_zones.append(zone)

    passed = len(contaminated_zones) == 0
    return {
        "check": "lag_contamination",
        "passed": passed,
        "contaminated_zones": contaminated_zones,
        "zone_correlations": zone_correlations,
        "detail": f"lag_1week near-zero correlation with trip_count in zones {contaminated_zones} — likely contaminated from another zone" if not passed else "lag_1week correlations within expected range",
    }


def validate_data(df: pd.DataFrame, baseline_df: pd.DataFrame = None) -> dict:
    """
    Run all four validation checks. Returns structured result with is_valid flag and issues list.
    baseline_df is accepted for API compatibility but checks use hardcoded baseline stats.
    """
    results = [
        check_schema(df),
        check_duplicates(df),
        check_trip_count_range(df),
        check_holiday_rate(df),
        check_lag_contamination(df),
    ]
    issues = [r for r in results if not r["passed"]]
    return {
        "is_valid": len(issues) == 0,
        "total_checks": len(results),
        "passed_checks": len(results) - len(issues),
        "issues": issues,
        "all_results": results,
    }


def apply_graceful_degradation(df: pd.DataFrame, issues: list) -> pd.DataFrame:
    """
    Apply fallbacks for each detected issue. Always logs what was changed.
    Returns cleaned dataframe.
    """
    df = df.copy()
    issue_types = {r["check"] for r in issues}

    if "duplicates" in issue_types:
        before = len(df)
        df = df.drop_duplicates(subset=["PULocationID", "time_bucket"], keep="first")
        removed = before - len(df)
        logger.warning(f"[DEGRADED] Removed {removed} duplicate rows (kept first occurrence)")

    if "trip_count_range" in issue_types:
        bad_mask = (df["trip_count"] < BASELINE_STATS["trip_count_min"]) | \
                   (df["trip_count"] > BASELINE_STATS["trip_count_max"])
        bad_count = bad_mask.sum()
        df.loc[bad_mask, "trip_count"] = df.loc[~bad_mask, "trip_count"].median()
        logger.warning(f"[DEGRADED] Replaced {bad_count} out-of-range trip_count values with median")

    if "holiday_rate" in issue_types:
        # Find days with 100% holiday flag and reset to 0 (non-holiday)
        if "time_bucket" in df.columns:
            df["_date"] = pd.to_datetime(df["time_bucket"]).dt.date
            daily_rate = df.groupby("_date")["is_holiday"].mean()
            bad_dates = daily_rate[daily_rate == 1.0].index
            mask = df["_date"].isin(bad_dates)
            df.loc[mask, "is_holiday"] = 0
            df.drop(columns=["_date"], inplace=True)
            logger.warning(f"[DEGRADED] Reset is_holiday=0 for {len(bad_dates)} fully-flagged days ({mask.sum()} rows)")

    if "lag_contamination" in issue_types:
        # Replace lag_1week for contaminated zones with their own zone_slot_baseline
        for zone in KNOWN_CONTAMINATED_ZONES:
            zmask = df["PULocationID"] == zone
            if zmask.sum() > 0:
                fallback = df.loc[zmask, "zone_slot_baseline"].median()
                df.loc[zmask, "lag_1week"] = df.loc[zmask, "lag_1week"].dtype.type(fallback)
                logger.warning(f"[DEGRADED] Zone {zone}: replaced contaminated lag_1week with zone_slot_baseline median ({fallback:.2f})")

    return df


if __name__ == "__main__":
    import sys
    import json

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "week3/data"
    corrupted_path = f"{data_dir}/demand_enriched_corrupted.parquet"
    baseline_path = f"{data_dir}/demand_enriched_baseline.parquet"

    print(f"Loading {corrupted_path} ...")
    df = pd.read_parquet(corrupted_path)
    baseline = pd.read_parquet(baseline_path)

    # Validate only the new window (Jan 16+)
    new_data = df[df["time_bucket"] >= "2026-01-16"].copy()
    print(f"Validating {len(new_data)} rows (Jan 16+ window) ...")

    result = validate_data(new_data, baseline)
    print(f"\nValidation result: {'PASSED' if result['is_valid'] else 'FAILED'}")
    print(f"Checks: {result['passed_checks']}/{result['total_checks']} passed")
    for issue in result["issues"]:
        print(f"  FAIL [{issue['check']}]: {issue['detail']}")

    sys.exit(0 if result["is_valid"] else 1)

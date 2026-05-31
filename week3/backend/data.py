import os
import logging
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from week3.validation.check_data_quality import validate_data, apply_graceful_degradation

logger = logging.getLogger(__name__)

# ── Module-level cache so we only load once per pod startup ──────────────────
_demand_data: pd.DataFrame = None
_data_quality_status: dict = {"validated": False, "is_valid": None, "issues": []}


def get_data_quality_status() -> dict:
    return _data_quality_status


def load_demand_data(data_path: str, baseline_path: str = None) -> pd.DataFrame:
    """
    Load demand data from parquet, validate it, and apply graceful degradation
    if issues are found. Logs all quality issues and fallbacks applied.
    Always returns a usable DataFrame — never crashes the API.
    """
    global _demand_data, _data_quality_status

    if _demand_data is not None:
        return _demand_data

    # ── Step 1: Load the data ────────────────────────────────────────────────
    try:
        logger.info(f"Loading demand data from {data_path}")
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        logger.error(f"[CRITICAL] Failed to load demand data from {data_path}: {e}")
        _demand_data = _get_empty_fallback()
        _data_quality_status = {"validated": False, "is_valid": False,
                                 "issues": [{"check": "load", "detail": str(e)}]}
        return _demand_data

    # ── Step 2: Load baseline for comparison (optional) ─────────────────────
    baseline_df = None
    if baseline_path and os.path.exists(baseline_path):
        try:
            baseline_df = pd.read_parquet(baseline_path)
            logger.info(f"Loaded baseline from {baseline_path} ({len(baseline_df)} rows)")
        except Exception as e:
            logger.warning(f"Could not load baseline data: {e} — continuing without it")

    # ── Step 3: Validate ─────────────────────────────────────────────────────
    try:
        logger.info("Running data quality validation...")
        result = validate_data(df, baseline_df)
        _data_quality_status["validated"] = True
        _data_quality_status["is_valid"] = result["is_valid"]
        _data_quality_status["issues"] = result["issues"]

        if result["is_valid"]:
            logger.info(f"Data quality validation PASSED ({result['passed_checks']}/{result['total_checks']} checks)")
        else:
            for issue in result["issues"]:
                logger.warning(f"[DATA QUALITY] FAILED check '{issue['check']}': {issue['detail']}")

            # ── Step 4: Graceful degradation ─────────────────────────────────
            logger.info("Applying graceful degradation for detected issues...")
            df = apply_graceful_degradation(df, result["issues"])
            logger.info(f"Graceful degradation complete. Serving {len(df)} rows.")

    except Exception as e:
        logger.error(f"[ERROR] Validation step failed unexpectedly: {e} — serving raw data")
        _data_quality_status["validated"] = False
        _data_quality_status["issues"] = [{"check": "validation_error", "detail": str(e)}]

    _demand_data = df
    return _demand_data


def get_zone_data(zone_id: int) -> pd.DataFrame:
    """Return rows for a specific zone. Returns empty DataFrame if zone not found."""
    df = _demand_data
    if df is None:
        logger.error("Demand data not loaded — call load_demand_data() at startup")
        return pd.DataFrame()
    zone_df = df[df["PULocationID"] == zone_id]
    if len(zone_df) == 0:
        logger.warning(f"No data found for zone {zone_id}")
    return zone_df


def _get_empty_fallback() -> pd.DataFrame:
    """Return an empty DataFrame with the correct schema as last-resort fallback."""
    logger.error("[FALLBACK] Returning empty DataFrame — all API predictions will be unavailable")
    return pd.DataFrame(columns=[
        "PULocationID", "time_bucket", "trip_count", "hour", "dayofweek",
        "is_holiday", "lag_1week", "zone_slot_baseline"
    ])


def reset_cache():
    """Force reload on next call. Used in tests."""
    global _demand_data, _data_quality_status
    _demand_data = None
    _data_quality_status = {"validated": False, "is_valid": None, "issues": []}

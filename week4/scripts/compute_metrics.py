"""
compute_metrics.py
Loads baseline and Feb 2-28 2026 data, computes 8 monitoring metrics,
prints a report, and exits with code 1 if any CRITICAL alerts fire.
"""
import sys
import pandas as pd
import numpy as np
from scipy import stats

BASELINE_PATH = "week4/data/demand_enriched_baseline.parquet"
WEEK4_PATH    = "week4/data/demand_enriched_week4.parquet"

def compute_psi(expected, actual, bins=10):
    expected = expected.dropna().values
    actual   = actual.dropna().values
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0
    exp_counts, _ = np.histogram(expected, bins=breakpoints)
    act_counts, _ = np.histogram(actual,   bins=breakpoints)
    exp_pct = exp_counts / len(expected)
    act_pct = act_counts / len(actual)
    exp_pct = np.where(exp_pct == 0, 0.0001, exp_pct)
    act_pct = np.where(act_pct == 0, 0.0001, act_pct)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

def alert_level(psi=None, p_value=None, rate_change=None):
    if psi is not None:
        if psi > 0.25:  return "CRITICAL"
        if psi > 0.10:  return "WARNING"
        return "ok"
    if p_value is not None:
        if p_value < 0.01: return "CRITICAL"
        if p_value < 0.05: return "WARNING"
        return "ok"
    if rate_change is not None:
        if abs(rate_change) > 0.50: return "CRITICAL"
        if abs(rate_change) > 0.20: return "WARNING"
        return "ok"
    return "ok"

def run_metrics(baseline, feb):
    results = []

    # Metric 1: trip_count PSI
    psi = compute_psi(baseline['trip_count'], feb['trip_count'])
    results.append({
        'metric': 'trip_count PSI',
        'category': 'Data Drift',
        'baseline': f"mean={baseline['trip_count'].mean():.3f}",
        'current':  f"mean={feb['trip_count'].mean():.3f}",
        'value': f"PSI={psi:.4f}",
        'alert': alert_level(psi=psi)
    })

    # Metric 2: trip_count KS test
    ks_stat, p_val = stats.ks_2samp(baseline['trip_count'].dropna(), feb['trip_count'].dropna())
    results.append({
        'metric': 'trip_count KS test',
        'category': 'Data Drift',
        'baseline': 'Jan 1-15 distribution',
        'current':  'Feb 2-28 distribution',
        'value': f"KS={ks_stat:.4f}, p={p_val:.2e}",
        'alert': alert_level(p_value=p_val)
    })

    # Metric 3: roll_mean_1h PSI (lag cascade indicator)
    psi_rm = compute_psi(baseline['roll_mean_1h'], feb['roll_mean_1h'])
    results.append({
        'metric': 'roll_mean_1h PSI',
        'category': 'Data Drift',
        'baseline': f"mean={baseline['roll_mean_1h'].mean():.3f}",
        'current':  f"mean={feb['roll_mean_1h'].mean():.3f}",
        'value': f"PSI={psi_rm:.4f}",
        'alert': alert_level(psi=psi_rm)
    })

    # Metric 4: zone_slot_baseline PSI
    psi_zsb = compute_psi(baseline['zone_slot_baseline'], feb['zone_slot_baseline'])
    results.append({
        'metric': 'zone_slot_baseline PSI',
        'category': 'Data Drift',
        'baseline': f"mean={baseline['zone_slot_baseline'].mean():.3f}",
        'current':  f"mean={feb['zone_slot_baseline'].mean():.3f}",
        'value': f"PSI={psi_zsb:.4f}",
        'alert': alert_level(psi=psi_zsb)
    })

    # Metric 5: cbd_pricing_active rate change
    b_cbd = baseline['cbd_pricing_active'].mean()
    f_cbd = feb['cbd_pricing_active'].mean()
    change = f_cbd - b_cbd
    results.append({
        'metric': 'cbd_pricing_active rate',
        'category': 'Concept Drift',
        'baseline': f"{b_cbd:.4f}",
        'current':  f"{f_cbd:.4f}",
        'value': f"change={change:+.4f}",
        'alert': alert_level(rate_change=change)
    })

    # Metric 6: is_holiday rate change
    b_hol = (baseline['is_holiday'] != 0).mean()
    f_hol = (feb['is_holiday'] != 0).mean()
    change_hol = f_hol - b_hol
    results.append({
        'metric': 'is_holiday rate',
        'category': 'Data Drift',
        'baseline': f"{b_hol:.4f}",
        'current':  f"{f_hol:.4f}",
        'value': f"change={change_hol:+.4f}",
        'alert': alert_level(rate_change=change_hol)
    })

    # Metric 7: zone count expansion
    b_zones = baseline['PULocationID'].nunique()
    f_zones = feb['PULocationID'].nunique()
    zone_ratio = f_zones / b_zones
    zone_alert = "CRITICAL" if zone_ratio > 2 else ("WARNING" if zone_ratio > 1.5 else "ok")
    results.append({
        'metric': 'zone count',
        'category': 'Data Drift',
        'baseline': str(b_zones),
        'current':  str(f_zones),
        'value': f"ratio={zone_ratio:.1f}x",
        'alert': zone_alert
    })

    # Metric 8: concept drift proxy — corr(trip_count, zone_slot_baseline)
    b_corr = baseline['trip_count'].corr(baseline['zone_slot_baseline'])
    f_corr = feb['trip_count'].corr(feb['zone_slot_baseline'])
    corr_drop = b_corr - f_corr
    corr_alert = "CRITICAL" if corr_drop > 0.10 else ("WARNING" if corr_drop > 0.05 else "ok")
    results.append({
        'metric': 'corr(trip_count, zone_slot_baseline)',
        'category': 'Concept Drift',
        'baseline': f"{b_corr:.4f}",
        'current':  f"{f_corr:.4f}",
        'value': f"drop={corr_drop:.4f}",
        'alert': corr_alert
    })

    return results

def main():
    print("Loading data...")
    baseline = pd.read_parquet(BASELINE_PATH)
    week4    = pd.read_parquet(WEEK4_PATH)
    feb = week4[(week4['time_bucket'] >= '2026-02-02') &
                (week4['time_bucket'] <= '2026-02-28 23:45:00')].copy()
    print(f"Baseline: {len(baseline):,} rows | Feb window: {len(feb):,} rows\n")

    results = run_metrics(baseline, feb)

    critical = [r for r in results if r['alert'] == 'CRITICAL']
    warning  = [r for r in results if r['alert'] == 'WARNING']

    print(f"{'Metric':<42} {'Category':<16} {'Value':<28} {'Alert'}")
    print("-" * 105)
    for r in results:
        print(f"{r['metric']:<42} {r['category']:<16} {r['value']:<28} {r['alert']}")

    print(f"\nSummary: {len(critical)} CRITICAL, {len(warning)} WARNING, "
          f"{len(results)-len(critical)-len(warning)} ok")

    if critical:
        print("\nCRITICAL alerts:")
        for r in critical:
            print(f"  [{r['category']}] {r['metric']}: {r['value']} "
                  f"(baseline={r['baseline']}, current={r['current']})")
        sys.exit(1)

if __name__ == "__main__":
    main()

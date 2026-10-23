"""
Phase 6 (arm B): decomposition-informed forecasting with Chronos-2.

Pipeline A (raw, `run_forecast.py`, label 'chronos2'): hr -> Chronos-2.
Pipeline B (this script, label 'decomp_acf'):
    1. STL decomposition of the CONTEXT window only (the model never sees
       ground truth). The seasonal period is estimated per series from the
       context window itself via ACF peak detection (see
       `estimate_period_acf`) - no DGP metadata is used, so the arm sees
       exactly the same information as pipeline A.
    2. Each component (trend / seasonal / resid) is forecast SEPARATELY by
       Chronos-2 with the same context length / horizon / quantiles.
    3. Recomposition: the quantile forecasts are summed component-wise
       (qX_total = qX_trend + qX_seasonal + qX_resid). Note: a quantile of a
       sum is not the sum of quantiles - the band assumes comonotone
       components and is therefore a (slightly conservative) approximation;
       the median is exact as a point forecast.

Both arms share the identical series, split and metrics code, so the
comparison in metrics_per_series.csv / summary_scenario.csv is paired.

Outputs in --out-dir:
    forecasts_decomp_acf_{scenario}.parquet  (long: series_id, t, actual, total q
        columns + per-component q columns + per-component medians)
    periods_decomp_acf_{scenario}.*     (per-series ACF-estimated STL period)
    metrics_per_series.csv               (updated per label - A/B-safe)
    summary_scenario.csv                 (median per scenario, all labels)
    manifest_decomp_acf.json             (model, STL settings, DGP hash)

Usage:
    python src/04_forecast/run_forecast_decomp.py            # all scenarios
    python src/04_forecast/run_forecast_decomp.py --max-series 5   # quick test
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf

from run_forecast import (TS_ORIGIN, DEFAULT_QUANTILES, load_scenario,
                          split_context, forecast_chronos, series_metrics,
                          save_df)


# -------------------------------------------------- seasonal period (ACF)

# Candidate range for the ultradian period (minutes), informed by the phase-02
# calibration (period_range [75, 448] in dgp_params_mmash.json). NOT read from
# DGP metadata. Periods close to the 512-min context length barely complete two
# cycles in-window and are hard to detect, but remain allowed as candidates.
PERIOD_MIN, PERIOD_MAX = 60, 448
PERIOD_FALLBACK = 254
ACF_MIN_PEAK = 0.10


def estimate_period_acf(y: np.ndarray,
                         min_period: int = PERIOD_MIN,
                         max_period: int = PERIOD_MAX,
                         fallback: int = PERIOD_FALLBACK,
                         min_peak_acf: float = ACF_MIN_PEAK) -> int:
    """Estimate the seasonal period from observed context data only.

    A quadratic trend is removed first (the trend channel contains circadian
    curvature, which a linear fit leaves behind and which distorts the ACF),
    then the strongest ACF peak inside [min_period, max_period] is used.
    Returns the fallback period if the series is too short, flat/noisy or no
    sufficiently strong peak is found. Does not use DGP metadata or future
    observations.
    """
    y = np.asarray(y, dtype=float)

    if len(y) < max_period + 30 or not np.all(np.isfinite(y)):
        return fallback

    t = np.arange(len(y), dtype=float)
    coeffs = np.polyfit(t, y, 2)
    detrended = y - np.polyval(coeffs, t)

    if np.std(detrended) < 1e-8:
        return fallback

    acf_values = acf(detrended, nlags=max_period, fft=True)
    candidate = acf_values[min_period:max_period + 1]
    peaks, _ = find_peaks(candidate)

    if len(peaks) == 0:
        return fallback

    best_peak = peaks[np.argmax(candidate[peaks])]
    best_lag = min_period + int(best_peak)
    best_strength = float(candidate[best_peak])

    if not np.isfinite(best_strength) or best_strength < min_peak_acf:
        return fallback

    return best_lag


# ------------------------------------------------------------------ STL

def stl_decompose(y: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STL on the context window; returns (trend, seasonal, resid)."""
    from statsmodels.tsa.seasonal import STL
    res = STL(pd.Series(y), period=period, robust=True).fit()
    return (res.trend.to_numpy(), res.seasonal.to_numpy(), res.resid.to_numpy())


def _stl_one(task: tuple) -> dict:
    """STL for one series (worker; picklable args for joblib processes)."""
    sid, t, y, period = task
    trend, seasonal, resid = stl_decompose(y, period)
    return {"sid": sid, "t": t, "trend": trend, "seasonal": seasonal,
            "resid": resid}


def component_frames(ctx: pd.DataFrame,
                     period_fallback: int = PERIOD_FALLBACK
                     ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """STL per series (parallel via joblib processes; STL holds the GIL, so
    processes are required for real parallelism). The seasonal period is
    estimated per series from the observed context via ACF
    (`estimate_period_acf`); no DGP metadata is used. Returns (per-component
    context DataFrames with columns series_id/t/hr, combined long DataFrame
    of the STL context decomposition with columns series_id/t/trend/seasonal/
    resid)."""
    from joblib import Parallel, delayed

    tasks, period_rows = [], []
    for sid, g in ctx.groupby("series_id"):
        g = g.sort_values("t")
        y = g["hr"].to_numpy(dtype=float)
        estimated_period = estimate_period_acf(y, fallback=period_fallback)
        period_rows.append({"series_id": sid, "estimated_period": estimated_period})
        tasks.append((sid, g["t"].to_numpy(), y, estimated_period))
    periods = pd.DataFrame(period_rows)
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(_stl_one)(task) for task in tasks)

    parts = {"trend": [], "seasonal": [], "resid": []}
    stl_rows = []
    for r in results:
        sid, t = r["sid"], r["t"]
        stl_rows.append(pd.DataFrame({"series_id": sid, "t": t,
                                      "trend": r["trend"],
                                      "seasonal": r["seasonal"],
                                      "resid": r["resid"]}))
        for name in ("trend", "seasonal", "resid"):
            parts[name].append(pd.DataFrame({"series_id": sid, "t": t,
                                             "hr": r[name]}))
    comps = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    stl_ctx = pd.concat(stl_rows, ignore_index=True)
    return comps, stl_ctx, periods


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Decomposition-informed Chronos-2 forecast (arm B)")
    ap.add_argument("--input-dir", default="data/synthetic")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--scenarios", nargs="+", default=["residual", "trend", "none"])
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=256)
    ap.add_argument("--quantiles", nargs="+", type=float, default=DEFAULT_QUANTILES)
    ap.add_argument("--model", default="amazon/chronos-2")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--max-series", type=int, default=None)
    args = ap.parse_args()

    input_dir, out_dir = Path(args.input_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = "decomp_acf"
    quantiles = sorted(args.quantiles)

    dgp = {}
    dgp_cfg = input_dir / "dgp_config_mmash.json"
    if dgp_cfg.exists():
        cfg = json.loads(dgp_cfg.read_text(encoding="utf-8"))
        dgp = {"dgp_hash": cfg.get("dgp_hash"),
               "params_sha256": cfg.get("params_sha256"),
               "generator_version": cfg.get("generator_version")}
        print(f"[DGP] config hash={dgp.get('dgp_hash')} adopted.")
    else:
        print("[DGP] dgp_config_mmash.json not found -> no hash in the manifest.")

    t_start = time.perf_counter()
    metric_rows, n_series_map = [], {}

    for scen in args.scenarios:
        df = load_scenario(input_dir, scen)
        ctx, truth = split_context(df, args.context, args.max_series)
        n = int(ctx["series_id"].nunique())
        horizon = min(args.horizon, len(truth) // max(n, 1))
        print(f"\n=== Scenario '{scen}': {n} series | context {args.context} | "
              f"horizon {horizon} | STL + Chronos-2 per component ===")

        t0 = time.perf_counter()
        comps, stl_ctx, periods = component_frames(ctx)
        n_fallback = int((periods["estimated_period"] == PERIOD_FALLBACK).sum())
        print(f"[ACF] periods: estimated {len(periods) - n_fallback}, "
              f"fallback {n_fallback} | "
              f"median {periods['estimated_period'].median():.0f} min")
        save_df(periods, out_dir / f"periods_{label}_{scen}")
        save_df(stl_ctx, out_dir / f"stl_context_{scen}")
        fc_comps = {}
        for name, cdf in comps.items():
            fc = forecast_chronos(cdf, horizon, quantiles,
                                  args.model, args.device)
            fc["t"] = args.context + fc.groupby("series_id").cumcount()
            fc = fc.sort_values(["series_id", "t"]).reset_index(drop=True)
            fc_comps[name] = fc
            print(f"[Decomp] component '{name}' forecast "
                  f"({fc.shape[0]} rows)")

        # recompose: quantile-wise sum over the components
        base = fc_comps["trend"][["series_id", "t"]].copy()
        for name in ("trend", "seasonal", "resid"):
            f = fc_comps[name]
            assert len(f) == len(base), "component frame length mismatch"
            for q in quantiles:
                base[f"{name}_q{q:g}"] = f[f"q{q:g}"].to_numpy()
        for q in quantiles:
            base[f"q{q:g}"] = sum(base[f"{name}_q{q:g}"].to_numpy()
                                  for name in ("trend", "seasonal", "resid"))
        fc = base
        fc = fc.merge(truth[["series_id", "t", "hr"]], on=["series_id", "t"],
                      how="inner")
        fc = fc.rename(columns={"hr": "actual"}).assign(scenario=scen)
        save_df(fc, out_dir / f"forecasts_{label}_{scen}")

        for sid, g in fc.groupby("series_id"):
            m = series_metrics(g["actual"].to_numpy(),
                               g[[f"q{q:g}" for q in quantiles]].to_numpy(),
                               quantiles)
            metric_rows.append({"label": label, "scenario": scen,
                                "series_id": int(sid), **m})
        n_series_map[scen] = n
        print(f"[Run] {label} on '{scen}' finished in "
              f"{time.perf_counter() - t0:.1f}s")

    metrics = pd.DataFrame(metric_rows)
    metrics_path = out_dir / "metrics_per_series.csv"
    if metrics_path.exists():
        old = pd.read_csv(metrics_path)
        metrics = pd.concat([old[old["label"] != label], metrics],
                           ignore_index=True)
    metrics.to_csv(metrics_path, index=False)

    summary = (metrics.groupby(["label", "scenario"])[["mae", "wql", "coverage"]]
               .median().reset_index())
    summary["n_series"] = metrics.groupby(["label", "scenario"]).size().to_numpy()
    summary["nominal"] = quantiles[-1] - quantiles[0]
    summary.to_csv(out_dir / "summary_scenario.csv", index=False)

    manifest = {
        "label": label,
        "model": args.model,
        "device": args.device or "auto (cuda if available)",
        "quantiles": quantiles,
        "context": args.context, "horizon": args.horizon,
        "max_series": args.max_series,
        "n_series": n_series_map,
        "scenarios": args.scenarios,
        "decomposition": {"method": "STL (statsmodels, robust)",
                          "seasonal_period": "ACF estimated from context "
                                             "window only (no DGP metadata)",
                          "period_search_range": [PERIOD_MIN, PERIOD_MAX],
                          "period_fallback": PERIOD_FALLBACK,
                          "acf_min_peak": ACF_MIN_PEAK},
        "recomposition": "quantile-wise sum (comonotone approximation)",
        "dgp": dgp,
        "runtime_sec": round(time.perf_counter() - t_start, 1),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path = out_dir / f"manifest_{label}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print("\n=== Summary (median per scenario, all labels) ===")
    print(summary.to_string(index=False))
    print(f"\n[Manifest] {manifest_path}")


if __name__ == "__main__":
    main()

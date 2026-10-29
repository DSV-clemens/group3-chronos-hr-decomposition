"""Position-controlled WindowSHAP attribution contrast.

Extends plot_event_attribution.py: the overall contrast (mean |phi| on event
windows / mean |phi| on clean windows, pooled over all 8 context windows) is
inflated by a position confound, because anomalies are injected only in the
second half of the context (windows 0-3 are always clean). This script computes
the per-window contrast c_w (event vs. clean within the same window index) and
the position-controlled contrast contrast_pc = mean of c_w over the windows in
which both classes occur (windows 4-7 in the residual scenario, windows 4-6 in the trend scenario, 
where window 7 contains an anomaly in all series and has no clean counterpart)

Differences between pipelines (decomp minus raw) are assessed with a paired
bootstrap: whole series are resampled with replacement (a series drawn k times
contributes k times to every groupby mean) and both contrasts are recomputed
per resample, mirroring analyze_uncertainty.py.

Inputs (produced by plot_event_attribution.py):
    results/windowshap_enriched_{scenario}.csv

Outputs:
    results/attribution_contrast.csv

Invocation:
    python src/05_analysis/attribution_contrast.py --results-dir results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SCENARIOS = ("residual", "trend")
VALUE_FUNCTIONS = ("phi_median", "phi_band")
PIPELINES = ("raw", "decomp")


def per_window_contrast(df: pd.DataFrame, value_function: str) -> float:
    """Mean of the per-window contrasts c_w over windows with both classes."""
    df = df.assign(abs_phi=df[value_function].abs())
    grouped = (df.groupby(["window", "has_event"])["abs_phi"]
                 .mean().unstack())
    shared = grouped.dropna()
    if shared.empty:
        return np.nan
    return float((shared[True] / shared[False]).mean())


def overall_contrast(df: pd.DataFrame, value_function: str) -> float:
    abs_phi = df[value_function].abs()
    ev, cl = abs_phi[df["has_event"]], abs_phi[~df["has_event"]]
    if cl.mean() == 0:
        return np.nan
    return float(ev.mean() / cl.mean())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Position-controlled attribution contrast with paired bootstrap.")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=67)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    rng = np.random.default_rng(args.seed)
    rows = []

    for scen in args.scenarios:
        path = results_dir / f"windowshap_enriched_{scen}.csv"
        if not path.exists():
            print(f"[skip] {scen}: missing {path.name}")
            continue
        enriched = pd.read_csv(path)
        series_ids = np.sort(enriched["series_id"].unique())

        point = {}
        for pipe in PIPELINES:
            sub = enriched[enriched.pipeline == pipe]
            for val in VALUE_FUNCTIONS:
                point[(pipe, val)] = (
                    overall_contrast(sub, val),
                    per_window_contrast(sub, val),
                )

        # Paired bootstrap over whole series; recompute both contrasts per resample.
        # A series drawn k times contributes k copies to the resample, so groupby
        # means weight it k times (same semantics as analyze_uncertainty.py).
        # Pre-index per-series frames once for speed (concat of 150 small frames
        # per resample is much faster than filtering the full frame per series).
        by_pipe = {pipe: enriched[enriched.pipeline == pipe] for pipe in PIPELINES}
        series_frames = {
            pipe: {s: by_pipe[pipe][by_pipe[pipe].series_id == s] for s in series_ids}
            for pipe in PIPELINES
        }
        diffs = {val: np.empty(args.bootstrap_reps) for val in VALUE_FUNCTIONS}
        for i in range(args.bootstrap_reps):
            sample = rng.choice(series_ids, size=len(series_ids), replace=True)
            for val in VALUE_FUNCTIONS:
                pc = []
                for pipe in PIPELINES:
                    resampled = pd.concat([series_frames[pipe][s] for s in sample])
                    pc.append(per_window_contrast(resampled, val))
                diffs[val][i] = pc[1] - pc[0]

        for val in VALUE_FUNCTIONS:
            low, high = np.quantile(diffs[val], [0.025, 0.975])
            rows.append({
                "scenario": scen, "value_function": val,
                "overall_raw": point[("raw", val)][0],
                "overall_decomp": point[("decomp", val)][0],
                "pc_raw": point[("raw", val)][1],
                "pc_decomp": point[("decomp", val)][1],
                "pc_diff_mean": float(diffs[val].mean()),
                "pc_diff_ci95_lower": float(low),
                "pc_diff_ci95_upper": float(high),
                "n_series": len(series_ids),
                "bootstrap_reps": args.bootstrap_reps,
                "seed": args.seed,
            })
            print(f"[{scen} {val}] overall {point[('raw', val)][0]:.2f} -> "
                  f"{point[('decomp', val)][0]:.2f} | "
                  f"position-controlled {point[('raw', val)][1]:.3f} -> "
                  f"{point[('decomp', val)][1]:.3f} "
                  f"(Delta {diffs[val].mean():+.3f} [{low:+.3f}, {high:+.3f}])")

    if not rows:
        raise SystemExit("no data found")
    out = results_dir / "attribution_contrast.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
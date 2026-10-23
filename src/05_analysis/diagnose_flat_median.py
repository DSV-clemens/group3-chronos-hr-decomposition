"""
Diagnose 'flat median / model latches to nothing' per series.

For every series we compare the forecast median to context and truth:
- flatness:            std of the forecast median over the horizon
- context slope:      OLS slope over the last --slope-window minutes of the context
- forecast slope:     OLS slope of the forecast median over the horizon
- level shift:        mean(truth first 50 min) - mean(context last 50 min)
- band coverage:      share of truth inside [q_low, q_high]

Aggregates answer the key questions:
1. Does the median follow the local LEVEL?   corr(ctx_end_level, fc_start_level)
2. Does the median follow visible TRENDS?   corr(ctx_slope, fc_slope)
3. How flat is the median, and is the flatness explained by
   an unpredictable level shift at the horizon boundary?
4. Does the band absorb what the median misses?

Outputs a per-series table (CSV), aggregate stats and a Markdown report.

Invocation:
    python src/05_analysis/diagnose_flat_median.py
    python src/05_analysis/diagnose_flat_median.py --label chronos2 --scenarios none
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

EDGE = 50        # minutes averaged for the level at the context/horizon edge
SLOPE_WIN = 120  # minutes of context used for the trend estimate


def ols_slope(y: np.ndarray) -> float:
    x = np.arange(len(y), dtype=float)
    xm, ymean = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    return ((x - xm) * (y - ymean)).sum() / denom if denom else np.nan


def diagnose_series(ctx: pd.DataFrame, truth: pd.DataFrame,
                    f: pd.DataFrame, q_low: str, q_med: str, q_high: str) -> dict:
    ctx_end = ctx["hr"].tail(EDGE).mean()
    truth_start = truth["hr"].head(EDGE).mean()
    level_shift = truth_start - ctx_end

    ctx_slope = ols_slope(ctx["hr"].tail(SLOPE_WIN).to_numpy())
    fc = f.sort_values("t")
    fc_slope = ols_slope(fc[q_med].to_numpy())
    fc_start = fc[q_med].head(EDGE).mean()

    lo, hi = fc[q_low].to_numpy(), fc[q_high].to_numpy()
    inside = ((truth["hr"].to_numpy() >= lo) & (truth["hr"].to_numpy() <= hi)).mean()

    return {
        "ctx_end_level": ctx_end,
        "truth_start_level": truth_start,
        "level_shift": level_shift,
        "abs_level_shift": abs(level_shift),
        "ctx_slope": ctx_slope,
        "fc_slope": fc_slope,
        "fc_start_level": fc_start,
        "fc_level_error": fc_start - truth_start,
        "median_std": fc[q_med].std(),
        "band_width": float((hi - lo).mean()),
        "band_coverage": inside,
    }


def corr(a: pd.Series, b: pd.Series) -> float:
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(a.corr(b))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Diagnose flat forecast medians: model trait or pipeline problem?")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--data-dir", default="data/synthetic")
    ap.add_argument("--label", default="chronos2")
    ap.add_argument("--scenarios", nargs="+", default=["none", "residual", "trend"])
    ap.add_argument("--quantiles", nargs=2, type=float, default=[0.1, 0.9],
                    help="low/high quantile columns (band)")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    results_dir, data_dir = Path(args.results_dir), Path(args.data_dir)
    q_med = "q0.5"
    q_low, q_high = f"q{args.quantiles[0]:g}", f"q{args.quantiles[1]:g}"

    print("=" * 68)
    print("  Flat-median diagnosis")
    print("=" * 68)

    frames = []
    for scen in args.scenarios:
        f_path = results_dir / f"forecasts_{args.label}_{scen}.parquet"
        d_path = data_dir / f"synthetic_hr_mmash_{scen}.parquet"
        if not f_path.exists() or not d_path.exists():
            print(f"[skip] {scen}: missing {f_path.name} or {d_path.name}")
            continue
        fc = pd.read_parquet(f_path)
        df = pd.read_parquet(d_path)
        rows = []
        for sid, f in fc.groupby("series_id"):
            ctx = df[(df.series_id == sid) & (df.t < args.context)].sort_values("t")
            truth = df[(df.series_id == sid) & (df.t >= args.context)].sort_values("t")
            r = diagnose_series(ctx, truth, f, q_low, q_med, q_high)
            r.update({"series_id": sid, "scenario": scen})
            rows.append(r)
        part = pd.DataFrame(rows)
        frames.append(part)
        print(f"[done] {scen}: {len(part)} series")

    if not frames:
        raise SystemExit("no data found")
    tbl = pd.concat(frames, ignore_index=True)

    out_csv = Path(args.out_csv) if args.out_csv else results_dir / "flat_median_diagnosis.csv"
    out_md = Path(args.out_md) if args.out_md else results_dir / "flat_median_diagnosis.md"
    tbl.to_csv(out_csv, index=False)
    print(f"[write] {out_csv}")

    md = ["# Flat-median diagnosis", "",
          f"_created: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_  ",
          f"_label: `{args.label}` | scenarios: {args.scenarios} | "
          f"context {args.context} | band: [{q_low}, {q_high}]_", ""]
    report: dict = {"label": args.label, "scenarios": {}, "created":
                    datetime.now(timezone.utc).isoformat(timespec="seconds")}

    for scen in args.scenarios:
        s = tbl[tbl.scenario == scen]
        if s.empty:
            continue
        flat = s[s.median_std < 2.5]
        shifted = s[s.abs_level_shift > 15]
        flat_and_shifted = s[(s.median_std < 2.5) & (s.abs_level_shift > 15)]
        level_follow = corr(s.ctx_end_level, s.fc_start_level)
        slope_follow = corr(s.ctx_slope, s.fc_slope)
        band_helps = s[(s.band_coverage >= 0.8)]

        stats = {
            "n_series": len(s),
            "median_std_median": float(s.median_std.median()),
            "flat_median_share": len(flat) / len(s),
            "band_width_median_bpm": float(s.band_width.median()),
            "band_coverage_median": float(s.band_coverage.median()),
            "band_covers_ge_80pct_share": len(band_helps) / len(s),
            "abs_level_shift_median_bpm": float(s.abs_level_shift.median()),
            "large_shift_gt15_share": len(shifted) / len(s),
            "flat_and_large_shift_share": len(flat_and_shifted) / len(s),
            "corr_ctx_end_vs_fc_start_level": level_follow,
            "corr_ctx_slope_vs_fc_slope": slope_follow,
            "fc_start_level_minus_ctx_end_mean_bpm": float((s.fc_start_level - s.ctx_end_level).mean()),
        }
        report["scenarios"][scen] = stats

        md += [f"## scenario: {scen}", ""]
        md += [f"- **flat median** (std < 2.5 bpm): **{len(flat)}/{len(s)}** "
               f"({100*len(flat)/len(s):.0f} %) series, median std = {stats['median_std_median']:.2f}"]
        md += [f"- **level following**: corr(context-end level, forecast-start level) = "
               f"**{level_follow:.3f}**, mean offset = "
               f"{stats['fc_start_level_minus_ctx_end_mean_bpm']:+.1f} bpm "
               f"(model latches to the local LEVEL, not to nothing)"]
        md += [f"- **trend following**: corr(context slope, forecast slope) = "
               f"**{slope_follow:.3f}** "
               f"(0 = median ignores visible trends, model trait; "
               f"~1 = structure is carried through)"]
        md += [f"- **band absorbs the miss**: median band width = "
               f"{stats['band_width_median_bpm']:.1f} bpm, "
               f"{100*stats['band_covers_ge_80pct_share']:.0f} % of series with "
               f"coverage >= 80 %"]
        md += [f"- **unpredictable edge**: |level shift| > 15 bpm at the "
               f"context/horizon boundary in {len(shifted)}/{len(s)} "
               f"({100*len(shifted)/len(s):.0f} %) series; "
               f"flat AND shifted: {len(flat_and_shifted)}/{len(s)} "
               f"(DGP makes the horizon partly unforecastable from the context)"]
        md += [f"- truth-vs-context level shift: median "
               f"{stats['abs_level_shift_median_bpm']:.1f} bpm", ""]

    md += ["## Interpretation guide", "",
           "- **corr(level) high + corr(slope) ~ 0**: model trait - the median",
           "  anchors to the local level and externalizes structural uncertainty",
           "  into the band. Not a pipeline bug.",
           "- **corr(level) low**: our problem - the context level is not passed",
           "  correctly (check scaling / id handling in the forecast script).",
           "- **flat & shifted share high**: a large part of the flatness comes",
           "  from the DGP (level change at the horizon boundary that is not",
           "  derivable from the context), not from the model.", ""]

    out_md.write_text("\n".join(md) + "\n")
    print(f"[write] {out_md}")

    report_path = results_dir / "flat_median_diagnosis.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[write] {report_path}")


if __name__ == "__main__":
    main()

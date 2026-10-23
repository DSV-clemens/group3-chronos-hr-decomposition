"""
Horizon-resolved evaluation: MAE / coverage / band width per horizon bucket.

Answers the question "WHERE does each arm win or lose?" - e.g. the naive
persistence baseline is hard to beat at short lead times (strong AR(1)
persistence of the DGP) but is overtaken at longer ranges, while the
foundation model wins the short range. A single horizon-averaged MAE hides
this structure.

Reads the per-label forecast parquets written by the phase-04 arms
(chronos2 / naive / dlinear / decomp_acf) and writes:
    metrics_by_horizon.csv     (long: label, scenario, bucket, mae, coverage,
                                band width, n)
    and prints the pivot tables.

Usage:
    python src/05_analysis/metrics_by_horizon.py
    python src/05_analysis/metrics_by_horizon.py --labels chronos2 naive
        --buckets 32 64 128 256
    python src/05_analysis/metrics_by_horizon.py --series-min 100
        (restrict to held-out series, e.g. to pair with the dlinear arm)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_forecast(results_dir: Path, label: str, scenario: str) -> pd.DataFrame | None:
    base = results_dir / f"forecasts_{label}_{scenario}"
    for ext, reader in ((".parquet", "read_parquet"), (".csv.gz", "read_csv")):
        p = Path(str(base) + ext)
        if p.exists():
            return getattr(pd, reader)(p)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="MAE / coverage by horizon bucket")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--scenarios", nargs="+", default=["residual", "trend", "none"])
    ap.add_argument("--labels", nargs="+",
                    default=["chronos2", "naive", "dlinear", "decomp_acf"])
    ap.add_argument("--buckets", nargs="+", type=int, default=[32, 64, 128, 256],
                    help="bucket edges in horizon steps (cumulative, last = horizon)")
    ap.add_argument("--series-min", type=int, default=None,
                    help="only series_id >= this (e.g. 100 for the dlinear "
                         "held-out split, so all arms are compared on the "
                         "same series)")
    ap.add_argument("--out", default="results/metrics_by_horizon.csv")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    rows = []
    for scen in args.scenarios:
        for label in args.labels:
            fc = load_forecast(results_dir, label, scen)
            if fc is None:
                print(f"[Skip] {label}/{scen}: no forecasts found")
                continue
            if args.series_min is not None:
                fc = fc[fc["series_id"] >= args.series_min]
            if fc.empty:
                continue
            fc = fc.sort_values(["series_id", "t"])
            fc["h"] = fc.groupby("series_id").cumcount()
            edges = [e for e in args.buckets if e <= fc["h"].max() + 1]
            fc["err"] = (fc["q0.5"] - fc["actual"]).abs()
            fc["band"] = fc["q0.9"] - fc["q0.1"]
            fc["in_band"] = (fc["actual"] >= fc["q0.1"]) & (fc["actual"] <= fc["q0.9"])
            labels_ = [f"h<= {e}" for e in edges]
            # bins start at -1 so h=0 (first horizon step) is included in the
            # first bucket; with bins starting at 0 and right=True, h=0 would
            # fall out of every bucket (pd.cut bins are (left, right]).
            fc["bucket"] = pd.Categorical(
                pd.cut(fc["h"], [-1] + edges, right=True, labels=labels_),
                categories=labels_, ordered=True)
            for b, g in fc.groupby("bucket", observed=True):
                rows.append({"label": label, "scenario": scen, "bucket": str(b),
                             "mae": float(g["err"].mean()),
                             "coverage": float(g["in_band"].mean()),
                             "band_width": float(g["band"].mean()),
                             "n": int(len(g))})
    if not rows:
        raise SystemExit("No forecasts found for the requested labels/scenarios.")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"[Saved] {args.out}\n")

    for scen in args.scenarios:
        sub = out[out.scenario == scen]
        if sub.empty:
            continue
        print(f"=== {scen}: mean MAE by horizon bucket ===")
        piv = sub.pivot_table(index="bucket", columns="label", values="mae")
        piv = piv[[l for l in args.labels if l in piv.columns]]
        print(piv.round(2).to_string())
        print()
        print(f"=== {scen}: coverage by horizon bucket ===")
        piv = sub.pivot_table(index="bucket", columns="label", values="coverage")
        piv = piv[[l for l in args.labels if l in piv.columns]]
        print(piv.round(3).to_string())
        print()


if __name__ == "__main__":
    main()

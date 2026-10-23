"""Evaluate saved probabilistic HR forecasts; 

Primary contrast: decomp_acf - chronos2 (negative interval-score difference is better).
Bootstrap resamples series IDs within each scenario, not forecast minutes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SCENARIOS = ("none", "residual", "trend")
METHODS = ("chronos2", "decomp_acf", "dlinear")
ALPHA = 0.2
QUANTILES = (0.1, 0.5, 0.9)


def load_forecasts(results_dir: Path) -> pd.DataFrame:
    frames = []
    for scenario in SCENARIOS:
        for method in METHODS:
            path = results_dir / f"forecasts_{method}_{scenario}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}")
            df = pd.read_parquet(path)
            required = ["series_id", "t", "actual", "q0.1", "q0.5", "q0.9"]
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise ValueError(f"{path}: missing columns {missing}")
            df = df[required].copy()
            if df.duplicated(["series_id", "t"]).any():
                raise ValueError(f"{path}: duplicate series_id/t rows")
            if df[required[2:]].isna().any().any():
                raise ValueError(f"{path}: missing forecast/actual values")
            if not np.isfinite(df[required[2:]].to_numpy(dtype=float)).all():
                raise ValueError(f"{path}: nonfinite forecast/actual values")
            if (df["q0.1"] > df["q0.5"]).any() or (df["q0.5"] > df["q0.9"]).any():
                raise ValueError(f"{path}: crossed quantiles")
            df["scenario"] = scenario
            df["method"] = method
            frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    # Require exactly the same (scenario, series_id, t) and actual for each method.
    ref = data.loc[data.method == "chronos2", ["scenario", "series_id", "t", "actual"]]
    for method in METHODS[1:]:
        other = data.loc[data.method == method, ["scenario", "series_id", "t", "actual"]]
        paired = ref.merge(other, on=["scenario", "series_id", "t"], how="outer",
                           suffixes=("_direct", "_other"), indicator=True,
                           validate="one_to_one")
        if not paired["_merge"].eq("both").all():
            raise ValueError(f"{method}: forecast indices are not fully paired")
        if not np.allclose(paired.actual_direct, paired.actual_other, rtol=0, atol=1e-10):
            raise ValueError(f"{method}: actual values differ from direct Chronos-2")
    return data


def add_scores(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    y = df["actual"].to_numpy(dtype=float)
    lo = df["q0.1"].to_numpy(dtype=float)
    md = df["q0.5"].to_numpy(dtype=float)
    hi = df["q0.9"].to_numpy(dtype=float)
    df["covered"] = ((y >= lo) & (y <= hi)).astype(float)
    df["interval_width"] = hi - lo
    df["interval_score"] = (hi - lo) + (2 / ALPHA) * np.maximum(lo - y, 0) + (2 / ALPHA) * np.maximum(y - hi, 0)
    df["absolute_error"] = np.abs(y - md)
    pinballs = []
    for q in QUANTILES:
        err = y - df[f"q{q:g}"].to_numpy(dtype=float)
        pinballs.append(np.maximum(q * err, (q - 1) * err))
    df["sum_pinball"] = np.sum(pinballs, axis=0)
    df["absolute_actual"] = np.abs(y)
    df["horizon"] = df["t"] - df.groupby(["method", "scenario", "series_id"])["t"].transform("min") + 1
    df["horizon_block"] = pd.cut(df["horizon"], bins=[0, 64, 128, 192, 256],
                                  labels=["1-64", "65-128", "129-192", "193-256"])
    if df["horizon_block"].isna().any():
        raise ValueError("Forecast horizons outside 1-256; adjust block boundaries")
    return df


def aggregate(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = df.groupby(keys, observed=True, sort=True).agg(
        n_observations=("covered", "size"),
        coverage=("covered", "mean"),
        mpiw=("interval_width", "mean"),
        interval_score=("interval_score", "mean"),
        mae=("absolute_error", "mean"),
        pinball_sum=("sum_pinball", "sum"),
        actual_abs_sum=("absolute_actual", "sum"),
    ).reset_index()
    # Match the repository's original normalized weighted quantile loss.
    out["wql"] = 2 * out.pop("pinball_sum") / (len(QUANTILES) * out.pop("actual_abs_sum").clip(lower=1e-9))
    out["coverage_gap"] = out["coverage"] - (1 - ALPHA)
    out["abs_coverage_gap"] = out["coverage_gap"].abs()
    return out


def paired_differences(per_series: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metrics = ["coverage", "mpiw", "interval_score", "mae", "wql"]
    rows = []
    for scenario in SCENARIOS:
        subset = per_series[per_series.scenario == scenario]
        left = subset[subset.method == "chronos2"].set_index("series_id")
        right = subset[subset.method == "decomp_acf"].set_index("series_id")
        if set(left.index) != set(right.index):
            raise ValueError(f"{scenario}: series IDs differ")
        right = right.loc[left.index]
        for metric in metrics:
            differences = (right[metric] - left[metric]).to_numpy(dtype=float)
            # One observation per series in bootstrap: do not resample minutes.
            means = np.empty(n_boot)
            for i in range(n_boot):
                means[i] = rng.choice(differences, size=len(differences), replace=True).mean()
            low, high = np.quantile(means, [0.025, 0.975])
            rows.append({"scenario": scenario, "metric": metric,
                         "comparison": "decomp_acf minus chronos2",
                         "n_series": len(differences),
                         "mean_paired_difference": differences.mean(),
                         "ci95_lower": low, "ci95_upper": high})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/uncertainty"))
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.bootstrap_reps < 100:
        parser.error("--bootstrap-reps must be >= 100")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = add_scores(load_forecasts(args.results_dir))
    overall = aggregate(df, ["method", "scenario"])
    per_series = aggregate(df, ["method", "scenario", "series_id"])
    by_horizon = aggregate(df, ["method", "scenario", "horizon_block"])
    paired = paired_differences(per_series, args.bootstrap_reps, args.seed)

    outputs = {
        "uncertainty_overall.csv": overall,
        "uncertainty_per_series.csv": per_series,
        "uncertainty_by_horizon.csv": by_horizon,
        "uncertainty_paired_bootstrap.csv": paired,
    }
    for filename, frame in outputs.items():
        path = args.out_dir / filename
        frame.to_csv(path, index=False)
        print(f"Saved {path} ({len(frame)} rows)")
    print("\nOverall scores (coverage target = 0.80; lower interval score is better):")
    print(overall[["method", "scenario", "coverage", "mpiw", "interval_score", "wql"]].round(4).to_string(index=False))
    print("\nNote: bootstrap CIs assume independent series within each scenario; verify DGP dependence before inferential claims.")


if __name__ == "__main__":
    main()

"""
Uncertainty quantification figures (from the notebook by Alexandra Kordali).

Reads the CSVs written by src/05_analysis/analyze_uncertainty.py and renders
the five figures of the uncertainty analysis as PNGs into plots/:

  1. uq_coverage.png        - 80% interval coverage by model x scenario
  2. uq_interval_width.png  - mean prediction interval width (MPIW)
  3. uq_interval_score.png  - interval score (lower is better)
  4. uq_paired_bootstrap.png - paired interval-score difference (STL - Direct)
                              with 95% bootstrap confidence intervals
  5. uq_horizon_effect.png  - interval-score difference per horizon block

Usage:
    python src/05_analysis/plot_uncertainty.py
    python src/05_analysis/plot_uncertainty.py --results-dir results \\
        --out-dir plots
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCENARIOS = ("none", "residual", "trend")
SCENARIO_LABELS = ("None", "Residual", "Trend")
METHODS = ("chronos2", "decomp_acf", "dlinear")
METHOD_NAMES = {"chronos2": "Direct Chronos-2",
                "decomp_acf": "STL + Chronos-2",
                "dlinear": "DLinear-style"}
COLORS = {"chronos2": "#3679B5", "decomp_acf": "#E18A35", "dlinear": "#339878"}
SCEN_COLORS = ("#3679B5", "#E18A35", "#339878")


def _read(results_dir: Path, name: str) -> pd.DataFrame:
    path = results_dir / "uncertainty" / name
    if not path.exists():
        raise SystemExit(
            f"{path} not found - run src/05_analysis/analyze_uncertainty.py "
            f"first (it writes results/uncertainty/*.csv)")
    return pd.read_csv(path)


def _grouped_bars(ax, overall: pd.DataFrame, metric: str) -> None:
    x = np.arange(len(SCENARIOS))
    width = 0.25
    for i, method in enumerate(METHODS):
        values = [overall.loc[(overall["method"] == method)
                              & (overall["scenario"] == scen), metric].iloc[0]
                  for scen in SCENARIOS]
        bars = ax.bar(x + (i - 1) * width, values, width,
                      label=METHOD_NAMES[method], color=COLORS[method])
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)


def plot_coverage(overall: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped_bars(ax, overall * 1 if "coverage" in overall else overall,
                  "coverage")
    # coverage is a fraction; convert axis to percent
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels([f"{t * 100:.0f}" for t in ax.get_yticks()])
    ax.axhline(0.8, color="black", linestyle="--", linewidth=1.5,
               label="Target: 80%")
    ax.set_xticks(np.arange(len(SCENARIOS)), SCENARIO_LABELS)
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("80% Prediction Interval Coverage")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_interval_width(overall: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped_bars(ax, overall, "mpiw")
    ax.set_xticks(np.arange(len(SCENARIOS)), SCENARIO_LABELS)
    ax.set_ylabel("Mean interval width (bpm)")
    ax.set_title("Prediction Interval Width by Model and Scenario")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_interval_score(overall: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped_bars(ax, overall, "interval_score")
    ax.set_xticks(np.arange(len(SCENARIOS)), SCENARIO_LABELS)
    ax.set_ylabel("Mean interval score (lower is better)")
    ax.set_title("Probabilistic Forecast Quality: Interval Score")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_paired_bootstrap(paired: pd.DataFrame, out: Path) -> None:
    subset = paired[paired["metric"] == "interval_score"].copy()
    subset["scenario"] = pd.Categorical(subset["scenario"],
                                        categories=SCENARIOS, ordered=True)
    subset = subset.sort_values("scenario")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (_, r) in enumerate(subset.iterrows()):
        mean = r["mean_paired_difference"]
        lower, upper = r["ci95_lower"], r["ci95_upper"]
        ax.errorbar(mean, i, xerr=[[mean - lower], [upper - mean]],
                    fmt="o", markersize=8, capsize=5, linewidth=2,
                    color=COLORS["decomp_acf"])
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_yticks(range(len(SCENARIO_LABELS)), SCENARIO_LABELS)
    ax.invert_yaxis()
    ax.set_xlabel("Paired interval-score difference (STL \u2212 Direct)")
    ax.set_title("95% Bootstrap Confidence Intervals")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_horizon_effect(horizon: pd.DataFrame, out: Path) -> None:
    blocks = ["1-64", "65-128", "129-192", "193-256"]
    block_labels = ["1\u201364", "65\u2013128", "129\u2013192", "193\u2013256"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for scen, color in zip(SCENARIOS, SCEN_COLORS):
        direct = (horizon[(horizon["method"] == "chronos2")
                          & (horizon["scenario"] == scen)]
                  .set_index("horizon_block").reindex(blocks)["interval_score"])
        decomposed = (horizon[(horizon["method"] == "decomp_acf")
                               & (horizon["scenario"] == scen)]
                      .set_index("horizon_block").reindex(blocks)["interval_score"])
        ax.plot(block_labels, decomposed - direct, marker="o", linewidth=2,
                markersize=7, label=scen.capitalize(), color=color)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel("STL interval score \u2212 Direct interval score")
    ax.set_title("Effect of STL Decomposition Across Forecast Horizons")
    ax.legend(title="Scenario")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Uncertainty quantification figures (PNG)")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out-dir", type=Path, default=Path("plots"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    overall = _read(args.results_dir, "uncertainty_overall.csv")
    paired = _read(args.results_dir, "uncertainty_paired_bootstrap.csv")
    horizon = _read(args.results_dir, "uncertainty_by_horizon.csv")

    plot_coverage(overall, args.out_dir / "uq_coverage.png")
    plot_interval_width(overall, args.out_dir / "uq_interval_width.png")
    plot_interval_score(overall, args.out_dir / "uq_interval_score.png")
    plot_paired_bootstrap(paired, args.out_dir / "uq_paired_bootstrap.png")
    plot_horizon_effect(horizon, args.out_dir / "uq_horizon_effect.png")
    print(f"[Saved] 5 figures -> {args.out_dir}/uq_*.png")


if __name__ == "__main__":
    main()

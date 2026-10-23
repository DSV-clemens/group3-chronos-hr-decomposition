"""
Phase 5: Visualization of the forecast results.

One figure per series: context (blue), true continuation (black dashed),
forecast median + quantile band (crimson), anomaly events shaded, context/
horizon boundary at t=512. The title shows the series metrics (MAE, WQL,
coverage vs. nominal).

Additionally:
- overview grid per scenario (first 6 series, compact)
- metric comparison across all series (boxplots MAE / WQL / coverage per scenario)

Invocation:
    python plot_forecast_results.py                                  # first 6 series per scenario
    python plot_forecast_results.py --label chronos2                 # label of the run
    python plot_forecast_results.py --series 0 7 23                 # specific series
    python plot_forecast_results.py --label naive                    # baseline plots

Expected in --input-dir (= results/, from run_forecast.py):
    forecasts_{label}_{scenario}.parquet   (forecasts)
    metrics_per_series.csv                 (series metrics, optional)
Expected in --data-dir (= data/synthetic/, from generate_hr_mmash.py):
    synthetic_hr_mmash_{scenario}.parquet  (original series, for the context)
    metadata_{scenario}.jsonl              (events, optional)
Output: PNGs in --out-dir (default: plots/)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEV_COLOR = {"subtle": "#ffe08a", "moderate": "#ffab40", "severe": "#ef5350"}
QUANTILE_ALPHA = {"outer": 0.15, "inner": 0.30}


# ------------------------------------------------------------------ loading

def _read_any(base: Path) -> pd.DataFrame | None:
    """parquet before csv.gz."""
    for ext, reader in ((".parquet", "read_parquet"), (".csv.gz", "read_csv")):
        p = Path(str(base) + ext)
        if p.exists():
            return getattr(pd, reader)(p)
    return None


def load_inputs(input_dir: Path, data_dir: Path, label: str, scenario: str):
    fc = _read_any(input_dir / f"forecasts_{label}_{scenario}")
    if fc is None:
        raise FileNotFoundError(f"forecasts_{label}_{scenario}(.parquet/.csv.gz) "
                                f"not in {input_dir} -- run run_forecast.py first.")
    series = _read_any(data_dir / f"synthetic_hr_mmash_{scenario}")
    if series is None:
        series = _read_any(data_dir / f"synthetic_hr_{scenario}")
    if series is None:  # fallback for old layouts (everything in one folder)
        series = _read_any(input_dir / f"synthetic_hr_mmash_{scenario}")
    events: dict[int, list[dict]] = {}
    meta_p = data_dir / f"metadata_{scenario}.jsonl"
    if not meta_p.exists():
        meta_p = input_dir / f"metadata_{scenario}.jsonl"
    if meta_p.exists():
        with open(meta_p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    m = json.loads(line)
                    events[int(m["series_id"])] = m.get("events", [])
    return fc, series, events


def load_metrics(input_dir: Path, label: str) -> pd.DataFrame | None:
    p = input_dir / "metrics_per_series.csv"
    if not p.exists():
        return None
    m = pd.read_csv(p)
    return m[m["label"] == label]


# ------------------------------------------------------------------ single plot

def plot_one(ctx, truth, fc_s, events, title, out_path, nominal=None) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    # Shade events (in the context/horizon region)
    for e in events:
        ax.axvspan(e["start"], e["end"], color=SEV_COLOR.get(e["severity"], "#cccccc"),
                   alpha=0.35, lw=0, zorder=0)

    # Context + true continuation
    ax.plot(ctx["t"], ctx["hr"], color="steelblue", lw=1.0, label="Context (model input)")
    ax.plot(truth["t"], truth["hr"], color="black", lw=1.0, ls="--",
            label="True continuation")

    # Forecast: outer + inner band (if available)
    qcols = sorted([c for c in fc_s.columns if c.startswith("q")],
                   key=lambda c: float(c[1:]))
    if len(qcols) >= 2:
        lo, hi = qcols[0], qcols[-1]
        ax.fill_between(fc_s["t"], fc_s[lo], fc_s[hi], color="crimson",
                        alpha=QUANTILE_ALPHA["outer"],
                        label=f"Band q{float(lo[1:]):g}\u2013q{float(hi[1:]):g}")
    if len(qcols) >= 4:  # e.g. 0.1/0.25/0.75/0.9
        ax.fill_between(fc_s["t"], fc_s[qcols[1]], fc_s[qcols[-2]], color="crimson",
                        alpha=QUANTILE_ALPHA["inner"], label="inner band")
    if "q0.5" in fc_s.columns:
        ax.plot(fc_s["t"], fc_s["q0.5"], color="crimson", lw=1.6,
                label="Forecast median")

    # Context/horizon boundary
    split = int(truth["t"].min())
    ax.axvline(split - 0.5, color="grey", ls=":", lw=1)
    ax.text(split + 4, ax.get_ylim()[0], "Horizon \u2192", fontsize=9, color="grey",
            va="bottom")

    ax.set_xlabel("t [min]")
    ax.set_ylabel("Heart rate [bpm]")
    suffix = f" | nominal {nominal:.2f}" if nominal is not None else ""
    ax.set_title(f"{title}{suffix}")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ grid + metrics

def plot_grid(rows, scenario, label, out_path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
    for ax, (sid, ax_data) in zip(axes.flat, rows):
        plot_data_on_ax(ax, **ax_data)
        ax.set_title(f"Series {sid}", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("t [min]", fontsize=9)
    fig.suptitle(f"Forecast overview | scenario={scenario} | label={label}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_data_on_ax(ax, ctx, truth, fc_s, events) -> None:
    for e in events:
        ax.axvspan(e["start"], e["end"], color=SEV_COLOR.get(e["severity"], "#cccccc"),
                   alpha=0.35, lw=0, zorder=0)
    ax.plot(ctx["t"], ctx["hr"], color="steelblue", lw=0.8)
    ax.plot(truth["t"], truth["hr"], color="black", lw=0.8, ls="--")
    qcols = sorted([c for c in fc_s.columns if c.startswith("q")],
                   key=lambda c: float(c[1:]))
    if len(qcols) >= 2:
        ax.fill_between(fc_s["t"], fc_s[qcols[0]], fc_s[qcols[-1]], color="crimson",
                        alpha=0.2, lw=0)
    if "q0.5" in fc_s.columns:
        ax.plot(fc_s["t"], fc_s["q0.5"], color="crimson", lw=1.2)
    ax.axvline(int(truth["t"].min()) - 0.5, color="grey", ls=":", lw=0.8)


def plot_metrics_box(metrics: pd.DataFrame, label: str, out_path: str) -> None:
    """Boxplots MAE / WQL / coverage per scenario across all series."""
    scenarios = sorted(metrics["scenario"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    specs = [("mae", "MAE (median) [bpm]", None),
             ("wql", "WQL (weighted quantile loss)", None),
             ("coverage", "Coverage (empirical)", "nominal")]
    for ax, (col, ylabel, ref) in zip(axes, specs):
        data = [metrics[metrics["scenario"] == s][col].dropna().values for s in scenarios]
        bp = ax.boxplot(data, tick_labels=scenarios, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#bcd6f0")
        if ref == "nominal":
            nominal = metrics["nominal"].dropna().iloc[0] if "nominal" in metrics else None
            if nominal is not None:
                ax.axhline(nominal, color="crimson", ls="--", lw=1,
                           label=f"nominal ({nominal:.2f})")
                ax.legend(fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(col.upper(), fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Metrics per scenario | label={label} | n={len(metrics)} series runs",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize forecast results")
    ap.add_argument("--input-dir", default="results",
                    help="Folder with forecasts_* and metrics_per_series.csv")
    ap.add_argument("--data-dir", default="data/synthetic",
                    help="Folder with the original series + metadata")
    ap.add_argument("--out-dir", default="plots")
    ap.add_argument("--label", default="chronos2", help="Label of the run (chronos2/naive)")
    ap.add_argument("--scenarios", nargs="+", default=["residual", "trend", "none"])
    ap.add_argument("--series", nargs="+", type=int, default=None,
                    help="Specific series IDs to plot (default: first 6 per scenario)")
    ap.add_argument("--grid", action="store_true", default=True,
                    help="Additionally overview grid per scenario")
    ap.add_argument("--no-single", action="store_true",
                    help="No single PNGs, only grid + metrics")
    args = ap.parse_args()

    input_dir, out_dir = Path(args.input_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(input_dir, args.label)
    metrics_lookup = {}
    if metrics is not None:
        for _, r in metrics.iterrows():
            metrics_lookup[(r["scenario"], int(r["series_id"]))] = r

    written = []
    for scen in args.scenarios:
        fc, series, events = load_inputs(input_dir, Path(args.data_dir), args.label, scen)
        ids = args.series or sorted(fc["series_id"].unique())[:6]
        print(f"=== {scen}: plotting {len(ids)} series ===")

        grid_rows = []
        for sid in ids:
            fc_s = fc[fc["series_id"] == sid].sort_values("t")
            if fc_s.empty:
                print(f"  [Skip] Series {sid} not in the forecast file")
                continue
            if series is None:
                print("  [Skip] Original series missing -> no figure possible")
                break
            s_all = series[series["series_id"] == sid].sort_values("t")
            t_min = int(fc_s["t"].min())
            ctx = s_all[s_all["t"] < t_min]
            truth = s_all[s_all["t"] >= t_min]
            ev = events.get(int(sid), [])

            m = metrics_lookup.get((scen, int(sid)))
            if m is not None:
                title = (f"Series {sid} | {scen} | MAE {m['mae']:.2f} | "
                         f"WQL {m['wql']:.3f} | Cov {m['coverage']:.2f}")
                nominal = float(m["nominal"]) if "nominal" in m and pd.notna(m["nominal"]) else None
            else:
                title, nominal = f"Series {sid} | {scen}", None

            if not args.no_single:
                p = out_dir / f"forecast_{args.label}_{scen}_s{sid}.png"
                plot_one(ctx, truth, fc_s, ev, title, p, nominal)
                written.append(p)
            grid_rows.append((sid, {"ctx": ctx, "truth": truth,
                                    "fc_s": fc_s, "events": ev}))

        if args.grid and grid_rows:
            p = out_dir / f"overview_{args.label}_{scen}.png"
            plot_grid(grid_rows[:6], scen, args.label, p)
            written.append(p)

    if metrics is not None and len(metrics):
        p = out_dir / f"metrics_{args.label}.png"
        plot_metrics_box(metrics, args.label, str(p))
        written.append(p)
    else:
        print("[Note] metrics_per_series.csv (for this label) not found "
              "-> metric boxplots skipped.")

    print(f"\n{len(written)} PNGs -> {out_dir}/")
    for p in written:
        print("  ", p.name)


if __name__ == "__main__":
    main()

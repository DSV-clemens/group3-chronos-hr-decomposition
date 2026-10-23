"""
Where does the model LOOK before deciding? Event-window attribution analysis.

Uses the WindowSHAP attributions we already computed (Phase 07) and the
ground-truth event list (metadata_{scenario}.jsonl) to answer:
for context windows that contain a real anomaly event vs. clean windows,
how strongly does each pipeline rely on them?

Outputs:
    plots/windowshap_event_attribution.png   (the money plot)
    results/event_attribution.csv            (per-pipeline stats)
    results/event_attribution.json           (numbers for the report)

Per window w of series i:  hit = 1 if any event overlaps [w*64, (w+1)*64).
Event windows carry spikes/level-shifts that the DGP placed inside the
context; clean windows carry only baseline dynamics.

Invocation:
    python src/05_analysis/plot_event_attribution.py
    python src/05_analysis/plot_event_attribution.py --scenarios residual
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WINDOW = 64  # minutes per context window (8 x 64 = 512 context)


def event_window_mask(events: list[dict], n_windows: int, window: int) -> set[int]:
    hits: set[int] = set()
    for e in events:
        if e.get("start") is None:
            continue
        end = e.get("end", e["start"])
        for w in range(n_windows):
            ws, we = w * window, (w + 1) * window
            if e["start"] < we and end > ws:
                hits.add(w)
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Event-window attribution: where do the pipelines look before deciding?")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--data-dir", default="data/synthetic")
    ap.add_argument("--scenarios", nargs="+", default=["residual", "trend"])
    ap.add_argument("--pipelines", nargs="+", default=["raw", "decomp"])
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--out-dir", default="plots")
    ap.add_argument("--fig-name", default="windowshap_event_attribution.png")
    ap.add_argument("--csv-name", default="event_attribution.csv")
    args = ap.parse_args()

    results_dir, data_dir, out_dir = (Path(args.results_dir), Path(args.data_dir),
                                      Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    n_windows = args.context // WINDOW

    print("=" * 68)
    print("  Event-window attribution (WindowSHAP + ground-truth events)")
    print("=" * 68)

    rows, per_position = [], []
    for scen in args.scenarios:
        shap_path = results_dir / f"windowshap_{scen}.parquet"
        meta_path = data_dir / f"metadata_{scen}.jsonl"
        if not shap_path.exists() or not meta_path.exists():
            print(f"[skip] {scen}: missing {shap_path.name} or {meta_path.name}")
            continue
        shap = pd.read_parquet(shap_path)
        meta = {m["series_id"]: m for m in
                (json.loads(l) for l in meta_path.read_text().splitlines() if l.strip())}
        shap["has_event"] = shap.apply(
            lambda r: r["window"] in event_window_mask(
                meta.get(r["series_id"], {}).get("events", []), n_windows, WINDOW),
            axis=1)
        shap["scenario"] = scen

        for pipe in args.pipelines:
            p = shap[shap.pipeline == pipe]
            ev, cl = p[p.has_event], p[~p.has_event]
            for val in ("phi_median", "phi_band"):
                strength = val == "phi_median"
                stats = {
                    "scenario": scen, "pipeline": pipe, "value_function": val,
                    "n_event_windows": len(ev), "n_clean_windows": len(cl),
                    "mean_abs_event": float(ev[val].abs().mean()),
                    "mean_abs_clean": float(cl[val].abs().mean()),
                    "median_abs_event": float(ev[val].abs().median()),
                    "median_abs_clean": float(cl[val].abs().median()),
                }
                stats["contrast_mean"] = (stats["mean_abs_event"] /
                                          stats["mean_abs_clean"]
                                          if stats["mean_abs_clean"] else np.nan)
                rows.append(stats)
                print(f"[{scen} {pipe} {val}] event |phi| {stats['mean_abs_event']:.2f} "
                      f"vs clean {stats['mean_abs_clean']:.2f} "
                      f"(x{stats['contrast_mean']:.2f})")
            # positional profile: mean |phi| by window index, event vs clean
            prof = (p.assign(abs_phi=p["phi_median"].abs())
                      .groupby(["window", "has_event"])["abs_phi"].mean().unstack())
            per_position.append((scen, pipe, prof))

        # keep enriched frame for CSV
        shap.to_csv(results_dir / f"windowshap_enriched_{scen}.csv", index=False)

    if not rows:
        raise SystemExit("no data found")
    tbl = pd.DataFrame(rows)
    csv_path = results_dir / args.csv_name
    tbl.to_csv(csv_path, index=False)
    print(f"[write] {csv_path}")
    json_path = results_dir.parent / "results" / "event_attribution.json"
    json_path.write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window_minutes": WINDOW, "rows": rows,
    }, indent=2))
    print(f"[write] {json_path}")

    # ---- the money plot: one row per scenario, cols = pipelines ---------
    scens = tbl.scenario.unique().tolist()
    fig, axes = plt.subplots(len(scens), len(args.pipelines),
                             figsize=(6.5 * len(args.pipelines), 4.2 * len(scens)),
                             squeeze=False, sharey=True)
    colors = {"event": "#e07a5f", "clean": "#81b29a"}
    for i, scen in enumerate(scens):
        for j, pipe in enumerate(args.pipelines):
            ax = axes[i][j]
            prof = next(p for s, pl, p in per_position if s == scen and pl == pipe)
            x = np.arange(len(prof))
            if True in prof.columns:
                ax.bar(x, prof[True], color=colors["event"], label="window contains event")
            if False in prof.columns:
                ax.bar(x, prof[False], color=colors["clean"], label="clean window")
            ax.set_title(f"{scen} — {pipe} pipeline", fontsize=11)
            ax.set_xlabel("context window (0 = oldest, 7 = newest 64 min)")
            if j == 0:
                ax.set_ylabel(r"mean $|\phi_{\mathrm{median}}|$ (bpm)")
            ax.legend(fontsize=8)
            ax.set_xticks(x)
    fig.suptitle("Where the model looks before deciding: Shapley attribution of context windows,\n"
                 "windows containing ground-truth anomalies vs. clean windows",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig_path = out_dir / args.fig_name
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[write] {fig_path}")


if __name__ == "__main__":
    main()

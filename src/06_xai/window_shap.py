"""
Phase 7: WindowSHAP-style explainability for both forecast pipelines.

Model-agnostic WindowSHAP for Chronos-2 (the foundation model is treated as
a black box; no gradients are available for the sampled quantile forecast):

1. The CONTEXT window (t < 512) is cut into W windows of equal length.
2. For every subset S of windows (all 2^W subsets), the forecast median and
   the outer band (q90 - q10) are computed with all windows in S replaced by
   a BACKGROUND value (constant, per series).
3. Shapley values phi_i for window i over W players are estimated from the
   subset outputs:
        phi_i = sum over subsets S not containing i
                [ w(|S|) * ( v(S+{i}) - v(S) ) ],  w(s) = W / (W-1) * 1/(C(W-1,s))
   This is the exact Shapley formula evaluated over all 2^W subsets
   (tractable for W <= ~12; default W = 8 -> 256 model calls per series).
4. Game outputs (value functions):
   - median:   mean absolute deviation of the forecast median from the
               background-only forecast median (point-shift attribution)
   - band:     q90-q10 interval width (uncertainty attribution)

Run for BOTH pipelines on identical windows & background:
   pipeline A 'raw':        hr context -> Chronos-2
   pipeline B 'decomp':    STL context -> per-component Chronos-2 ->
                           quantile-wise recomposition (same code as
                           run_forecast_decomp.py)
-> paired explainability comparison: which context windows drive each
   pipeline's forecast, and does the decomposition pipeline attribute the
   anomaly windows differently?

Outputs in --out-dir:
    windowshap_{pipeline}_{scenario}.parquet  (long: series_id, window,
        phi_median, phi_band, plus subset diagnostics)
    windowshap_summary_{scenario}.csv          (mean |phi| per pipeline x window,
                                                aggregated over series)
    manifest_windowshap.json

Usage:
    python src/06_xai/window_shap.py --max-series 2            # quick test
    python src/06_xai/window_shap.py --windows 8              # default
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "04_forecast"))

from run_forecast import (TS_ORIGIN, load_scenario, split_context,
                          forecast_chronos)
from run_forecast_decomp import component_frames

import math


def _comb(n: int, k: int) -> int:
    return math.comb(n, k)


PIPELINES = ("raw", "decomp")

WINDOWSHAP_VERSION = "windowshap-1.2 (batched + parallel STL)"


# ------------------------------------------------------------------ game

def subsets(n: int) -> list[tuple[int, ...]]:
    """All subsets of {0..n-1} as tuples, ordered by size."""
    out = []
    for k in range(n + 1):
        out.extend(itertools.combinations(range(n), k))
    return out


def shapley_from_subsets(values: dict[tuple[int, ...], float],
                          n: int) -> np.ndarray:
    """Exact Shapley values from all 2^n subset values.

    phi_i = sum_{S not contains i} w(|S|) * (v(S+{i}) - v(S)),
    w(s) = n! / ((n-1)! * s! * (n-1-s)!)  (standard weight, normalized to
    sum over subsets -> each coalition counted once).
    """
    phi = np.zeros(n)
    for i in range(n):
        for S in subsets(n):
            if i in S:
                continue
            s = len(S)
            # classic Shapley weight: s!(n-1-s)!/n!
            w = float(math.factorial(s) * math.factorial(n - 1 - s)
                      / math.factorial(n))
            key_with = tuple(sorted(S + (i,)))
            phi[i] += w * (values[key_with] - values[S])
    return phi


# ------------------------------------------------------------------ value fns

def forecast_median_band(fc: pd.DataFrame, quantiles: list[float]) -> tuple[float, float]:
    """Mean median and mean band width of a forecast frame (all series pooled,
    but frames here always cover one series at a time)."""
    med = fc["q0.5"].to_numpy()
    lo = fc[f"q{quantiles[0]:g}"].to_numpy()
    hi = fc[f"q{quantiles[-1]:g}"].to_numpy()
    return float(np.mean(med)), float(np.mean(hi - lo))


def pipeline_forecast(ctx: pd.DataFrame, horizon: int, quantiles: list[float],
                     model: str, pipeline: str,
                     device: str | None) -> pd.DataFrame:
    """Runs the respective pipeline on the (possibly masked) context.
    Returns a frame with columns series_id, t, q{q}."""
    if pipeline == "raw":
        return forecast_chronos(ctx, horizon, quantiles, model, device)
    # decomp pipeline on masked context (identical recomposition code path
    # as run_forecast_decomp.main, but reusable here)
    comps, _, _ = component_frames(ctx)
    fc_comps = {}
    for name, cdf in comps.items():
        fc = forecast_chronos(cdf, horizon, quantiles, model, device)
        fc = fc.sort_values("series_id").reset_index(drop=True)
        fc_comps[name] = fc
    tr = fc_comps["trend"]
    base = tr[["series_id"]].copy()
    base["t"] = tr.groupby("series_id").cumcount()
    for q in quantiles:
        base[f"q{q:g}"] = sum(fc_comps[n][f"q{q:g}"].to_numpy()
                              for n in ("trend", "seasonal", "resid"))
    return base


def mask_context(ctx: pd.DataFrame, window_idx: set[int], n_windows: int,
                 background: float) -> pd.DataFrame:
    """Replaces the windows in window_idx with the background value."""
    ctx = ctx.copy()
    wlen = int(np.ceil(len(ctx) / n_windows)) if len(ctx) else 0
    mask = np.zeros(len(ctx), dtype=bool)
    for i in window_idx:
        mask[i * wlen:(i + 1) * wlen] = True
    ctx.loc[mask, "hr"] = background
    return ctx


def background_value(ctx: pd.DataFrame) -> float:
    """Per-series background: median of the context (a neutral 'empty' input)."""
    return float(ctx["hr"].median())


def series_context(df: pd.DataFrame, sid: int, context: int) -> pd.DataFrame:
    g = df[(df["series_id"] == sid) & (df["t"] < context)]
    return g[["series_id", "t", "hr"]].sort_values("t").reset_index(drop=True)


def series_meta(meta: list[dict], sid: int) -> list[dict]:
    return [m for m in meta if m.get("series_id") == sid]


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(
        description="WindowSHAP-style explainability for Chronos-2 pipelines")
    ap.add_argument("--input-dir", default="data/synthetic")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--scenarios", nargs="+", default=["residual", "trend"])
    ap.add_argument("--pipelines", nargs="+", default=list(PIPELINES),
                    choices=list(PIPELINES))
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=256)
    ap.add_argument("--quantiles", nargs="+", type=float, default=[0.1, 0.5, 0.9])
    ap.add_argument("--model", default="amazon/chronos-2")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--windows", type=int, default=8,
                    help="Context is cut into W windows (2^W model calls per "
                         "series & pipeline; 8 -> 256 calls)")
    ap.add_argument("--max-series", type=int, default=None)
    args = ap.parse_args()

    input_dir, out_dir = Path(args.input_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    W = args.windows

    manifest = {
        "method": "WindowSHAP (model-agnostic, exact Shapley over window subsets)",
        "windowshap_version": WINDOWSHAP_VERSION,
        "pipelines": args.pipelines,
        "n_windows": W,
        "n_model_calls_per_series": 2 ** W,
        "value_functions": ["median (mean forecast median)",
                            "band (mean q90-q10 width)"],
        "background": "per-series context median",
        "model": args.model,
        "quantiles": sorted(args.quantiles),
        "context": args.context, "horizon": args.horizon,
        "windows": W,
        "scenarios": args.scenarios,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    print("=" * 68)
    print(f"  WindowSHAP for Chronos-2  |  version: {WINDOWSHAP_VERSION}")
    print(f"  Hello from your friendly explainability pipeline! :)        ")
    print("=" * 68)
    print(f"[config] model: {args.model} | device: {args.device or 'auto'}")
    print(f"[config] context {args.context} -> {args.windows} windows of "
          f"{int(np.ceil(args.context / args.windows))} min each")
    print(f"[config] value functions: forecast median + band width "
          f"(quantiles {sorted(args.quantiles)})")
    print(f"[config] pipelines: {args.pipelines} | scenarios: {args.scenarios}")
    print(f"[config] model calls per pipeline & scenario: {2 ** args.windows} "
          f"(all series batched together per subset)")
    print()

    t_start = time.perf_counter()
    for scen in args.scenarios:
        df = load_scenario(input_dir, scen)
        ctx_all, _ = split_context(df, args.context, args.max_series)
        ids = sorted(ctx_all["series_id"].unique())
        print(f"=== scenario '{scen}': {len(ids)} series x "
              f"{len(args.pipelines)} pipelines x {2**W} subsets ===")

        rows_all = []
        # Batching: for every subset mask, ALL series are forecast in ONE
        # model call (predict_df is multi-id capable) -> 2^W calls per
        # pipeline & scenario INSTEAD OF 2^W per series.
        ctx_by_id, bg_by_id = {}, {}
        for sid in ids:
            ctx = series_context(df, int(sid), args.context)
            if len(ctx) < args.context:
                continue
            ctx_by_id[int(sid)] = ctx
            bg_by_id[int(sid)] = background_value(ctx)
        usable_ids = sorted(ctx_by_id)
        wlen = int(np.ceil(args.context / W))
        print(f"[{scen}] {len(usable_ids)} usable series, window length "
              f"{wlen} min", flush=True)

        for pipeline in args.pipelines:
            vals = {sid: {} for sid in usable_ids}
            for S in subsets(W):
                masked_all = [mask_context(ctx_by_id[sid], set(S), W, bg_by_id[sid])
                              for sid in usable_ids]
                masked = pd.concat(masked_all, ignore_index=True)
                fc = pipeline_forecast(masked, args.horizon,
                                       sorted(args.quantiles), args.model,
                                       pipeline, args.device)
                for sid, g in fc.groupby("series_id"):
                    med, band = forecast_median_band(g, sorted(args.quantiles))
                    vals[int(sid)][S] = (med, band)
                n_done = len(next(iter(vals.values())))
                print(f"[{scen}/{pipeline}] subset {n_done}/{2**W} done "
                      f"({time.perf_counter() - t_start:.0f}s)", flush=True)
            for sid in usable_ids:
                v_med = {S: v[0] for S, v in vals[sid].items()}
                v_band = {S: v[1] for S, v in vals[sid].items()}
                phi_med = shapley_from_subsets(v_med, W)
                phi_band = shapley_from_subsets(v_band, W)
                for i in range(W):
                    rows_all.append({
                        "pipeline": pipeline, "scenario": scen,
                        "series_id": int(sid), "window": i,
                        "window_start": i * wlen, "window_end": (i + 1) * wlen,
                        "phi_median": float(phi_med[i]),
                        "phi_band": float(phi_band[i]),
                    })

        result = pd.DataFrame(rows_all)
        path = out_dir / f"windowshap_{scen}.parquet"
        result.to_parquet(path)
        print(f"[{scen}] saved {len(result)} phi rows -> {path.name}")
        summary = (result.groupby(["pipeline", "window"])
                   [["phi_median", "phi_band"]]
                   .agg(["mean", "std", lambda x: float(np.mean(np.abs(x)))])
                   .reset_index())
        summary.columns = ["pipeline", "window",
                           "phi_median_mean", "phi_median_std", "phi_median_mae",
                           "phi_band_mean", "phi_band_std", "phi_band_mae"]
        summary.to_csv(out_dir / f"windowshap_summary_{scen}.csv", index=False)
        print(f"[Saving] {path.name} ({len(result)} rows)")

    (out_dir / "manifest_windowshap.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n[done] manifest -> {out_dir / 'manifest_windowshap.json'}")
    print(f"[done] total runtime: {time.perf_counter() - t_start:.0f}s - "
          f"bye, and good luck with the analysis!")


if __name__ == "__main__":
    main()

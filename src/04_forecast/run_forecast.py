"""
Phase 4: Chronos-2 batch forecast on all DGP series (A/B basis).

For each series: context = first 512 points (only 'hr'! no ground-truth
columns), forecast of the remaining 256 points with quantiles. Afterwards
metrics per series and per scenario: MAE (median), WQL (weighted quantile
loss), empirical coverage of the outer quantile band.

The run is labelled 'chronos2' or 'naive'; the naive persistence baseline
serves (a) to test the pipeline without a model and (b) as a dumb reference
for the A/B comparison.

Invocation (from the repository root):
    python src/04_forecast/run_forecast.py                     # Chronos-2, all scenarios
    python src/04_forecast/run_forecast.py --max-series 10    # quick test
    python src/04_forecast/run_forecast.py --naive             # persistence baseline (no model)
    python src/04_forecast/run_forecast.py --device cuda       # force GPU

Outputs in --out-dir:
    forecasts_{label}_{scenario}.parquet   (long: series_id, t, actual, q columns)
    metrics_per_series.csv                 (per series: mae, wql, coverage, label;
                                            A/B-safe: updated per label,
                                            runs with other labels are kept)
    summary_scenario.csv                  (median per scenario, all labels)
    manifest_{label}.json                  (model, device, quantiles, DGP hash, runtime)

Provenance: if data/synthetic/dgp_config_mmash.json exists, the DGP hash is
carried over into manifest_{label}.json -> metrics are traceable to the
frozen DGP.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

TS_ORIGIN = pd.Timestamp("2026-01-01 00:00:00")  # DGP minute grid -> timestamps
DEFAULT_QUANTILES = [0.1, 0.5, 0.9]

_PIPELINE = None  # Chronos pipeline loaded only once


# ------------------------------------------------------------------ loading

def load_scenario(input_dir: Path, scenario: str) -> pd.DataFrame:
    """Loads a scenario file; new + old DGP names, parquet before csv.gz."""
    names = [f"synthetic_hr_mmash_{scenario}", f"synthetic_hr_{scenario}"]
    for name in names:
        for ext, reader in ((".parquet", "read_parquet"), (".csv.gz", "read_csv")):
            p = input_dir / (name + ext)
            if p.exists():
                print(f"[Loading] {p.name}")
                return getattr(pd, reader)(p)
    raise FileNotFoundError(
        f"No file for '{scenario}' in {input_dir} "
        f"(searched: {', '.join(n + e for n in names for e in ('.parquet', '.csv.gz'))})")


def split_context(df: pd.DataFrame, context: int, max_series: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separates context (model input) from ground truth. The pipeline sees ONLY hr."""
    df = df[["series_id", "t", "hr"]].copy()
    ctx = df[df["t"] < context].copy()
    truth = df[df["t"] >= context].copy()
    if max_series:
        ids = sorted(ctx["series_id"].unique())[:max_series]
        ctx = ctx[ctx["series_id"].isin(ids)]
        truth = truth[truth["series_id"].isin(ids)]
    return ctx, truth


# ------------------------------------------------------------------ forecast

def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam approximation, accurate to ~1e-9)."""
    a = [-3.969683028134653e+01, 2.209470982863222e+02, -2.759285104302205e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288362e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539346629e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = (-2.0 * np.log(p)) ** 0.5
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p > phigh:
        return -_norm_ppf(1.0 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)


def forecast_chronos(ctx: pd.DataFrame, horizon: int, quantiles: list[float],
                     model: str, device: str | None) -> pd.DataFrame:
    """Chronos-2 on ALL series at once (predict_df is multi-id capable)."""
    global _PIPELINE
    if _PIPELINE is None:
        from chronos import Chronos2Pipeline  # pip install "chronos-forecasting>=2.0"
        import torch
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Chronos-2] loading model '{model}' on {dev} ...")
        t0 = time.perf_counter()
        _PIPELINE = Chronos2Pipeline.from_pretrained(model, device_map=dev)
        print(f"[Chronos-2] model loaded in {time.perf_counter() - t0:.1f}s")

    c = ctx.rename(columns={"series_id": "id", "hr": "target"})
    c = c.assign(timestamp=TS_ORIGIN + pd.to_timedelta(c["t"], unit="m"))
    pred = _PIPELINE.predict_df(
        c[["id", "timestamp", "target"]],
        prediction_length=horizon,          # max. 1024 for Chronos-2
        quantile_levels=quantiles,
        id_column="id", timestamp_column="timestamp", target="target",
    )

    # Robustly find the quantile columns ('0.5', '0.50', 0.5, ...)
    qcols: dict[float, str] = {}
    for q in quantiles:
        for cand in (f"{q:g}", f"{q:.2f}", f"{q:.3f}", str(q), float(q), q):
            if cand in pred.columns:
                qcols[q] = cand
                break
        else:
            raise KeyError(f"Quantile column for {q} not found: {list(pred.columns)}")

    pred = pred.sort_values(["id", "timestamp"]).reset_index(drop=True)
    out = pd.DataFrame({"series_id": pred["id"].to_numpy()})
    for q in quantiles:
        out[f"q{q:g}"] = pred[qcols[q]].to_numpy()
    return out


def forecast_naive(ctx: pd.DataFrame, horizon: int, quantiles: list[float]) -> pd.DataFrame:
    """Persistence baseline: median = last context value; band grows with
    sqrt(h) from the spread of the context minute diffs. Without a model ->
    test + reference."""
    z = {q: _norm_ppf(q) for q in quantiles}
    rows = []
    for sid, g in ctx.groupby("series_id"):
        y = g.sort_values("t")["hr"].to_numpy()
        last = float(y[-1])
        sigma1 = float(np.std(np.diff(y))) or 1e-6
        for h in range(horizon):
            scale = sigma1 * np.sqrt(h + 1)
            row = {"series_id": sid}
            for q in quantiles:
                row[f"q{q:g}"] = last + z[q] * scale
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ metrics

def series_metrics(y: np.ndarray, qmat: np.ndarray, quantiles: list[float]) -> dict:
    """MAE (median), WQL, coverage of the outer band for ONE series."""
    if 0.5 not in quantiles:
        raise ValueError("--quantiles must contain 0.5 (median basis).")
    med = qmat[:, quantiles.index(0.5)]
    mae = float(np.mean(np.abs(y - med)))

    denom = max(float(np.abs(y).sum()), 1e-9)
    tot = 0.0
    for i, q in enumerate(quantiles):
        e = y - qmat[:, i]
        tot += float((np.maximum(e, 0.0) * q + np.maximum(-e, 0.0) * (1.0 - q)).sum())
    wql = 2.0 * tot / (len(quantiles) * denom)

    cover = float(np.mean((y >= qmat[:, 0]) & (y <= qmat[:, -1])))
    return {"mae": mae, "wql": wql, "coverage": cover,
            "nominal": quantiles[-1] - quantiles[0]}


# ------------------------------------------------------------------ saving

def save_df(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path.with_suffix(".parquet"))
        print(f"[Saving] {path.with_suffix('.parquet').name}")
    except ImportError:
        df.to_csv(path.with_suffix(".csv.gz"), index=False)
        print(f"[Saving] {path.with_suffix('.csv.gz').name} (CSV fallback, pyarrow missing)")


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description="Chronos-2 batch forecast on DGP data")
    ap.add_argument("--input-dir", default="data/synthetic")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--scenarios", nargs="+", default=["residual", "trend", "none"])
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=256)
    ap.add_argument("--quantiles", nargs="+", type=float, default=DEFAULT_QUANTILES)
    ap.add_argument("--model", default="amazon/chronos-2")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"],
                    help="Default: auto (cuda if available)")
    ap.add_argument("--max-series", type=int, default=None,
                    help="Only the first N series per scenario (quick test)")
    ap.add_argument("--naive", action="store_true",
                    help="Persistence baseline instead of Chronos-2 (no model needed)")
    args = ap.parse_args()

    input_dir, out_dir = Path(args.input_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = "naive" if args.naive else "chronos2"
    quantiles = sorted(args.quantiles)

    # Adopt DGP provenance if available
    dgp = {}
    dgp_cfg = input_dir / "dgp_config_mmash.json"
    if dgp_cfg.exists():
        cfg = json.loads(dgp_cfg.read_text(encoding="utf-8"))
        dgp = {"dgp_hash": cfg.get("dgp_hash"), "params_sha256": cfg.get("params_sha256"),
               "generator_version": cfg.get("generator_version")}
        print(f"[DGP] config hash={dgp.get('dgp_hash')} adopted.")
    else:
        print("[DGP] dgp_config_mmash.json not found -> no hash in the manifest.")

    t_start = time.perf_counter()
    metric_rows, summaries, n_series_map = [], {}, {}

    for scen in args.scenarios:
        df = load_scenario(input_dir, scen)
        ctx, truth = split_context(df, args.context, args.max_series)
        n = int(ctx["series_id"].nunique())
        avail = len(truth) // max(n, 1)
        horizon = min(args.horizon, avail)
        print(f"\n=== Scenario '{scen}': {n} series | context {args.context} | horizon {horizon} ===")

        t0 = time.perf_counter()
        if args.naive:
            fc = forecast_naive(ctx, horizon, quantiles)
        else:
            fc = forecast_chronos(ctx, horizon, quantiles, args.model, args.device)
        dt = time.perf_counter() - t0

        # t column: per series position 0..horizon-1 -> absolute minute
        fc["t"] = args.context + fc.groupby("series_id").cumcount()
        fc = fc.merge(truth[["series_id", "t", "hr"]], on=["series_id", "t"], how="inner")
        fc = fc.rename(columns={"hr": "actual"}).assign(scenario=scen)
        save_df(fc, out_dir / f"forecasts_{label}_{scen}")

        for sid, g in fc.groupby("series_id"):
            m = series_metrics(g["actual"].to_numpy(),
                               g[[f"q{q:g}" for q in quantiles]].to_numpy(),
                               quantiles)
            metric_rows.append({"label": label, "scenario": scen, "series_id": int(sid), **m})

        n_series_map[scen] = n
        print(f"[Run] {label} on '{scen}' finished in {dt:.1f}s")

    # A/B-safe: metrics_per_series.csv is updated per label; runs with OTHER
    # labels are kept. (Previously the file was overwritten completely ->
    # the Chronos metrics were lost on the naive run.)
    metrics = pd.DataFrame(metric_rows)
    metrics_path = out_dir / "metrics_per_series.csv"
    if metrics_path.exists():
        old = pd.read_csv(metrics_path)
        metrics = pd.concat([old[old["label"] != label], metrics], ignore_index=True)
    metrics.to_csv(metrics_path, index=False)

    summary = (metrics.groupby(["label", "scenario"])[["mae", "wql", "coverage"]]
               .median().reset_index())
    summary["n_series"] = metrics.groupby(["label", "scenario"]).size().to_numpy()
    summary["nominal"] = quantiles[-1] - quantiles[0]
    summary.to_csv(out_dir / "summary_scenario.csv", index=False)

    manifest = {
        "label": label,
        "model": None if args.naive else args.model,
        "device": args.device or "auto (cuda if available)",
        "quantiles": quantiles,
        "context": args.context, "horizon": args.horizon,
        "max_series": args.max_series,
        "n_series": n_series_map,
        "scenarios": args.scenarios,
        "dgp": dgp,
        "runtime_sec": round(time.perf_counter() - t_start, 1),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path = out_dir / f"manifest_{label}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print("\n=== Summary (median per scenario) ===")
    print(summary.to_string(index=False))
    print(f"\n[Manifest] {manifest_path}")


if __name__ == "__main__":
    main()

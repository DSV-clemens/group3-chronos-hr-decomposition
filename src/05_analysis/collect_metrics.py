"""
Collect all pipeline metrics into one structured, machine-readable report.

Reads (defaults match the clean-run layout):
    results/manifest_{label}.json            (provenance per run)
    results/summary_scenario.csv             (median aggregates, written per run)
    results/metrics_per_series.csv           (per-series metrics, all labels)
    results/windowshap_summary_{scenario}.csv (WindowSHAP phi aggregates)
    data/synthetic/dgp_config_mmash.json     (DGP provenance: hash, params, seed)

Writes:
    results/report_summary.json              (everything, structured)
    results/report_summary.md                (human-readable tables + checks)

Checks:
    - provenance consistency: every manifest must carry the SAME dgp_hash /
      params_sha256 as data/synthetic/dgp_config_mmash.json (warn otherwise)
    - completeness: labels in manifests vs. labels found in metrics_per_series.csv
    - plausibility: coverage in [0, 1], metrics finite, n_series == expected

Invocation:
    python src/05_analysis/collect_metrics.py
    python src/05_analysis/collect_metrics.py --results-dir results --data-dir data/synthetic
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

METRIC_COLS = ["mae", "wql", "coverage"]
WS_SCENARIOS = ["residual", "trend"]


def load_manifests(results_dir: Path) -> dict[str, dict]:
    manifests = {}
    for p in sorted(results_dir.glob("manifest_*.json")):
        label = p.stem.replace("manifest_", "")
        manifests[label] = json.loads(p.read_text())
    return manifests


def load_dgp_provenance(data_dir: Path) -> dict:
    cfg_path = data_dir / "dgp_config_mmash.json"
    if not cfg_path.exists():
        return {}
    cfg = json.loads(cfg_path.read_text())
    return {
        "dgp_hash": cfg.get("dgp_hash"),
        "params_sha256": cfg.get("params_sha256"),
        "seed": cfg.get("seed"),
        "n_series": cfg.get("n_series"),
    }


def provenance_check(manifests: dict[str, dict], dgp: dict) -> list[str]:
    issues = []
    for label, m in manifests.items():
        d = (m.get("dgp") or {})
        if not d:
            issues.append(
                f"manifest '{label}': no DGP provenance recorded "
                f"(dgp_hash/params_sha256 missing in manifest)")
            continue
        for key in ("dgp_hash", "params_sha256"):
            got = d.get(key)
            want = dgp.get(key)
            if want is not None and got != want:
                issues.append(
                    f"manifest '{label}': {key} {got!r} != data {want!r} "
                    f"(results predate the current synthetic data)")
    return issues

def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics.groupby(["label", "scenario"])[METRIC_COLS]
    out = grouped.agg(["median", "mean", "std"]).reset_index()
    out.columns = ["_".join(c).strip("_") for c in out.columns]
    counts = grouped.size().reset_index(name="n_series")
    return out.merge(counts, on=["label", "scenario"])


def plausibility_checks(metrics: pd.DataFrame, expected_n: int | None) -> list[str]:
    issues = []
    for col in METRIC_COLS:
        if col in metrics and not np.isfinite(metrics[col]).all():
            issues.append(f"non-finite values in column '{col}'")
    if "coverage" in metrics:
        bad = metrics[(metrics["coverage"] < 0) | (metrics["coverage"] > 1)]
        if len(bad):
            issues.append(f"{len(bad)} rows with coverage outside [0, 1]")
    if expected_n is not None and "series_id" in metrics:
        counts = metrics.groupby(["label", "scenario"])["series_id"].nunique()
        wrong = counts[counts != expected_n]
        if len(wrong):
            issues.append(
                "series counts != expected "
                + ", ".join(f"{k}: {v}" for k, v in wrong.items()))
    return issues


def md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_(empty)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:{floatfmt}}" if not np.isnan(v) else "")
            elif isinstance(v, (np.floating,)):
                cells.append(f"{float(v):{floatfmt}}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Collect pipeline metrics + provenance into one structured report")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--data-dir", default="data/synthetic")
    ap.add_argument("--out-json", default=None,
                    help="default: <results-dir>/report_summary.json")
    ap.add_argument("--out-md", default=None,
                    help="default: <results-dir>/report_summary.md")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)
    out_json = Path(args.out_json) if args.out_json else results_dir / "report_summary.json"
    out_md = Path(args.out_md) if args.out_md else results_dir / "report_summary.md"

    print("=" * 68)
    print("  Metric collection report")
    print("=" * 68)

    manifests = load_manifests(results_dir)
    dgp = load_dgp_provenance(data_dir)
    print(f"[load] {len(manifests)} manifests: {sorted(manifests)}")
    print(f"[load] DGP provenance: hash={dgp.get('dgp_hash')} "
          f"seed={dgp.get('seed')} n_series={dgp.get('n_series')}")

    checks: dict[str, list[str]] = {}
    checks["provenance"] = provenance_check(manifests, dgp)

    metrics_path = results_dir / "metrics_per_series.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    print(f"[load] metrics_per_series: {len(metrics)} rows, "
          f"labels={sorted(metrics['label'].unique()) if not metrics.empty else []}")

    manifest_labels = set(manifests) - {"windowshap"}
    metric_labels = set(metrics["label"].unique()) if not metrics.empty else set()
    missing = manifest_labels - metric_labels
    extra = metric_labels - manifest_labels
    if missing:
        checks["completeness"] = [f"labels with manifest but no metrics: {sorted(missing)}"]
    if extra:
        checks.setdefault("completeness", []).append(
            f"labels with metrics but no manifest: {sorted(extra)}")
    checks.setdefault("completeness", [])

    expected_n = dgp.get("n_series")
    checks["plausibility"] = plausibility_checks(metrics, expected_n)

    summary_tbl = aggregate_metrics(metrics) if not metrics.empty else pd.DataFrame()
    manifest_labels = set(manifests)
    summary_tbl = summary_tbl  # already correct

    ws_tables: dict[str, pd.DataFrame] = {}
    for scen in WS_SCENARIOS:
        p = results_dir / f"windowshap_summary_{scen}.csv"
        if p.exists():
            ws_tables[scen] = pd.read_csv(p)
    ws_manifest = manifests.get("windowshap", {})
    print(f"[load] windowshap summaries: {sorted(ws_tables)}")

    for name, issues in checks.items():
        status = "OK" if not issues else "WARN"
        print(f"[check:{name}] {status}")
        for msg in issues:
            print(f"    - {msg}")

    report = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dgp_provenance": dgp,
        "checks": checks,
        "manifests": manifests,
        "metrics_per_label_scenario": (
            summary_tbl.to_dict(orient="records") if not summary_tbl.empty else []),
        "windowshap": {
            "manifest": ws_manifest,
            "summary_tables": {s: t.to_dict(orient="records")
                               for s, t in ws_tables.items()},
        },
    }
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"[write] {out_json}")

    md = ["# Pipeline metric report", "",
          f"_created: {report['created']}_", "",
          "## Provenance", ""]
    md += [f"- **dgp_hash**: `{dgp.get('dgp_hash')}`",
           f"- **params_sha256**: `{str(dgp.get('params_sha256'))[:16]}...`",
           f"- **seed**: {dgp.get('seed')} | **n_series**: {dgp.get('n_series')}", ""]
    md += ["## Checks", ""]
    for name, issues in checks.items():
        md.append(f"- **{name}**: {'OK' if not issues else 'WARN'}")
        for msg in issues:
            md.append(f"  - {msg}")
    md += ["", "## Forecast metrics (per label x scenario)", ""]
    if not summary_tbl.empty:
        cols = ["label", "scenario", "n_series",
                "mae_median", "mae_mean", "mae_std",
                "wql_median", "wql_mean", "wql_std",
                "coverage_median", "coverage_mean", "coverage_std"]
        md += [md_table(summary_tbl[cols]), ""]
    md += ["## WindowSHAP (phi per window, mean over series)", ""]
    for scen, t in ws_tables.items():
        md += [f"### scenario: {scen}", "",
               md_table(t[["pipeline", "window", "phi_median_mean", "phi_band_mean"]]),
               ""]
    out_md.write_text("\n".join(md) + "\n")
    print(f"[write] {out_md}")


if __name__ == "__main__":
    main()

"""
Interactive viewer for WindowSHAP results (Phase 7 output).

Shows, per scenario and series:
- bar chart of phi_median per context window, raw vs. decomp pipeline side
  by side (which context windows drive the forecast median?)
- bar chart of phi_band (which windows drive the forecast uncertainty?)
- the series' hr curve with the window boundaries + anomaly events, so phi
  values can be read against the data
- summary: mean |phi| per pipeline x window over all series

Usage:
    python src/05_analysis/view_windowshap.py --forecast-dir results \
        --data-dir data/synthetic --out windowshap_viewer.html

Reads:  results/windowshap_{scenario}.parquet
        results/windowshap_summary_{scenario}.csv
        data/synthetic/synthetic_hr_mmash_{scenario}.* + metadata
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCENARIOS = ("residual", "trend", "none")


def load_shap(forecast_dir: Path, scenario: str) -> pd.DataFrame | None:
    p = forecast_dir / f"windowshap_{scenario}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    p = forecast_dir / f"windowshap_{scenario}.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


def load_summary(forecast_dir: Path, scenario: str) -> pd.DataFrame | None:
    p = forecast_dir / f"windowshap_summary_{scenario}.csv"
    return pd.read_csv(p) if p.exists() else None


def load_series(data_dir: Path, scenario: str, sid: int) -> tuple[dict | None, list[dict]]:
    base = data_dir / f"synthetic_hr_mmash_{scenario}"
    ctx = None
    for ext, reader in ((".parquet", "read_parquet"), (".csv.gz", "read_csv")):
        p = Path(str(base) + ext)
        if p.exists():
            df = getattr(pd, reader)(p)
            g = df[df["series_id"] == sid].sort_values("t")
            ctx = {"t": g["t"].tolist(), "hr": [round(v, 2) for v in g["hr"]]}
            break
    events = []
    meta_path = data_dir / f"metadata_{scenario}.jsonl"
    if meta_path.exists():
        for line in meta_path.read_text().splitlines():
            if line.strip():
                m = json.loads(line)
                if m.get("series_id") == sid:
                    events = m.get("events", [])
                    break
    return ctx, events


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>WindowSHAP Viewer</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 16px; color: #1a1a2e; }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 14px; align-items: center;
               margin-bottom: 8px; }}
  label {{ font-size: 13px; display: flex; align-items: center; gap: 4px; }}
  select {{ padding: 3px 6px; }}
  #plot {{ width: 100%; }}
  .card {{ background: #f7f8fa; border: 1px solid #e3e5ea; border-radius: 8px;
           padding: 8px 12px; font-size: 12.5px; margin: 8px 0; }}
  table {{ border-collapse: collapse; font-size: 12.5px; }}
  td, th {{ padding: 2px 10px 2px 0; text-align: left; }}
</style>
</head>
<body>
<h3>WindowSHAP: which context windows drive the forecast?</h3>
<div class="controls">
  <label>Scenario <select id="scenario" onchange="fillSeries()"></select></label>
  <label>Series <select id="series" onchange="render()"></select></label>
</div>
<div id="plot"></div>
<div class="card" id="summary"></div>

<script>
window.DATA = {data};

const $ = (id) => document.getElementById(id);
const RAW = "#1f77b4", DEC = "#d62728";
const SEV_COLOR = {{subtle: "rgba(255,193,7,0.15)", moderate: "rgba(255,152,0,0.2)",
                    severe: "rgba(244,67,54,0.22)"}};

function init() {{
  const sc = $("scenario");
  for (const s of DATA.scenarios) {{
    const o = document.createElement("option");
    o.value = s; o.textContent = s; sc.appendChild(o);
  }}
  fillSeries();
  $("summary").innerHTML = summaryTable();
}}

function fillSeries() {{
  const s = $("scenario").value;
  const sel = $("series");
  sel.innerHTML = "";
  for (const sid of DATA.series_ids[s]) {{
    const o = document.createElement("option");
    o.value = sid; o.textContent = "Series " + sid; sel.appendChild(o);
  }}
  render();
}}

function current() {{
  const s = $("scenario").value;
  const sid = parseInt($("series").value);
  return DATA.shap[s].filter((r) => r.series_id === sid);
}}

function traces() {{
  const rows = current();
  const tr = [];
  const wins = [...new Set(rows.map((r) => r.window))].sort((a, b) => a - b);
  const starts = wins.map((w) => rows.find((r) => r.window === w).window_start);
  const labels = wins.map((w, i) => "W" + w + " (" + starts[i] + ")");
  for (const [pipe, color, ax] of [["raw", RAW, "y"], ["decomp", DEC, "y"]]) {{
    tr.push({{x: labels, y: wins.map((w) => {{
                const r = rows.find((x) => x.pipeline === pipe && x.window === w);
                return r ? r.phi_median.toFixed(2) : null; }}),
              name: "phi_median " + pipe, type: "bar", yaxis: ax,
              marker: {{color: color}}}});
  }}
  return tr;
}}

function layout() {{
  const rows = current();
  const scen = $("scenario").value, sid = parseInt($("series").value);
  return {{
    margin: {{t: 40, r: 20, b: 40, l: 60}}, height: 420,
    barmode: "group",
    xaxis: {{title: "Context window (minute)"}},
    yaxis: {{title: "phi (median attribution, bpm)"}},
    legend: {{orientation: "h", y: 1.12}},
    hovermode: "closest"
  }};
}}

function render() {{
  Plotly.react("plot", traces(), layout(), {{responsive: true, displaylogo: false}});
}}

function summaryTable() {{
  const scen = $("scenario") ? $("scenario").value : DATA.scenarios[0];
  const s = DATA.summary[scen];
  if (!s || !s.length) return "No summary for " + scen;
  let html = "<b>Mean |phi_median| per window (" + scen +
             ", aggregated over series)</b><table><tr><th>Pipeline</th>" +
             "<th>Window</th><th>mean |phi|</th><th>std</th></tr>";
  for (const r of s) {{
    html += `<tr><td>${{r.pipeline}}</td><td>${{r.window}}</td>` +
      `<td>${{r.phi_median_mae.toFixed(2)}}</td><td>${{r.phi_median_std.toFixed(2)}}</td></tr>`;
  }}
  return html + "</table>";
}}

init();
</script>
</body>
</html>
"""


def build_html(data: dict) -> str:
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    payload = payload.replace("</", "<\\/")
    return HTML_TEMPLATE.format(data=payload)


def main() -> None:
    ap = argparse.ArgumentParser(description="WindowSHAP viewer")
    ap.add_argument("--forecast-dir", default="results")
    ap.add_argument("--data-dir", default="data/synthetic")
    ap.add_argument("--out", default="windowshap_viewer.html")
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    args = ap.parse_args()

    forecast_dir, data_dir = Path(args.forecast_dir), Path(args.data_dir)
    shap, summary, used = {}, {}, []
    for scen in args.scenarios:
        df = load_shap(forecast_dir, scen)
        if df is None or df.empty:
            print(f"[Skip] no windowshap_{scen}.parquet")
            continue
        shap[scen] = df.to_dict("records")
        s = load_summary(forecast_dir, scen)
        summary[scen] = s.to_dict("records") if s is not None else []
        used.append(scen)
        print(f"[OK] {scen}: {df.series_id.nunique()} series, "
              f"{len(df)} phi rows")
    if not used:
        raise SystemExit("No WindowSHAP results found.")

    data = {"scenarios": used,
            "shap": shap,
            "summary": summary,
            "series_ids": {s: sorted({r["series_id"] for r in shap[s]})
                           for s in used}}
    out = Path(args.out)
    out.write_text(build_html(data), encoding="utf-8")
    print(f"\\n-> {out} ({out.stat().st_size / 1e6:.1f} MB) created.")


if __name__ == "__main__":
    main()

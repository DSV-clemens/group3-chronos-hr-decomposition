"""
Interactive viewer for the generated HR data (Phase-3 output).

Note: this is a standalone exploration tool for the synthetic benchmark —
it reads ONLY the Phase-3 files (series + event metadata) and never touches
forecast results; those are covered by plot_forecast_results.py. It can be
run right after Phase 03, before any forecast exists.

Creates ONE standalone HTML file (plotly.js via CDN) with:
- scenario (residual/trend/none) and series selection
- TWO panels: hr on top, ground-truth components below on their own y scale
  (residual/seasonal fluctuate around 0, hr around ~60-150 bpm -> a shared
  axis would squeeze the components invisibly together)
- hr / true_trend / true_seasonal / true_residual individually toggleable
- anomaly events as shaded spans + labels (from metadata_*.jsonl)
- context/horizon boundary at t=512 (what Chronos-2 sees vs. has to forecast)
- subject profile parameters (baseline, phi, sigma, slope, period) per series
- summary per scenario (series, points, events, HR bands)

Usage:
    python view_hr_parquet.py --input-dir data/synthetic --out hr_viewer.html
    python view_hr_parquet.py --input-dir data/synthetic --max-series 20   # larger datasets

Reads:  synthetic_hr_mmash_{scenario}.parquet  +  metadata_{scenario}.jsonl
Fallback: .csv.gz (if pyarrow is missing). Pure pandas/numpy dependency;
plotly only runs in the browser (CDN), the script itself does not need it.

Tip for a quick numerical look without the viewer:
    import pandas as pd
    df = pd.read_parquet("data/synthetic/synthetic_hr_mmash_residual.parquet")
    df.head(); df.describe(); df.groupby("series_id")["hr"].agg(["mean","std"])
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCENARIOS = ("residual", "trend", "none")


# ------------------------------------------------------------------ loading

def load_scenario(input_dir: Path, scenario: str) -> tuple[pd.DataFrame, list[dict]]:
    """Loads DataFrame + metadata for one scenario (parquet before csv.gz)."""
    base = input_dir / f"synthetic_hr_mmash_{scenario}"
    for ext, reader in ((".parquet", "read_parquet"), (".csv.gz", "read_csv")):
        p = Path(str(base) + ext)
        if p.exists():
            df = getattr(pd, reader)(p)
            if ext == ".csv.gz":
                print(f"[Note] {p.name} (CSV fallback) loaded.")
            break
    else:
        raise FileNotFoundError(f"No file for scenario '{scenario}' in {input_dir} "
                                f"(searched: {base}.parquet / .csv.gz)")

    meta = []
    meta_path = input_dir / f"metadata_{scenario}.jsonl"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = [json.loads(line) for line in f if line.strip()]
    else:
        print(f"[Note] {meta_path.name} not found -> no event markers.")
    return df, meta


def df_to_records(df: pd.DataFrame, meta: list[dict], max_series: int) -> list[dict]:
    """DataFrame + metadata -> compact JSON-able series records."""
    meta_by_id = {m.get("series_id"): m for m in meta}
    out = []
    ids = sorted(df["series_id"].unique())[:max_series]
    for sid in ids:
        g = df[df["series_id"] == sid].sort_values("t")
        m = meta_by_id.get(sid, {})
        out.append({
            "id": int(sid),
            "t": g["t"].tolist(),
            "hr": [round(v, 2) for v in g["hr"]],
            "trend": [round(v, 2) for v in g["true_trend"]] if "true_trend" in g else None,
            "seasonal": [round(v, 2) for v in g["true_seasonal"]] if "true_seasonal" in g else None,
            "residual": [round(v, 2) for v in g["true_residual"]] if "true_residual" in g else None,
            "events": m.get("events", []),
            "params": m.get("params", {}),
        })
    return out


def scenario_summary(df: pd.DataFrame, meta: list[dict]) -> dict:
    g = df.groupby("series_id")["hr"]
    n_events = sum(len(m.get("events", [])) for m in meta)
    return {
        "n_series": int(g.ngroups),
        "n_points": int(len(df)),
        "length": int(df.groupby("series_id").size().iloc[0]),
        "hr_mean_min": round(float(g.mean().min()), 1),
        "hr_mean_max": round(float(g.mean().max()), 1),
        "hr_span": [round(float(df["hr"].min()), 1), round(float(df["hr"].max()), 1)],
        "n_events": n_events,
    }


# ------------------------------------------------------------------ HTML

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>HR-DGP Viewer</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 16px; color: #1a1a2e; }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 14px; align-items: center;
               margin-bottom: 8px; }}
  label {{ font-size: 13px; display: flex; align-items: center; gap: 4px; }}
  select {{ padding: 3px 6px; }}
  .chk span {{ display: inline-block; width: 10px; height: 10px; margin-right: 3px; }}
  #info {{ font-size: 13px; color: #444; margin: 6px 0 10px 0; white-space: pre-line; }}
  #plot {{ width: 100%; }}
  .card {{ background: #f7f8fa; border: 1px solid #e3e5ea; border-radius: 8px;
           padding: 8px 12px; font-size: 12.5px; }}
  table {{ border-collapse: collapse; font-size: 12.5px; }}
  td, th {{ padding: 2px 10px 2px 0; text-align: left; }}
</style>
</head>
<body>
<h3>Synthetic HR series &mdash; interactive view</h3>
<div class="controls">
  <label>Scenario
    <select id="scenario" onchange="render()"></select></label>
  <label>Series <select id="series" onchange="render()"></select></label>
  <label class="chk"><input type="checkbox" id="hr" checked onchange="render()">
    <span style="background:#1f77b4"></span>hr</label>
  <label class="chk"><input type="checkbox" id="trend" onchange="render()">
    <span style="background:#d62728"></span>true_trend</label>
  <label class="chk"><input type="checkbox" id="seasonal" onchange="render()">
    <span style="background:#2ca02c"></span>true_seasonal</label>
  <label class="chk"><input type="checkbox" id="residual" onchange="render()">
    <span style="background:#9467bd"></span>true_residual</label>
</div>
<div id="info" class="card"></div>
<div id="plot"></div>
<div style="margin-top:14px" class="card" id="summary"></div>

<script>
window.DATA = {data};

const $ = (id) => document.getElementById(id);
const COLOR = {{hr: "#1f77b4", trend: "#d62728", seasonal: "#2ca02c",
                residual: "#9467bd"}};
const SEV_COLOR = {{subtle: "rgba(255,193,7,0.15)", moderate: "rgba(255,152,0,0.2)",
                    severe: "rgba(244,67,54,0.22)"}};

function init() {{
  const sc = $("scenario");
  for (const s of DATA.scenarios) {{
    const o = document.createElement("option");
    o.value = s; o.textContent = `${{s}} (${{DATA.summary[s].n_series}} series)`;
    sc.appendChild(o);
  }}
  sc.onchange = fillSeries;
  fillSeries();          // fills the series list AND draws the initial plot
  $("summary").innerHTML = summaryTable();
}}

function fillSeries() {{
  const s = $("scenario").value;
  const sel = $("series");
  sel.innerHTML = "";
  for (const r of DATA.series[s]) {{
    const o = document.createElement("option");
    o.value = r.id; o.textContent = "Series " + r.id;
    sel.appendChild(o);
  }}
  render();  // scenario switch: rebuild series list + plot
}}

function current() {{
  const s = $("scenario").value;
  const id = parseInt($("series").value);
  return DATA.series[s].find((r) => r.id === id);
}}

function traces(r) {{
  const tr = [];
  const add = (key, label, y, dash, ax) => tr.push({{
    x: r.t, y: y, name: label, visible: $(key).checked, yaxis: ax,
    line: {{color: COLOR[key], width: key === "hr" ? 1.8 : 1.1, dash: dash}}
  }});
  add("hr", "hr (pipeline view)", r.hr, "", "y");
  if (r.trend)    add("trend", "true_trend", r.trend, "dot", "y2");
  if (r.seasonal) add("seasonal", "true_seasonal", r.seasonal, "dot", "y2");
  if (r.residual) add("residual", "true_residual", r.residual, "dot", "y2");
  return tr;
}}

function layout(r, contextAt) {{
  const shapes = [], annotations = [];
  for (const e of r.events) {{
    const x0 = e.start, x1 = e.end;
    const col = SEV_COLOR[e.severity] || "rgba(200,200,200,0.18)";
    shapes.push({{type: "rect", x0: x0, x1: x1, y0: 0, y1: 1,
                  yref: "paper", fillcolor: col, line: {{width: 0}}}});
    annotations.push({{x: (x0 + x1) / 2, y: 1.06, yref: "paper",
                      text: `${{e.type}} (${{e.severity}})`, showarrow: false,
                      font: {{size: 11}}}});
  }}
  if (contextAt) {{
    shapes.push({{type: "line", x0: contextAt, x1: contextAt, y0: 0, y1: 1,
                 yref: "paper", line: {{color: "#333", dash: "dash", width: 1}}}});
    annotations.push({{x: contextAt, y: 1.06, yref: "paper", text: "Context | Horizon",
                      showarrow: false, font: {{size: 11}}}});
  }}
  return {{
    margin: {{t: 40, r: 20, b: 40, l: 50}},
    xaxis: {{title: "Minute", range: [0, r.t[r.t.length - 1]]}},
    yaxis: {{title: "hr (bpm)", domain: [0.56, 1.0]}},
    yaxis2: {{title: "true components (bpm)", domain: [0.0, 0.48]}},
    shapes: shapes, annotations: annotations,
    legend: {{orientation: "h", y: 1.12}},
    hovermode: "closest"
  }};
}}

function render() {{
  const r = current();
  Plotly.react("plot", traces(r), layout(r, DATA.context_length),
               {{responsive: true, displaylogo: false}});
  const p = r.params || {{}};
  const lines = [];
  if (p.baseline) lines.push(`Subject profile: baseline=${{p.baseline.toFixed(1)}} bpm, ` +
      `slope=${{p.slope.toFixed(4)}} bpm/min, period=${{p.seasonal_period}} min`);
  if (p.ar_phi !== undefined) lines.push(`AR(1): phi=${{p.ar_phi.toFixed(3)}}, ` +
      `sigma=${{p.ar_sigma.toFixed(2)}} bpm (source: ${{p.profile_source || "?"}})`);
  if (r.events.length)
    lines.push(`Events: ` + r.events.map((e) =>
        `${{e.type}} ${{e.severity}} @${{e.start}}-${{e.end}} (${{e.channel}}, ` +
        `${{e.amplitude.toFixed(1)}} bpm)`).join(" | "));
  else lines.push("Events: none (baseline series)");
  $("info").textContent = lines.join("\\n");
}}

function summaryTable() {{
  let html = "<b>Summary per scenario</b><table><tr><th>Scenario</th>" +
    "<th>Series</th><th>Points</th><th>Length</th>" +
    "<th>HR mean per series</th><th>Overall span</th><th>Events</th></tr>";
  for (const s of DATA.scenarios) {{
    const q = DATA.summary[s];
    html += `<tr><td>${{s}}</td><td>${{q.n_series}}</td><td>${{q.n_points}}</td>` +
      `<td>${{q.length}}</td>` +
      `<td>${{q.hr_mean_min}}\u2013${{q.hr_mean_max}} bpm</td>` +
      `<td>${{q.hr_span[0]}}\u2013${{q.hr_span[1]}} bpm</td><td>${{q.n_events}}</td></tr>`;
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
    payload = payload.replace("</", "<\\/")  # prevent a </script> breakout
    return HTML_TEMPLATE.format(data=payload)


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive HR parquet viewer")
    ap.add_argument("--input-dir", default="data/synthetic", help="Folder with the scenario files")
    ap.add_argument("--out", default="hr_viewer.html")
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--max-series", type=int, default=150,
                    help="Max. series per scenario to embed (HTML size)")
    ap.add_argument("--context-length", type=int, default=512,
                    help="Context/horizon boundary to draw in the plot")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    series, summary, used = {}, {}, []
    for scen in args.scenarios:
        try:
            df, meta = load_scenario(input_dir, scen)
        except FileNotFoundError as e:
            print(f"[Skip] {e}")
            continue
        series[scen] = df_to_records(df, meta, args.max_series)
        summary[scen] = scenario_summary(df, meta)
        used.append(scen)
        print(f"[OK] {scen}: {summary[scen]['n_series']} series, "
              f"{summary[scen]['n_points']} points, {summary[scen]['n_events']} events")
    if not used:
        raise SystemExit(f"No scenario files found in '{input_dir}'.")

    data = {
        "scenarios": used,
        "series": series,
        "summary": summary,
        "context_length": args.context_length,
    }
    out = Path(args.out)
    out.write_text(build_html(data), encoding="utf-8")
    size_mb = out.stat().st_size / 1e6
    print(f"\n-> {out} ({size_mb:.1f} MB) created. Open it in a browser to explore.")


if __name__ == "__main__":
    main()

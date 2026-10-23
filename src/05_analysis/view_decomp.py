"""
Interactive A/B comparison viewer: raw Chronos-2 vs. decomposition pipeline.

For a selected scenario/series, ONE standalone HTML file (plotly CDN) shows:
- context window (grey) + actuals (black, context & horizon)
- A/B mode switch (radio buttons):
    * "A: raw"        - median + quantile band of the raw pipeline (chronos2)
    * "B: decomp"     - recomposed median + band + the three component
                        forecasts on a second panel (trend/seasonal/resid),
                        PLUS the STL decomposition of the CONTEXT window, so
                        the components are visible over the whole series
    * "A vs. B"       - both medians + both bands overlaid (overview)
- anomaly event markers + the context/horizon boundary at t=512
- per-series metrics for both pipelines (from metrics_per_series.csv)
- summary table (median per label x scenario)

Usage:
    python src/05_analysis/view_decomp.py --forecast-dir results \
        --data-dir data/synthetic --out decomp_viewer.html
    python src/05_analysis/view_decomp.py --decomp-label decomp ...  # old
        DGP-metadata-based runs instead of the ACF-estimated ones

Reads:  results/forecasts_chronos2_{scenario}.*   (pipeline A)
        results/forecasts_{decomp-label}_{scenario}.*  (pipeline B + component
                                                  cols; default 'decomp_acf')
        results/stl_context_{scenario}.*         (STL decomposition of the
                                                  context, written by
                                                  run_forecast_decomp.py)
        results/metrics_per_series.csv, results/summary_scenario.csv
        data/synthetic/synthetic_hr_mmash_{scenario}.* + metadata (context, events)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCENARIOS = ("residual", "trend", "none")
COMPONENTS = ("trend", "seasonal", "resid")


# ------------------------------------------------------------------ loading

def load_forecast(forecast_dir: Path, label: str, scenario: str) -> pd.DataFrame | None:
    base = forecast_dir / f"forecasts_{label}_{scenario}"
    for ext, reader in ((".parquet", "read_parquet"), (".csv.gz", "read_csv")):
        p = Path(str(base) + ext)
        if p.exists():
            return getattr(pd, reader)(p)
    return None


def load_stl_context(forecast_dir: Path, scenario: str) -> pd.DataFrame | None:
    base = forecast_dir / f"stl_context_{scenario}"
    for ext, reader in ((".parquet", "read_parquet"), (".csv.gz", "read_csv")):
        p = Path(str(base) + ext)
        if p.exists():
            return getattr(pd, reader)(p)
    return None


def load_context_events(data_dir: Path, scenario: str, sid: int,
                        context_length: int) -> tuple[dict | None, list[dict]]:
    base = data_dir / f"synthetic_hr_mmash_{scenario}"
    ctx = None
    for ext, reader in ((".parquet", "read_parquet"), (".csv.gz", "read_csv")):
        p = Path(str(base) + ext)
        if p.exists():
            df = getattr(pd, reader)(p)
            g = df[(df["series_id"] == sid) & (df["t"] < context_length)]
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


def series_record(forecast_dir: Path, data_dir: Path, scenario: str, sid: int,
                  context_length: int, decomp_label: str = "decomp_acf") -> dict | None:
    """Aggregates pipeline A, pipeline B (incl. components) and the STL
    context decomposition for one series."""
    rec: dict = {"id": int(sid)}
    raw = load_forecast(forecast_dir, "chronos2", scenario)
    dec = load_forecast(forecast_dir, decomp_label, scenario)
    for label, df in (("raw", raw), ("dec", dec)):
        if df is None:
            return None
        g = df[df["series_id"] == sid].sort_values("t")
        if g.empty:
            return None
        entry = {"t": g["t"].tolist(),
                 "actual": [round(v, 2) for v in g["actual"]],
                 "med": [round(v, 2) for v in g["q0.5"]],
                 "lo": [round(v, 2) for v in g["q0.1"]],
                 "hi": [round(v, 2) for v in g["q0.9"]]}
        if label == "dec":
            for c in COMPONENTS:
                col = f"{c}_q0.5"
                if col in g.columns:
                    entry[f"{c}_med"] = [round(v, 2) for v in g[col]]
        rec[label] = entry
    if len(rec["raw"]["t"]) != len(rec["dec"]["t"]):
        return None

    # STL decomposition of the context window (full components over context)
    stl = load_stl_context(forecast_dir, scenario)
    if stl is not None and sid in set(stl["series_id"]):
        g = stl[stl["series_id"] == sid].sort_values("t")
        rec["stl"] = {"t": g["t"].tolist(),
                      **{c: [round(v, 2) for v in g[c]] for c in COMPONENTS}}
    else:
        rec["stl"] = None

    ctx, events = load_context_events(data_dir, scenario, int(sid), context_length)
    rec["ctx"] = ctx
    rec["events"] = events
    return rec


def load_metrics(forecast_dir: Path) -> dict:
    p = forecast_dir / "metrics_per_series.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    out = {}
    for r in df.to_dict("records"):
        out[f"{r['label']}|{r['scenario']}|{r['series_id']}"] = {
            "mae": round(float(r["mae"]), 2),
            "wql": round(float(r["wql"]), 4),
            "coverage": round(float(r["coverage"]), 3),
            "nominal": float(r.get("nominal", np.nan)),
        }
    return out


def load_summary(forecast_dir: Path) -> list[dict]:
    p = forecast_dir / "summary_scenario.csv"
    if not p.exists():
        return []
    return pd.read_csv(p).round(4).to_dict("records")


# ------------------------------------------------------------------ HTML

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>A/B Viewer: raw vs. decomposition</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 16px; color: #1a1a2e; }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 14px; align-items: center;
               margin-bottom: 8px; }}
  label {{ font-size: 13px; display: flex; align-items: center; gap: 4px; }}
  select {{ padding: 3px 6px; }}
  .mode {{ display: flex; gap: 2px; border: 1px solid #cfd4dc; border-radius: 6px;
           overflow: hidden; }}
  .mode button {{ border: 0; padding: 4px 12px; font-size: 13px; cursor: pointer;
                  background: #fff; color: #1a1a2e; }}
  .mode button.active {{ background: #1a1a2e; color: #fff; }}
  #info {{ font-size: 13px; color: #444; margin: 6px 0 10px 0; white-space: pre-line; }}
  #plot {{ width: 100%; }}
  .card {{ background: #f7f8fa; border: 1px solid #e3e5ea; border-radius: 8px;
           padding: 8px 12px; font-size: 12.5px; }}
  table {{ border-collapse: collapse; font-size: 12.5px; }}
  td, th {{ padding: 2px 10px 2px 0; text-align: left; }}
</style>
</head>
<body>
<h3>Forecast A/B: raw Chronos-2 vs. decomposition pipeline</h3>
<div class="controls">
  <div class="mode">
    <button id="m-raw"   onclick="setMode('raw')">A: raw</button>
    <button id="m-dec"   onclick="setMode('dec')">B: decomp</button>
    <button id="m-both" class="active" onclick="setMode('both')">A vs. B</button>
  </div>
  <label>Scenario <select id="scenario" onchange="fillSeries()"></select></label>
  <label>Series <select id="series" onchange="render()"></select></label>
</div>
<div id="info" class="card"></div>
<div id="plot"></div>
<div style="margin-top:14px" class="card" id="summary"></div>

<script>
window.DATA = {data};
const DECOMP_LABEL = "{decomp_label}";

const $ = (id) => document.getElementById(id);
let MODE = "both";
const SEV_COLOR = {{subtle: "rgba(255,193,7,0.15)", moderate: "rgba(255,152,0,0.2)",
                    severe: "rgba(244,67,54,0.22)"}};
const RAW_MED = "#1f77b4", DEC_MED = "#d62728";
const COMP_COLOR = {{trend: "#ff7f0e", seasonal: "#2ca02c", resid: "#9467bd"}};

function setMode(m) {{
  MODE = m;
  for (const k of ["raw", "dec", "both"]) $("m-" + k).classList.toggle("active", k === m);
  render();
}}

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
  for (const r of DATA.series[s]) {{
    const o = document.createElement("option");
    o.value = r.id; o.textContent = "Series " + r.id; sel.appendChild(o);
  }}
  render();
}}

function current() {{
  const s = $("scenario").value;
  const id = parseInt($("series").value);
  return DATA.series[s].find((r) => r.id === id);
}}

function bandTrace(t, lo, hi, color, name) {{
  return {{x: t.concat(t.slice().reverse()),
          y: hi.concat(lo.slice().reverse()), name: name,
          fill: "toself", fillcolor: color, line: {{width: 0}},
          hoverinfo: "skip", legendgroup: name}};
}}

function traces(r) {{
  // Only the traces of the active mode are emitted at all -> no trace ever
  // references an axis that the layout does not define.
  const tr = [];
  if (r.ctx)
    tr.push({{x: r.ctx.t, y: r.ctx.hr, name: "context (model input)",
              line: {{color: "#9aa0a6", width: 1.2}}, hoverinfo: "skip"}});
  // actuals: ONE continuous curve over context + horizon (in t order)
  const ax = r.ctx ? r.ctx.t.concat(r.raw.t) : r.raw.t;
  const ay = r.ctx ? r.ctx.hr.concat(r.raw.actual) : r.raw.actual;
  tr.push({{x: ax, y: ay, name: "actual", line: {{color: "#111", width: 1.6}}}});

  if (MODE === "raw" || MODE === "both") {{
    tr.push(bandTrace(r.raw.t, r.raw.lo, r.raw.hi, "rgba(31,119,180,0.14)",
                      "raw band (chronos2)"));
    tr.push({{x: r.raw.t, y: r.raw.med, name: "raw median (chronos2)",
              line: {{color: RAW_MED, width: 1.8}}, legendgroup: "raw"}});
  }}
  if (MODE === "dec" || MODE === "both") {{
    tr.push(bandTrace(r.dec.t, r.dec.lo, r.dec.hi, "rgba(214,39,40,0.14)",
                      "decomp band"));
    tr.push({{x: r.dec.t, y: r.dec.med, name: "decomp median (recomposed)",
              line: {{color: DEC_MED, width: 1.8, dash: "dash"}},
              legendgroup: "decomp"}});
  }}
  // component traces (pipeline B): STL over the context + per-component
  // forecast in the horizon. Only meaningful where the decomposition is in
  // play (modes B and A-vs-B); in mode A: raw the raw pipeline uses no
  // decomposition -> no component panel there.
  if (MODE !== "raw") {{
    for (const c of ["trend", "seasonal", "resid"]) {{
      if (!r.dec[c + "_med"]) continue;
      const stl = r.stl && r.stl[c] ? r.stl[c] : null;
      const x = stl ? r.stl.t.concat(r.dec.t) : r.dec.t;
      const y = stl ? stl.concat(r.dec[c + "_med"]) : r.dec[c + "_med"];
      tr.push({{x: x, y: y, yaxis: "y2", name: "decomp: " + c,
                line: {{color: COMP_COLOR[c], width: 1.3}}}});
    }}
  }}
  return tr;
}}

function layout(r) {{
  const shapes = [], annotations = [];
  for (const e of r.events) {{
    const col = SEV_COLOR[e.severity] || "rgba(200,200,200,0.18)";
    shapes.push({{type: "rect", x0: e.start, x1: e.end, y0: 0, y1: 1,
                  yref: "paper", fillcolor: col, line: {{width: 0}}}});
    annotations.push({{x: (e.start + e.end) / 2, y: 1.06, yref: "paper",
                      text: `${{e.type}} (${{e.severity}})`, showarrow: false,
                      font: {{size: 11}}}});
  }}
  const ctxEnd = r.ctx ? r.ctx.t[r.ctx.t.length - 1] : null;
  if (ctxEnd !== null) {{
    shapes.push({{type: "line", x0: ctxEnd + 1, x1: ctxEnd + 1, y0: 0, y1: 1,
                 yref: "paper", line: {{color: "#333", dash: "dash", width: 1}}}});
    annotations.push({{x: ctxEnd + 1, y: 1.06, yref: "paper",
                      text: "Context | Horizon", showarrow: false,
                      font: {{size: 11}}}});
  }}
  // The hr panel keeps the SAME pixel height in EVERY mode (its y-axis is
  // therefore identically scaled when switching modes). In modes B and
  // A-vs-B the component panel is added BELOW it (the plot grows); in mode
  // A: raw no decomposition is in play -> single full panel.
  const T = 40, B = 40;                       // top/bottom margins
  const HR_H = 460;                           // hr plot area, fixed everywhere
  const COMP_H = 400;                         // component plot area
  if (MODE === "raw") {{
    return {{
      margin: {{t: T, r: 20, b: B, l: 50}}, height: HR_H + T + B,
      xaxis: {{title: "Minute"}},
      yaxis: {{title: "hr (bpm)"}},
      shapes: shapes, annotations: annotations,
      legend: {{orientation: "h", y: 1.12}},
      hovermode: "closest"
    }};
  }}
  const total = HR_H + COMP_H + T + B;        // area above = hr, below = comps
  const frac = (HR_H + COMP_H) / total;        // usable (non-margin) fraction
  const comp_top = COMP_H / (HR_H + COMP_H);  // upper edge of comp panel
  const gap = 0.03;                           // small gap between panels
  return {{
    margin: {{t: T, r: 20, b: B, l: 50}}, height: total,
    xaxis: {{title: "Minute"}},
    yaxis: {{title: "hr (bpm)", domain: [comp_top * frac + gap * frac, 1.0]}},
    yaxis2: {{title: "components (bpm)",
              domain: [0.0, (comp_top - gap) * frac]}},
    shapes: shapes, annotations: annotations,
    legend: {{orientation: "h", y: 1.12}},
    hovermode: "closest"
  }};
}}

function render() {{
  const r = current();
  if (!r) {{ $("info").textContent = "No data."; Plotly.purge("plot"); return; }}
  Plotly.react("plot", traces(r), layout(r), {{responsive: true, displaylogo: false}});
  const scen = $("scenario").value;
  const fmt = (label) => {{
    const m = DATA.metrics[`${{label}}|${{scen}}|${{r.id}}`];
    if (!m) return `${{label}}: n/a`;
    return `${{label}}: MAE=${{m.mae.toFixed(2)}} | WQL=${{m.wql.toFixed(4)}} | ` +
           `cov=${{(m.coverage * 100).toFixed(1)}}% (nom. ${{(m.nominal * 100).toFixed(0)}}%)`;
  }};
  const lines = ["Pipeline A  " + fmt("chronos2"),
                 "Pipeline B  " + fmt(DECOMP_LABEL)];
  if (r.events.length)
    lines.push("Events: " + r.events.map((e) =>
        `${{e.type}} ${{e.severity}} @${{e.start}}\u2013${{e.end}} (${{e.channel}}, ` +
        `${{e.amplitude.toFixed(1)}} bpm)`).join(" | "));
  else lines.push("Events: none");
  $("info").textContent = lines.join("\\n");
}}

function summaryTable() {{
  if (!DATA.summary.length) return "No summary_scenario.csv found.";
  let html = "<b>Summary (median per label &times; scenario)</b>" +
    "<table><tr><th>Label</th><th>Scenario</th><th>MAE</th><th>WQL</th>" +
    "<th>Coverage</th><th>Series</th></tr>";
  for (const s of DATA.summary) {{
    html += `<tr><td>${{s.label}}</td><td>${{s.scenario}}</td>` +
      `<td>${{s.mae.toFixed(2)}}</td><td>${{s.wql.toFixed(4)}}</td>` +
      `<td>${{(s.coverage * 100).toFixed(1)}}%</td><td>${{s.n_series}}</td></tr>`;
  }}
  return html + "</table>";
}}

init();
</script>
</body>
</html>
"""


def build_html(data: dict, decomp_label: str) -> str:
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    payload = payload.replace("</", "<\\/")  # prevent a </script> breakout
    return HTML_TEMPLATE.format(data=payload, decomp_label=decomp_label)


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description="A/B viewer: raw vs. decomposition")
    ap.add_argument("--forecast-dir", default="results")
    ap.add_argument("--data-dir", default="data/synthetic")
    ap.add_argument("--out", default="decomp_viewer.html")
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--context-length", type=int, default=512)
    ap.add_argument("--max-series", type=int, default=150)
    ap.add_argument("--decomp-label", default="decomp_acf",
                    help="Forecast label of pipeline B to display "
                         "(e.g. 'decomp' for the old DGP-metadata runs, "
                         "'decomp_acf' for the ACF-estimated runs)")
    args = ap.parse_args()

    forecast_dir, data_dir = Path(args.forecast_dir), Path(args.data_dir)
    series, used = {}, []
    for scen in args.scenarios:
        raw = load_forecast(forecast_dir, "chronos2", scen)
        dec = load_forecast(forecast_dir, args.decomp_label, scen)
        if raw is None or dec is None:
            print(f"[Skip] scenario '{scen}': needs forecasts_chronos2_* AND "
                  f"forecasts_{args.decomp_label}_*")
            continue
        ids = sorted(set(raw["series_id"]) & set(dec["series_id"]))[:args.max_series]
        recs = []
        for sid in ids:
            r = series_record(forecast_dir, data_dir, scen, int(sid),
                              args.context_length, args.decomp_label)
            if r:
                recs.append(r)
        if recs:
            series[scen] = recs
            used.append(scen)
            has_stl = sum(1 for r in recs if r["stl"])
            print(f"[OK] {scen}: {len(recs)} series embedded "
                  f"(STL context: {has_stl})")
    if not used:
        raise SystemExit("No scenario with both pipelines found.")

    data = {
        "scenarios": used,
        "series": series,
        "metrics": load_metrics(forecast_dir),
        "summary": load_summary(forecast_dir),
        "context_length": args.context_length,
    }
    out = Path(args.out)
    out.write_text(build_html(data, args.decomp_label), encoding="utf-8")
    print(f"\n-> {out} ({out.stat().st_size / 1e6:.1f} MB) created. "
          f"Open it in a browser to explore.")


if __name__ == "__main__":
    main()

"""
Interactive viewer for the Phase-4 forecast results (run_forecast.py output).

Companion to view_hr_parquet.py (which shows only the Phase-3 series): this
tool shows WHAT THE MODELS FORECAST for each series — the context window
(from the Phase-3 parquet), the actuals, and the quantile band + median
forecast of one or more labels (e.g. chronos2 vs. naive vs. dlinear), plus
the per-series metrics from metrics_per_series.csv and the event metadata.

Creates ONE standalone HTML file (plotly.js via CDN) with:
- MODEL TOGGLE: checkboxes for every discovered label (chronos2 / naive /
  dlinear / decomp / decomp_acf / ...); from none to all at once, the
  selected medians and bands are overlaid in per-label colors
- scenario (residual/trend/none) and series selection
- context (grey) + actual (black) + per-label median forecast and outer
  quantile band (q_min / q_max are taken from the forecast files)
- anomaly events as shaded spans + labels (from metadata_*.jsonl)
- context/horizon boundary at t=512
- per-series metrics (MAE, WQL, coverage) of ALL toggled labels side by side
- summary per label x scenario (median MAE / WQL / coverage, from
  summary_scenario.csv)

Usage:
    python view_forecasts.py --forecast-dir results --data-dir data/synthetic \
        --out forecast_viewer.html

Reads:  results/forecasts_{label}_{scenario}.parquet (or .csv.gz)
        results/metrics_per_series.csv (optional, per-series metrics)
        results/summary_scenario.csv  (optional, summary table)
        data/synthetic/synthetic_hr_mmash_{scenario}.parquet  (context window)
        data/synthetic/metadata_{scenario}.jsonl               (event markers)
Pure pandas/numpy dependency; plotly runs in the browser (CDN).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCENARIOS = ("residual", "trend", "none")


# ------------------------------------------------------------------ loading

def discover_labels(forecast_dir: Path, scenarios: list[str]) -> list[str]:
    """Labels = everything for which a forecasts_{label}_{scenario} file exists."""
    labels = []
    for p in sorted(forecast_dir.glob("forecasts_*")):
        stem = p.name.split(".")[0]
        if not stem.startswith("forecasts_"):
            continue
        label = stem[len("forecasts_"):]
        if label.endswith(tuple(f"_{s}" for s in scenarios)):
            label = label.rsplit("_", 1)[0]
        if label and label not in labels:
            labels.append(label)
    return labels


def load_forecast(forecast_dir: Path, label: str, scenario: str) -> pd.DataFrame | None:
    base = forecast_dir / f"forecasts_{label}_{scenario}"
    for ext, reader in ((".parquet", "read_parquet"), (".csv.gz", "read_csv")):
        p = Path(str(base) + ext)
        if p.exists():
            df = getattr(pd, reader)(p)
            return df
    return None


def load_context_series(data_dir: Path, scenario: str, sid: int,
                        context_length: int) -> dict | None:
    """Context window of series sid from the Phase-3 parquet (grey background)."""
    base = data_dir / f"synthetic_hr_mmash_{scenario}"
    for ext, reader in ((".parquet", "read_parquet"), (".csv.gz", "read_csv")):
        p = Path(str(base) + ext)
        if p.exists():
            df = getattr(pd, reader)(p)
            g = df[(df["series_id"] == sid) & (df["t"] < context_length)]
            return {"t": g["t"].tolist(),
                    "hr": [round(v, 2) for v in g["hr"]]}
    return None


def load_events(data_dir: Path, scenario: str, sid: int) -> list[dict]:
    meta_path = data_dir / f"metadata_{scenario}.jsonl"
    if not meta_path.exists():
        return []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                m = json.loads(line)
                if m.get("series_id") == sid:
                    return m.get("events", [])
    return []


# ------------------------------------------------------------------ records

def forecast_to_records(forecast_dir: Path, label: str, scenario: str,
                         data_dir: Path, context_length: int,
                         max_series: int) -> tuple[list[dict], list[float]]:
    df = load_forecast(forecast_dir, label, scenario)
    if df is None:
        return [], []
    qcols = sorted([c for c in df.columns if c.startswith("q")],
                   key=lambda c: float(c[1:]))
    quantiles = [float(c[1:]) for c in qcols]
    lo_col, med_col, hi_col = qcols[0], qcols[qcols.index("q0.5")], qcols[-1]
    records, ids = [], sorted(df["series_id"].unique())[:max_series]
    for sid in ids:
        g = df[df["series_id"] == sid].sort_values("t")
        ctx = load_context_series(data_dir, scenario, int(sid), context_length)
        records.append({
            "id": int(sid),
            "t": g["t"].tolist(),
            "actual": [round(v, 2) for v in g["actual"]],
            "med": [round(v, 2) for v in g[med_col]],
            "lo": [round(v, 2) for v in g[lo_col]],
            "hi": [round(v, 2) for v in g[hi_col]],
            "ctx": ctx,
            "events": load_events(data_dir, scenario, int(sid)),
        })
    return records, quantiles


def load_metrics(forecast_dir: Path) -> dict:
    """(label, scenario, series_id) -> metrics row; empty if the file is missing."""
    p = forecast_dir / "metrics_per_series.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    out = {}
    for r in df.to_dict("records"):
        out[(r["label"], r["scenario"], int(r["series_id"]))] = {
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
    df = pd.read_csv(p)
    return df.round(4).to_dict("records")


# ------------------------------------------------------------------ HTML

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Forecast Viewer</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 16px; color: #1a1a2e; }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 14px; align-items: center;
               margin-bottom: 8px; }}
  label {{ font-size: 13px; display: flex; align-items: center; gap: 4px; }}
  select {{ padding: 3px 6px; }}
  #info {{ font-size: 13px; color: #444; margin: 6px 0 10px 0; white-space: pre-line; }}
  #plot {{ width: 100%; }}
  .card {{ background: #f7f8fa; border: 1px solid #e3e5ea; border-radius: 8px;
           padding: 8px 12px; font-size: 12.5px; }}
  table {{ border-collapse: collapse; font-size: 12.5px; }}
  td, th {{ padding: 2px 10px 2px 0; text-align: left; }}
</style>
</head>
<body>
<h3>Forecast results &mdash; interactive view</h3>
<div class="controls">
  <span style="font-size:13px"><b>Models:</b></span>
  <span id="labelbox" style="display:flex;gap:10px;flex-wrap:wrap"></span>
  <button id="all" onclick="setAll(true)" style="padding:2px 8px">all</button>
  <button id="none" onclick="setAll(false)" style="padding:2px 8px">none</button>
</div>
<div class="controls">
  <label>Scenario <select id="scenario" onchange="fillSeries()"></select></label>
  <label>Series <select id="series" onchange="render()"></select></label>
</div>
<div id="info" class="card"></div>
<div id="plot"></div>
<div style="margin-top:14px" class="card" id="summary"></div>

<script>
window.DATA = {data};

const $ = (id) => document.getElementById(id);
const SEV_COLOR = {{subtle: "rgba(255,193,7,0.15)", moderate: "rgba(255,152,0,0.2)",
                    severe: "rgba(244,67,54,0.22)"}};

const LCOLOR = {{chronos2: "#1f77b4", naive: "#7f7f7f", dlinear: "#2ca02c",
                 decomp: "#d62728", decomp_acf: "#9467bd"}};
const lcolor = (l, i) => LCOLOR[l] || ["#ff7f0e", "#8c564b", "#e377c2",
  "#17becf", "#bcbd22"][i % 5];

function init() {{
  const box = $("labelbox"), sc = $("scenario");
  DATA.labels.forEach((l, i) => {{
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.id = "cb_" + l; cb.checked = (i === 0);
    cb.onchange = () => fillSeries();
    const lab = document.createElement("label");
    lab.appendChild(cb);
    lab.appendChild(document.createTextNode("\u25CF " + l));
    lab.style.color = lcolor(l, i);
    box.appendChild(lab);
  }});
  for (const s of DATA.scenarios) {{
    const o = document.createElement("option");
    o.value = s; o.textContent = s; sc.appendChild(o);
  }}
  fillSeries();
  $("summary").innerHTML = summaryTable();
}}

function setAll(v) {{
  for (const l of DATA.labels) $("cb_" + l).checked = v;
  fillSeries();
}}

function activeLabels() {{
  return DATA.labels.filter((l) => $("cb_" + l) && $("cb_" + l).checked);
}}

function fillSeries() {{
  const s = $("scenario").value;
  const sel = $("series");
  sel.innerHTML = "";
  let recs = [];
  for (const l of DATA.labels) {{
    recs = recs.concat((DATA.series[l] || {{}})[s] || []);
  }}
  const ids = [...new Set(recs.map((r) => r.id))].sort((a, b) => a - b);
  for (const id of ids) {{
    const o = document.createElement("option");
    o.value = id; o.textContent = "Series " + id; sel.appendChild(o);
  }}
  render();
}}

function current() {{
  const s = $("scenario").value, acts = activeLabels();
  const id = parseInt($("series").value);
  const out = {{}};
  for (const l of acts) {{
    const r = ((DATA.series[l] || {{}})[s] || []).find((x) => x.id === id);
    if (r) out[l] = r;
  }}
  return out;
}}

function groundTruth(s, id) {{
  for (const l of DATA.labels) {{
    const r = ((DATA.series[l] || {{}})[s] || []).find((x) => x.id === id);
    if (r) return r;
  }}
  return null;
}}

function traces(recs, gt) {{
  const tr = [];
  const any = Object.values(recs)[0] || gt;
  if (!any) return tr;
  if (any.ctx) {{
    tr.push({{x: any.ctx.t, y: any.ctx.hr, name: "context (model input)",
              line: {{color: "#9aa0a6", width: 1.2}}, hoverinfo: "skip"}});
  }}
  tr.push({{x: any.t, y: any.actual, name: "actual",
            mode: "lines", line: {{color: "#111", width: 1.6}}}});
  DATA.labels.forEach((l, i) => {{
    const r = recs[l];
    if (!r) return;
    const col = lcolor(l, i);
    tr.push({{x: r.t.concat(r.t.slice().reverse()),
              y: r.hi.concat(r.lo.slice().reverse()),
              name: l + " band (q" + DATA.qlo + "\u2013q" + DATA.qhi + ")",
              fill: "toself",
              fillcolor: col.replace("#", "#") + "22",
              line: {{width: 0}}, hoverinfo: "skip",
              legendgroup: l, showlegend: false}});
    tr.push({{x: r.t, y: r.med, name: l + " median (q0.5)",
              line: {{color: col, width: 1.8}}, legendgroup: l}});
  }});
  return tr;
}}

function layout(any) {{
  const shapes = [], annotations = [];
  for (const e of any.events) {{
    const col = SEV_COLOR[e.severity] || "rgba(200,200,200,0.18)";
    shapes.push({{type: "rect", x0: e.start, x1: e.end, y0: 0, y1: 1,
                  yref: "paper", fillcolor: col, line: {{width: 0}}}});
    annotations.push({{x: (e.start + e.end) / 2, y: 1.06, yref: "paper",
                      text: `${{e.type}} (${{e.severity}})`, showarrow: false,
                      font: {{size: 11}}}});
  }}
  const ctxEnd = any.ctx ? any.ctx.t[any.ctx.t.length - 1] : null;
  if (ctxEnd !== null) {{
    shapes.push({{type: "line", x0: ctxEnd + 1, x1: ctxEnd + 1, y0: 0, y1: 1,
                 yref: "paper", line: {{color: "#333", dash: "dash", width: 1}}}});
    annotations.push({{x: ctxEnd + 1, y: 1.06, yref: "paper",
                      text: "Context | Horizon", showarrow: false,
                      font: {{size: 11}}}});
  }}
  return {{
    margin: {{t: 40, r: 20, b: 40, l: 50}},
    xaxis: {{title: "Minute"}},
    yaxis: {{title: "hr (bpm)"}},
    shapes: shapes, annotations: annotations,
    legend: {{orientation: "h", y: 1.12}},
    hovermode: "closest"
  }};
}}

function render() {{
  const recs = current();
  const acts = activeLabels();
  const s = $("scenario").value;
  const id = parseInt($("series").value);
  const gt = groundTruth(s, id);
  const any = Object.values(recs)[0] || gt;
  if (!any) {{
    $("info").textContent = "No data for this scenario/series.";
    Plotly.purge("plot"); return;
  }}
  Plotly.react("plot", traces(recs, gt), layout(any),
               {{responsive: true, displaylogo: false}});
  let html = "";
  if (acts.length) {{
    html = `<b>Metrics (series ${{id}}, scenario ${{s}})</b>` +
      "<table><tr><th>Model</th><th>MAE</th><th>WQL</th><th>Coverage</th></tr>";
    for (const l of acts) {{
      const m = DATA.metrics[`${{l}}|${{s}}|${{id}}`];
      if (m) html += `<tr><td>${{l}}</td><td>${{m.mae.toFixed(2)}}</td>` +
        `<td>${{m.wql.toFixed(4)}}</td><td>${{(m.coverage * 100).toFixed(1)}}%</td></tr>`;
      else html += `<tr><td>${{l}}</td><td colspan="3">n/a</td></tr>`;
    }}
    html += "</table>";
  }} else {{
    html = `<b>Ground truth only (series ${{id}}, scenario ${{s}})</b>` +
      " \u2014 toggle at least one model to overlay forecasts.";
  }}
  if (any.events.length)
    html += `<div style="margin-top:6px">Events: ` + any.events.map((e) =>
      `${{e.type}} ${{e.severity}} @${{e.start}}\u2013${{e.end}} (${{e.channel}}, ` +
      `${{e.amplitude.toFixed(1)}} bpm)`).join(" | ") + "</div>";
  else html += `<div style="margin-top:6px">Events: none</div>`;
  $("info").innerHTML = html;
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


def build_html(data: dict) -> str:
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    payload = payload.replace("</", "<\\/")  # prevent a </script> breakout
    return HTML_TEMPLATE.format(data=payload)


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive forecast viewer")
    ap.add_argument("--forecast-dir", default="results",
                    help="Folder with forecasts_{label}_{scenario}.* "
                         "and metrics_per_series.csv")
    ap.add_argument("--data-dir", default="data/synthetic",
                    help="Phase-3 folder (context windows + event metadata)")
    ap.add_argument("--out", default="forecast_viewer.html")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Labels to include (default: all discovered)")
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--context-length", type=int, default=512)
    ap.add_argument("--max-series", type=int, default=150,
                    help="Max. series per label/scenario to embed (HTML size)")
    args = ap.parse_args()

    forecast_dir, data_dir = Path(args.forecast_dir), Path(args.data_dir)
    labels = args.labels or discover_labels(forecast_dir, args.scenarios)
    if not labels:
        raise SystemExit(f"No forecasts_* files found in '{forecast_dir}'.")

    series, used_scen, q_all = {}, [], None
    for label in labels:
        series[label] = {}
        for scen in args.scenarios:
            recs, quantiles = forecast_to_records(
                forecast_dir, label, scen, data_dir, args.context_length,
                args.max_series)
            if not recs:
                print(f"[Skip] no forecasts for label='{label}' scenario='{scen}'")
                continue
            series[label][scen] = recs
            q_all = quantiles
            if scen not in used_scen:
                used_scen.append(scen)
            print(f"[OK] {label}/{scen}: {len(recs)} series embedded")
        if not series[label]:
            del series[label]

    if not series:
        raise SystemExit("No forecast files could be loaded.")

    metrics = load_metrics(forecast_dir)
    metrics_key = {}
    for (label, scen, sid), m in metrics.items():
        metrics_key[f"{label}|{scen}|{sid}"] = m

    data = {
        "labels": list(series.keys()),
        "scenarios": used_scen,
        "series": series,
        "metrics": metrics_key,
        "summary": load_summary(forecast_dir),
        "context_length": args.context_length,
        "qlo": q_all[0] if q_all else 0.1,
        "qhi": q_all[-1] if q_all else 0.9,
    }
    out = Path(args.out)
    out.write_text(build_html(data), encoding="utf-8")
    size_mb = out.stat().st_size / 1e6
    print(f"\n-> {out} ({size_mb:.1f} MB) created. Open it in a browser to explore.")


if __name__ == "__main__":
    main()

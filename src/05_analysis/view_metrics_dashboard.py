"""
Metrics dashboard: ONE standalone HTML file aggregating all pipeline metrics
from their various result files into a single interactive view (plotly CDN).

Sections:
  1. Overview cards   -- median MAE / WQL / coverage per arm x scenario,
                         best arm highlighted, plus DGP provenance (hash, seed).
  2. Accuracy         -- boxplots of per-series MAE / WQL / coverage
                         (arm toggle, scenario toggle).
  3. Horizon          -- MAE + coverage per horizon bucket, all arms overlaid.
  4. Calibration      -- empirical coverage vs. nominal (0.8) per arm x scenario,
                         plus mean band width by horizon bucket.
  5. Explainability   -- WindowSHAP mean |phi_median| per context window
                         (raw vs decomp), + event-attribution contrast bars
                         (event vs clean windows, decomp/raw ratio).
  6. Diagnosis        -- flat-median share, level/trend-following correlations,
                         band absorption per scenario (chronos2).

Reads (all optional -- missing files are skipped with a note):
  results/summary_scenario.csv          (overview cards, boxplot source alt.)
  results/metrics_per_series.csv        (per-series boxplots)
  results/metrics_by_horizon.csv        (horizon curves)
  results/windowshap_summary_{scen}.csv (phi per window)
  results/event_attribution.json        (contrast bars)
  results/flat_median_diagnosis.json    (diagnosis cards)
  results/manifest_*.json               (provenance footer)
  data/synthetic/dgp_config_mmash.json  (dgp_hash / seed / n_series)

Pure pandas/numpy dependency; plotly runs in the browser (CDN).

Usage:
  python src/05_analysis/view_metrics_dashboard.py --results-dir results \
      --data-dir data/synthetic --out metrics_dashboard.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SCENARIOS = ("none", "residual", "trend")
ARMS = ("chronos2", "naive", "dlinear", "decomp_acf")
ARM_COLORS = {"chronos2": "#1f77b4", "naive": "#7f7f7f", "dlinear": "#2ca02c",
              "decomp_acf": "#9467bd"}
SCEN_COLORS = {"none": "#1f77b4", "residual": "#d62728", "trend": "#2ca02c"}


def _read_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def _read_json(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def collect(results_dir: Path, data_dir: Path) -> dict:
    """Collects all available metric tables into one payload dict."""
    payload: dict = {"available": {}, "missing": []}

    summary = _read_csv(results_dir / "summary_scenario.csv")
    if summary is not None:
        payload["summary"] = summary.round(4).to_dict("records")
        payload["available"]["summary_scenario.csv"] = len(summary)
    else:
        payload["missing"].append("summary_scenario.csv")

    mps = _read_csv(results_dir / "metrics_per_series.csv")
    if mps is not None:
        payload["metrics_per_series"] = mps.round(4).to_dict("records")
        payload["available"]["metrics_per_series.csv"] = len(mps)
    else:
        payload["missing"].append("metrics_per_series.csv")

    mbh = _read_csv(results_dir / "metrics_by_horizon.csv")
    if mbh is not None:
        payload["metrics_by_horizon"] = mbh.round(4).to_dict("records")
        payload["available"]["metrics_by_horizon.csv"] = len(mbh)
    else:
        payload["missing"].append("metrics_by_horizon.csv")

    ws_tables = {}
    for scen in ("residual", "trend"):
        t = _read_csv(results_dir / f"windowshap_summary_{scen}.csv")
        if t is not None:
            ws_tables[scen] = t.round(4).to_dict("records")
            payload["available"][f"windowshap_summary_{scen}.csv"] = len(t)
        else:
            payload["missing"].append(f"windowshap_summary_{scen}.csv")
    payload["windowshap_summary"] = ws_tables

    ea = _read_json(results_dir / "event_attribution.json")
    if ea is not None:
        payload["event_attribution"] = ea.get("rows", [])
        payload["available"]["event_attribution.json"] = len(ea.get("rows", []))
    else:
        payload["missing"].append("event_attribution.json")

    fm = _read_json(results_dir / "flat_median_diagnosis.json")
    if fm is not None:
        payload["flat_median"] = fm
        payload["available"]["flat_median_diagnosis.json"] = len(fm.get("scenarios", {}))
    else:
        payload["missing"].append("flat_median_diagnosis.json")

    manifests = {}
    for p in sorted(results_dir.glob("manifest_*.json")):
        m = _read_json(p)
        if m is not None:
            manifests[p.stem.replace("manifest_", "")] = {
                "model": m.get("model"),
                "created": m.get("created"),
                "dgp_hash": (m.get("dgp") or {}).get("dgp_hash"),
                "n_series": m.get("n_series"),
                "quantiles": m.get("quantiles"),
                "context": m.get("context"),
                "horizon": m.get("horizon"),
            }
    payload["manifests"] = manifests
    if manifests:
        payload["available"]["manifests"] = len(manifests)

    dgp = {}
    cfg = _read_json(data_dir / "dgp_config_mmash.json")
    if cfg is not None:
        dgp = {"dgp_hash": cfg.get("dgp_hash"), "seed": cfg.get("seed"),
               "n_series": cfg.get("n_series"),
               "generator_version": cfg.get("generator_version")}
    payload["dgp"] = dgp

    return payload


# ------------------------------------------------------------------ HTML

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Metrics Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 16px; color: #1a1a2e; background: #fafbfc; }}
  h2 {{ border-bottom: 2px solid #d0d4dc; padding-bottom: 4px; margin-top: 28px; }}
  .cards {{ display: flex; flex-wrap: wrap; gap: 10px; }}
  .card {{ background: #fff; border: 1px solid #e3e5ea; border-radius: 8px;
           padding: 10px 14px; min-width: 150px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
  .card .big {{ font-size: 22px; font-weight: 600; }}
  .card .lbl {{ font-size: 11.5px; color: #666; }}
  .best {{ border-color: #2ca02c; background: #f0f9f0; }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 14px; align-items: center;
               margin: 8px 0; font-size: 13px; }}
  select, button {{ padding: 3px 6px; }}
  .plot {{ width: 100%; background: #fff; border: 1px solid #e3e5ea;
           border-radius: 8px; margin-bottom: 10px; }}
  table {{ border-collapse: collapse; font-size: 12.5px; background: #fff; }}
  td, th {{ padding: 3px 12px 3px 0; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ color: #555; font-weight: 600; }}
  .note {{ font-size: 12px; color: #888; }}
  .missing {{ color: #b00; }}
  #footer {{ margin-top: 30px; font-size: 11.5px; color: #888; }}
</style>
</head>
<body>
<h2>Metrics Dashboard &mdash; chronos-hr-decomposition</h2>
<div id="dgpinfo" class="note"></div>
<div id="avail" class="note"></div>

<h2>1 &middot; Overview (median per arm &times; scenario)</h2>
<div class="controls">
  <span><b>Metric:</b>
    <select id="ovMetric" onchange="renderOverview()">
      <option value="mae">MAE [bpm] (lower = better)</option>
      <option value="wql">WQL (lower = better)</option>
      <option value="coverage">Coverage (target 0.8)</option>
    </select>
  </span>
</div>
<div id="overviewCards" class="cards"></div>
<div id="overviewTable"></div>

<h2>2 &middot; Accuracy &amp; Uncertainty (per-series distribution)</h2>
<div class="controls">
  <span><b>Metric:</b>
    <select id="distMetric" onchange="renderDist()">
      <option value="mae">MAE [bpm]</option>
      <option value="wql">WQL</option>
      <option value="coverage">Coverage</option>
    </select>
  </span>
  <span><b>Scenario:</b>
    <select id="distScen" onchange="renderDist()"></select>
  </span>
  <span><b>Arms:</b> <span id="distArms"></span></span>
</div>
<div id="distPlot" class="plot"></div>

<h2>3 &middot; Horizon-resolved (MAE &amp; coverage by lead time)</h2>
<div class="controls">
  <span><b>Scenario:</b> <select id="horScen" onchange="renderHorizon()"></select></span>
  <span><b>Metric:</b>
    <select id="horMetric" onchange="renderHorizon()">
      <option value="mae">MAE [bpm]</option>
      <option value="coverage">Coverage</option>
      <option value="band_width">Band width [bpm]</option>
    </select>
  </span>
</div>
<div id="horizonPlot" class="plot"></div>

<h2>4 &middot; Calibration (coverage vs. nominal 0.8)</h2>
<div id="calibPlot" class="plot"></div>
<div id="calibTable"></div>

<h2>5 &middot; Explainability (WindowSHAP)</h2>
<div class="controls">
  <span><b>Scenario:</b> <select id="wsScen" onchange="renderWS()"></select></span>
  <span><b>Value:</b>
    <select id="wsVal" onchange="renderWS()">
      <option value="phi_median_mae">|phi_median| (mean)</option>
      <option value="phi_band_mae">|phi_band| (mean)</option>
    </select>
  </span>
</div>
<div id="wsPlot" class="plot"></div>
<div id="contrastPlot" class="plot"></div>
<div id="contrastTable"></div>

<h2>6 &middot; Flat-median diagnosis (chronos2)</h2>
<div id="diagCards" class="cards"></div>
<div id="diagPlot" class="plot"></div>

<h2>Provenance</h2>
<div id="footer"></div>

<script>
window.DATA = @@DATA@@;

const $ = (id) => document.getElementById(id);
const ARMS = @@ARMS@@;
const ARM_COLORS = @@ARM_COLORS@@;
const SCENS = @@SCENS@@;
const SCEN_COLORS = @@SCEN_COLORS@@;
const NOMINAL = 0.8;

const fmt = (v, d=3) => (typeof v === "number") ? v.toFixed(d) : v;
const esc = (s) => String(s).replace(/[&<>]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;"}}[c]));

// ---------- header ----------
(function() {{
  const d = window.DATA.dgp || {{}};
  $("dgpinfo").innerHTML = "<b>DGP:</b> hash <code>" + esc(d.dgp_hash || "?") + "</code>" +
    " | seed " + esc(d.seed ?? "?") + " | n_series " + esc(d.n_series ?? "?") +
    " | generator " + esc(d.generator_version || "?");
  const av = window.DATA.available || {{}};
  const parts = Object.entries(av).map(([k,v]) => k + " (" + v + ")");
  $("avail").innerHTML = "<b>Loaded:</b> " + parts.join(" &middot; ") +
    (window.DATA.missing.length ? ' &nbsp; <span class="missing">missing: ' +
      window.DATA.missing.join(", ") + "</span>" : "");
}})();

// ---------- helpers ----------
function fillSelect(id, values) {{
  const s = $(id); s.innerHTML = "";
  values.forEach(v => {{ const o = document.createElement("option");
    o.value = v; o.textContent = v; s.appendChild(o); }});
}}
function armCheckboxes(containerId, onchange) {{
  const c = $(containerId); c.innerHTML = "";
  ARMS.forEach(a => {{
    const l = document.createElement("label");
    l.innerHTML = '<input type="checkbox" checked value="' + a + '" onchange="' + onchange + '">' +
      '<span style="color:' + ARM_COLORS[a] + '">' + a + "</span>";
    c.appendChild(l);
  }});
}}
function selectedArms(containerId) {{
  return Array.from($(containerId).querySelectorAll("input:checked")).map(i => i.value);
}}

// ---------- 1. overview ----------
function renderOverview() {{
  const metric = $("ovMetric").value;
  const rows = window.DATA.summary || [];
  const byScen = {{}};
  rows.forEach(r => {{ (byScen[r.scenario] = byScen[r.scenario] || []).push(r); }});
  const wrap = $("overviewCards"); wrap.innerHTML = "";
  const tbl = $("overviewTable"); tbl.innerHTML = "";
  SCENS.forEach(scen => {{
    const rs = (byScen[scen] || []).filter(r => ARMS.includes(r.label));
    if (!rs.length) return;
    const vals = rs.map(r => r[metric]);
    const best = metric === "coverage"
      ? rs.reduce((a,b) => Math.abs(b[metric]-NOMINAL) < Math.abs(a[metric]-NOMINAL) ? b : a)
      : rs.reduce((a,b) => b[metric] < a[metric] ? b : a);
    // cards
    const h = document.createElement("div");
    h.innerHTML = '<div style="font-size:12px;color:#666;margin-bottom:6px">' + scen + "</div>";
    rs.forEach(r => {{
      const c = document.createElement("div");
      c.className = "card" + (r.label === best.label ? " best" : "");
      c.style.minWidth = "120px";
      c.innerHTML = '<div class="lbl">' + r.label + "</div>" +
        '<div class="big">' + fmt(r[metric], metric === "mae" ? 2 : 3) + "</div>" +
        '<div class="lbl">n=' + r.n_series + "</div>";
      h.appendChild(c);
    }});
    wrap.appendChild(h);
    // table row
  }});
  // full table
  let html = "<table><tr><th>scenario</th><th>arm</th><th>MAE</th><th>WQL</th>" +
             "<th>coverage</th><th>nominal</th><th>n</th></tr>";
  rows.filter(r => ARMS.includes(r.label)).forEach(r => {{
    html += "<tr><td>" + r.scenario + "</td><td style='color:" + ARM_COLORS[r.label] + "'>" +
      r.label + "</td><td>" + fmt(r.mae,2) + "</td><td>" + fmt(r.wql) + "</td><td>" +
      fmt(r.coverage) + "</td><td>" + fmt(r.nominal,2) + "</td><td>" + r.n_series + "</td></tr>";
  }});
  tbl.innerHTML = html + "</table>";
}}

// ---------- 2. distribution boxplots ----------
function renderDist() {{
  const metric = $("distMetric").value, scen = $("distScen").value;
  const arms = selectedArms("distArms");
  const rows = (window.DATA.metrics_per_series || [])
    .filter(r => r.scenario === scen && arms.includes(r.label));
  const traces = arms.map(a => ({{
    y: (window.DATA.metrics_per_series || [])
       .filter(r => r.scenario === scen && r.label === a).map(r => r[metric]),
    name: a, type: "box", boxpoints: false, marker: {{color: ARM_COLORS[a]}},
    legendgroup: a
  }}));
  Plotly.newPlot("distPlot", traces, {{
    title: metric.toUpperCase() + " per series &mdash; " + scen + " (n=" +
           (arms.length ? rows.filter(r=>r.label===arms[0]).length : 0) + " per arm)",
    yaxis: {{title: metric, rangemode: "tozero"}},
    height: 420, margin: {{l: 60, r: 20, t: 50, b: 40}}
  }}, {{responsive: true}});
}}

// ---------- 3. horizon ----------
function bucketOrder(b) {{ return parseInt(b.replace("h<= ", "")); }}
function renderHorizon() {{
  const scen = $("horScen").value, metric = $("horMetric").value;
  const rows = (window.DATA.metrics_by_horizon || [])
    .filter(r => r.scenario === scen && ARMS.includes(r.label));
  const buckets = [...new Set(rows.map(r => r.bucket))]
    .sort((a,b) => bucketOrder(a) - bucketOrder(b));
  const traces = ARMS.filter(a => rows.some(r => r.label === a)).map(a => ({{
    x: buckets, y: buckets.map(b => {{
      const r = rows.find(r => r.label === a && r.bucket === b);
      return r ? r[metric] : null;
    }}),
    name: a, mode: "lines+markers", line: {{color: ARM_COLORS[a]}}
  }}));
  const layout = {{title: metric + " by horizon bucket &mdash; " + scen,
    xaxis: {{title: "bucket (cumulative)"}}, yaxis: {{title: metric}},
    height: 420, margin: {{l: 60, r: 20, t: 50, b: 40}}}};
  if (metric === "coverage") layout.shapes = [{{type: "line", x0: 0, x1: 1, xref: "paper",
    y0: NOMINAL, y1: NOMINAL, line: {{color: "crimson", dash: "dash"}}}}];
  Plotly.newPlot("horizonPlot", traces, layout, {{responsive: true}});
}}

// ---------- 4. calibration ----------
function renderCalib() {{
  const rows = (window.DATA.summary || []).filter(r => ARMS.includes(r.label));
  // grouped bars: coverage per arm, grouped by scenario
  const traces = ARMS.filter(a => rows.some(r => r.label === a)).map(a => ({{
    x: SCENS.filter(s => rows.some(r => r.label === a && r.scenario === s)),
    y: SCENS.filter(s => rows.some(r => r.label === a && r.scenario === s))
         .map(s => rows.find(r => r.label === a && r.scenario === s).coverage),
    name: a, type: "bar", marker: {{color: ARM_COLORS[a]}}, legendgroup: a
  }}));
  Plotly.newPlot("calibPlot", traces, {{
    title: "Empirical coverage of the 80% band vs. nominal",
    barmode: "group", yaxis: {{title: "coverage", range: [0, 1.05]}},
    shapes: [{{type: "line", x0: 0, x1: 1, xref: "paper", y0: NOMINAL, y1: NOMINAL,
              line: {{color: "crimson", dash: "dash"}}}}],
    height: 400, margin: {{l: 50, r: 20, t: 50, b: 40}}
  }}, {{responsive: true}});
  // table: |coverage - nominal| per arm x scenario
  let html = "<table><tr><th>arm</th>";
  SCENS.forEach(s => html += "<th>" + s + "</th>");
  html += "</tr>";
  ARMS.filter(a => rows.some(r => r.label === a)).forEach(a => {{
    html += "<tr><td style='color:" + ARM_COLORS[a] + "'>" + a + "</td>";
    SCENS.forEach(s => {{
      const r = rows.find(r => r.label === a && r.scenario === s);
      if (r) {{
        const dev = Math.abs(r.coverage - NOMINAL);
        html += "<td>" + fmt(r.coverage) + " (<span style='color:" +
          (dev < 0.05 ? "#2ca02c" : dev < 0.1 ? "#e6a817" : "#d62728") + "'>&Delta;" +
          fmt(dev) + "</span>)</td>";
      }} else html += "<td>&mdash;</td>";
    }});
    html += "</tr>";
  }});
  $("calibTable").innerHTML = html + "</table>";
}}

// ---------- 5. explainability ----------
function renderWS() {{
  const scen = $("wsScen").value, val = $("wsVal").value;
  const tables = window.DATA.windowshap_summary || {{}};
  const rows = tables[scen] || [];
  const windows = [...new Set(rows.map(r => r.window))].sort((a,b) => a-b);
  const pipes = ["raw", "decomp"];
  const traces = pipes.map(p => ({{
    x: windows, y: windows.map(w => {{
      const r = rows.find(r => r.pipeline === p && r.window === w);
      return r ? r[val] : null;
    }}),
    name: p, type: "bar", marker: {{color: p === "raw" ? "#1f77b4" : "#9467bd"}}
  }}));
  Plotly.newPlot("wsPlot", traces, {{
    title: val + " per context window &mdash; " + scen,
    barmode: "group", xaxis: {{title: "window (0=oldest, 7=newest)"}},
    yaxis: {{title: val}}, height: 380, margin: {{l: 60, r: 20, t: 50, b: 40}}
  }}, {{responsive: true}});

  // contrast bars
  const ea = (window.DATA.event_attribution || []).filter(r => r.scenario === scen);
  const cats = [...new Set(ea.map(r => r.pipeline + "/" + r.value_function))];
  const traces2 = cats.map(c => {{
    const r = ea.find(r => r.pipeline + "/" + r.value_function === c);
    return {{x: [c], y: [r ? r.contrast_mean : null], type: "bar",
             marker: {{color: c.startsWith("raw") ? "#1f77b4" : "#9467bd"}},
             text: [r ? "x" + fmt(r.contrast_mean, 2) : ""], textposition: "outside"}};
  }});
  Plotly.newPlot("contrastPlot", traces2, {{
    title: "Event-attribution contrast (mean|phi| event / clean) &mdash; " + scen,
    yaxis: {{title: "contrast", rangemode: "tozero"}},
    height: 360, margin: {{l: 50, r: 20, t: 50, b: 60}}, showlegend: false
  }}, {{responsive: true}});

  // table with decomp/raw ratio
  let html = "<table><tr><th>value function</th><th>raw contrast</th>" +
             "<th>decomp contrast</th><th>decomp/raw</th></tr>";
  ["phi_median", "phi_band"].forEach(vf => {{
    const rr = ea.find(r => r.pipeline === "raw" && r.value_function === vf);
    const dd = ea.find(r => r.pipeline === "decomp" && r.value_function === vf);
    if (rr && dd) {{
      html += "<tr><td>" + vf + "</td><td>" + fmt(rr.contrast_mean,2) + "</td><td>" +
        fmt(dd.contrast_mean,2) + "</td><td><b>x" + fmt(dd.contrast_mean/rr.contrast_mean,2) +
        "</b></td></tr>";
    }}
  }});
  $("contrastTable").innerHTML = html + "</table>";
}}

// ---------- 6. diagnosis ----------
function renderDiag() {{
  const fm = window.DATA.flat_median || null;
  if (!fm || !fm.scenarios) {{ $("diagCards").innerHTML = "<span class='note'>no data</span>"; return; }}
  const wrap = $("diagCards"); wrap.innerHTML = "";
  Object.entries(fm.scenarios).forEach(([scen, s]) => {{
    const c = document.createElement("div"); c.className = "card"; c.style.minWidth = "220px";
    c.innerHTML = '<div class="lbl">' + scen + "</div>" +
      '<div class="big">' + Math.round(s.flat_median_share*100) + '%</div>' +
      '<div class="lbl">flat median (std&lt;2.5)</div>' +
      '<div style="margin-top:6px;font-size:12px">' +
      'level-follow r=' + fmt(s.corr_ctx_end_vs_fc_start_level) + "<br>" +
      'trend-follow r=' + fmt(s.corr_ctx_slope_vs_fc_slope) + "<br>" +
      'band width ' + fmt(s.band_width_median_bpm,1) + " bpm<br>" +
      'cov&ge;80%: ' + Math.round(s.band_covers_ge_80pct_share*100) + "%</div>";
    wrap.appendChild(c);
  }});
  // bar: flat share + band covers
  const scens = Object.keys(fm.scenarios);
  Plotly.newPlot("diagPlot", [
    {{x: scens, y: scens.map(s => fm.scenarios[s].flat_median_share*100),
      name: "flat median %", type: "bar", marker: {{color: "#e07a5f"}}}},
    {{x: scens, y: scens.map(s => fm.scenarios[s].band_covers_ge_80pct_share*100),
      name: "band covers >=80% %", type: "bar", marker: {{color: "#81b29a"}}}}
  ], {{title: "Flat-median share vs. band absorption (chronos2)",
       barmode: "group", yaxis: {{title: "%", range: [0, 105]}},
       height: 360, margin: {{l: 50, r: 20, t: 50, b: 40}}}}, {{responsive: true}});
}}

// ---------- footer ----------
(function() {{
  const m = window.DATA.manifests || {{}};
  let html = "<table><tr><th>label</th><th>model</th><th>created</th><th>dgp_hash</th>" +
             "<th>context</th><th>horizon</th><th>n_series</th></tr>";
  Object.entries(m).forEach(([k,v]) => {{
    html += "<tr><td>" + esc(k) + "</td><td>" + esc(v.model || "&mdash;") + "</td><td>" +
      esc((v.created || "").slice(0,10)) + "</td><td><code>" + esc(v.dgp_hash || "&mdash;") +
      "</code></td><td>" + esc(v.context ?? "&mdash;") + "</td><td>" + esc(v.horizon ?? "&mdash;") +
      "</td><td>" + esc(JSON.stringify(v.n_series ?? "&mdash;")) + "</td></tr>";
  }});
  $("footer").innerHTML = html + "</table>";
}})();

// ---------- init ----------
fillSelect("distScen", SCENS);
fillSelect("horScen", SCENS);
fillSelect("wsScen", Object.keys((window.DATA.windowshap_summary) || {{}}).length
           ? Object.keys(window.DATA.windowshap_summary) : ["residual", "trend"]);
armCheckboxes("distArms", "renderDist()");
renderOverview();
renderDist();
renderHorizon();
renderCalib();
renderWS();
renderDiag();
</script>
</body>
</html>
"""


def build_html(payload: dict) -> str:
    # Unescape the doubled braces of the JS/CSS blocks FIRST, then substitute
    # the placeholders: the payload JSON legitimately contains "}}" sequences
    # (nested objects) that must not be collapsed.
    html = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    html = (html
            .replace("@@ARMS@@", json.dumps(list(ARMS)))
            .replace("@@ARM_COLORS@@", json.dumps(ARM_COLORS))
            .replace("@@SCENS@@", json.dumps(list(SCENARIOS)))
            .replace("@@SCEN_COLORS@@", json.dumps(SCEN_COLORS)))
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    data = data.replace("</", "<\\/")
    return html.replace("@@DATA@@", data)


def main() -> None:
    ap = argparse.ArgumentParser(description="Metrics dashboard (all results in one HTML)")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--data-dir", default="data/synthetic")
    ap.add_argument("--out", default="metrics_dashboard.html")
    args = ap.parse_args()

    results_dir, data_dir = Path(args.results_dir), Path(args.data_dir)
    payload = collect(results_dir, data_dir)

    print("[load] available:", payload["available"])
    if payload["missing"]:
        print("[skip] missing:", ", ".join(payload["missing"]))

    out = Path(args.out)
    out.write_text(build_html(payload), encoding="utf-8")
    size_kb = out.stat().st_size / 1024
    print(f"[write] {out} ({size_kb:.0f} KB)")
    print("[done] open in a browser (needs internet for plotly CDN)")


if __name__ == "__main__":
    main()

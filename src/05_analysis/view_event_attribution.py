"""
Interactive event-attribution viewer: XAI weights against the true data.

Per scenario and series, ONE standalone HTML file (plotly CDN) shows, aligned
on the same minute axis:
- top panel: the heart-rate series (context + horizon), event spans shaded,
  window boundaries as vertical lines, context/horizon split marked
- bottom panel: the WindowSHAP attribution |phi| per context window as bars,
  raw vs. decomp pipeline side by side, colored by whether the window
  contains a ground-truth event (event vs. clean)

So for every series you can SEE whether the model's attention (Shapley
weight) lands on the minutes where the DGP actually placed the anomaly.

Usage:
    python src/05_analysis/view_event_attribution.py
    python src/05_analysis/view_event_attribution.py --out event_attribution_viewer.html

Reads:  results/windowshap_{scenario}.parquet
        data/synthetic/synthetic_hr_mmash_{scenario}.parquet
        data/synthetic/metadata_{scenario}.jsonl   (ground-truth events)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SCENARIOS = ("residual", "trend")
WINDOW = 64  # minutes per context window


def event_window_set(events: list[dict], n_windows: int) -> set[int]:
    hits: set[int] = set()
    for e in events:
        if e.get("start") is None:
            continue
        end = e.get("end", e["start"])
        for w in range(n_windows):
            if e["start"] < (w + 1) * WINDOW and end > w * WINDOW:
                hits.add(w)
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Interactive viewer: Shapley weights vs. ground-truth events per series")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--data-dir", default="data/synthetic")
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--max-series", type=int, default=150)
    ap.add_argument("--out", default="event_attribution_viewer.html")
    args = ap.parse_args()

    results_dir, data_dir = Path(args.results_dir), Path(args.data_dir)
    n_windows = args.context // WINDOW

    data: dict = {"scenarios": [], "series": {}}
    for scen in args.scenarios:
        shap_path = results_dir / f"windowshap_{scen}.parquet"
        series_path = data_dir / f"synthetic_hr_mmash_{scen}.parquet"
        meta_path = data_dir / f"metadata_{scen}.jsonl"
        if not (shap_path.exists() and series_path.exists() and meta_path.exists()):
            print(f"[skip] {scen}: missing files")
            continue
        shap = pd.read_parquet(shap_path)
        df = pd.read_parquet(series_path)
        meta = {m["series_id"]: m for m in
                (json.loads(l) for l in meta_path.read_text().splitlines() if l.strip())}

        sids = sorted(shap["series_id"].unique())[:args.max_series]
        payload = []
        for sid in sids:
            g = df[df["series_id"] == sid].sort_values("t")
            events = meta.get(sid, {}).get("events", [])
            s = shap[shap["series_id"] == sid]
            rows = []
            for _, r in s.iterrows():
                rows.append({
                    "pipeline": str(r["pipeline"]),
                    "window": int(r["window"]),
                    "start": int(r["window_start"]),
                    "end": int(r["window_end"]),
                    "phi_median": round(float(r["phi_median"]), 3),
                    "phi_band": round(float(r["phi_band"]), 3),
                })
            hits = event_window_set(events, n_windows)
            for row in rows:
                row["has_event"] = row["window"] in hits
            payload.append({
                "series_id": int(sid),
                "t": g["t"].tolist(),
                "hr": [round(v, 2) for v in g["hr"]],
                "events": events,
                "windows": rows,
            })
            print(f"[{scen}] series {sid}: {len(events)} events, "
                  f"{len(hits)}/{n_windows} event windows")
        data["scenarios"].append(scen)
        data["series"][scen] = payload

    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data))
    Path(args.out).write_text(html)
    print(f"\n-> {args.out} created. Open it in a browser to explore.")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Event Attribution Viewer: where does the model look?</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body { font-family: system-ui, sans-serif; margin: 16px; color: #1a1a2e; }
  .controls { display: flex; flex-wrap: wrap; gap: 14px; align-items: center;
               margin-bottom: 8px; }
  label { font-size: 13px; display: flex; align-items: center; gap: 4px; }
  select { padding: 3px 6px; }
  #plot { width: 100%; height: 640px; }
  .card { background: #f7f8fa; border: 1px solid #e3e5ea; border-radius: 8px;
           padding: 8px 12px; font-size: 12.5px; margin: 8px 0; }
</style>
</head>
<body>
<h3>Event Attribution: Shapley weights vs. ground-truth anomalies</h3>
<div class="controls">
  <label>Scenario <select id="scenario" onchange="fillSeries()"></select></label>
  <label>Series <select id="series" onchange="render()"></select></label>
  <label>Value function
    <select id="vfun" onchange="render()">
      <option value="phi_median">forecast median</option>
      <option value="phi_band">band width</option>
    </select>
  </label>
</div>
<div id="plot"></div>
<div class="card" id="summary"></div>

<script>
window.DATA = __DATA__;

const $ = (id) => document.getElementById(id);
const EVT_COLOR = "#e07a5f", CLEAN_COLOR = "#81b29a";
const RAW = "#1f77b4", DEC = "#d62728";
const SEV_COLOR = {subtle: "rgba(255,193,7,0.18)",
                    moderate: "rgba(255,152,0,0.22)",
                    severe: "rgba(244,67,54,0.25)"};

function init() {
  const sc = $("scenario");
  for (const s of DATA.scenarios) {
    const o = document.createElement("option");
    o.value = s; o.textContent = s; sc.appendChild(o);
  }
  fillSeries();
}

function fillSeries() {
  const s = $("scenario").value;
  const sel = $("series");
  sel.innerHTML = "";
  for (const rec of DATA.series[s]) {
    const o = document.createElement("option");
    o.value = rec.series_id; o.textContent = "Series " + rec.series_id;
    sel.appendChild(o);
  }
  render();
}

function current() {
  return DATA.series[$("scenario").value]
    .find((r) => r.series_id === parseInt($("series").value));
}

function render() {
  const rec = current();
  const vf = $("vfun").value;
  const traces = [];

  // event spans on the data axis
  for (const ev of rec.events) {
    traces.push({
      type: "rect", xref: "x", yref: "paper",
      x0: ev.start, x1: Math.min(ev.end, rec.t[rec.t.length - 1]),
      y0: 0, y1: 1,
      fillcolor: SEV_COLOR[ev.severity] || "rgba(200,200,200,0.2)",
      line: {width: 0}, layer: "below",
      showlegend: false,
    });
  }

  traces.push({
    x: rec.t, y: rec.hr, mode: "lines", name: "heart rate",
    line: {color: "#333", width: 1.4},
  });

  // window boundaries
  const wstarts = rec.windows.filter((w) => w.pipeline === "raw")
    .map((w) => w.start);
  for (const ws of wstarts) {
    traces.push({
      x: [ws, ws], y: [0, 1], yref: "paper",
      mode: "lines", line: {color: "#bbb", width: 1, dash: "dot"},
      showlegend: false, hoverinfo: "skip",
    });
  }
  traces.push({
    x: [512, 512], y: [0, 1], yref: "paper",
    mode: "lines", name: "context | horizon",
    line: {color: "#555", width: 1.5, dash: "dash"},
  });

  // attribution bars, offset so raw/decomp sit side by side
  for (const pipe of ["raw", "decomp"]) {
    const ws = rec.windows.filter((w) => w.pipeline === pipe);
    const off = pipe === "raw" ? -14 : 14;
    traces.push({
      type: "bar", x: ws.map((w) => (w.start + w.end) / 2 + off),
      y: ws.map((w) => Math.abs(w[vf])),
      width: 28,
      name: pipe + (pipe === "raw" ? "" : ""),
      marker: {color: ws.map((w) => w.has_event ? EVT_COLOR : CLEAN_COLOR)},
      xaxis: "x", yaxis: "y2",
      text: ws.map((w) =>
        "win " + w.window + (w.has_event ? " [event]" : " [clean]") +
        "  phi=" + w[vf]),
      hoverinfo: "text",
    });
  }

  const layout = {
    grid: {rows: 2, columns: 1, pattern: "independent"},
    xaxis: {title: "minute", range: [0, Math.max(...rec.t)]},
    yaxis: {title: "heart rate (bpm)", domain: [0.52, 1]},
    yaxis2: {title: "|phi| (bpm)", domain: [0, 0.42],
             rangemode: "tozero"},
    barmode: "group",
    height: 640,
    margin: {l: 60, r: 30, t: 30, b: 45},
    legend: {orientation: "h", y: 1.08},
    shapes: [],
  };

  Plotly.react($("plot"), traces, layout, {displayModeBar: false});

  const ev = rec.windows.filter((w) => w.pipeline === "raw" && w.has_event).length;
  const evPhi = rec.windows.filter((w) => w.pipeline === "raw")
      .reduce((a, w) => a + (w.has_event ? Math.abs(w[vf]) : 0), 0) /
    Math.max(rec.windows.filter((w) => w.pipeline === "raw" && w.has_event).length, 1);
  const clPhi = rec.windows.filter((w) => w.pipeline === "raw")
      .reduce((a, w) => a + (!w.has_event ? Math.abs(w[vf]) : 0), 0) /
    Math.max(rec.windows.filter((w) => w.pipeline === "raw" && !w.has_event).length, 1);
  $("summary").innerHTML =
    `<b>Series ${rec.series_id}</b> (${$("scenario").value}): ` +
    `${rec.events.length} ground-truth event(s), ` +
    `${ev}/${rec.windows.filter(w=>w.pipeline==="raw").length} context windows contain an event. ` +
    `Mean |phi| (raw, ${vf}): event windows ${evPhi.toFixed(2)} vs clean ` +
    `windows ${clPhi.toFixed(2)} ` +
    `(contrast x${clPhi > 0 ? (evPhi / clPhi).toFixed(1) : "-"}). ` +
    `Bars: <span style="color:${EVT_COLOR}">&#9632;</span> window with event, ` +
    `<span style="color:${CLEAN_COLOR}">&#9632;</span> clean window. ` +
    `Top: HR with event spans shaded (severity colors) and window ` +
    `boundaries (dotted).`;
}

init();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()

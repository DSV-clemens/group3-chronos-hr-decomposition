# Metrics map — paper number → repository file → producing script

Every reported number is traceable to a committed file. Aggregation notes:
accuracy uses the **median across series**; uncertainty metrics are **pooled
across all forecast observations** (both CSVs state this in their columns).

| Paper item | Number(s) | File | Produced by |
|---|---|---|---|
| **Table I** — MAE / WQL per method & scenario (median over 150 series) | MAE, WQL | `results/summary_scenario.csv` | `src/04_forecast/run_forecast*.py` (A/B-safe per label) |
| Table I — per-series values behind the medians | MAE, WQL, coverage per series | `results/metrics_per_series.csv` | forecast arms + `src/05_analysis/collect_metrics.py` |
| **Table II** — PICP, MPIW, Interval Score, pooled WQL | coverage, mpiw, interval_score, wql | `results/uncertainty/uncertainty_overall.csv` | `src/05_analysis/analyze_uncertainty.py` |
| Table II — ΔIS with 95% bootstrap CIs | mean_paired_difference, ci95_lower/upper (metric = interval_score) | `results/uncertainty/uncertainty_paired_bootstrap.csv` | `src/05_analysis/analyze_uncertainty.py` |
| Section IV-B — interval scores per 64-min segment | interval_score by horizon_block | `results/uncertainty/uncertainty_by_horizon.csv` | `src/05_analysis/analyze_uncertainty.py` |
| **Table III** — overall & position-controlled contrast, Δ (95% CI) | overall_raw/decomp, pc_raw/decomp, pc_diff_* | `results/attribution_contrast.csv` | `src/05_analysis/attribution_contrast.py` |
| Table III — event/clean means behind the contrast | mean_abs_event/clean per pipeline | `results/event_attribution.csv` | `src/05_analysis/plot_event_attribution.py` |
| **Figure 1** — per-window mean absolute attribution | phi_median by window | `plots/windowshap_event_attribution.png` | `src/05_analysis/plot_event_attribution.py` |
| Section IV-A — flat-median diagnosis of Chronos-2 | per-series flatness stats | `results/flat_median_diagnosis.csv` | `src/05_analysis/diagnose_flat_median.py` |
| Horizon-segment accuracy metrics | MAE etc. by horizon | `results/metrics_by_horizon.csv` | `src/05_analysis/metrics_by_horizon.py` |

## Provenance chain

`results/manifest_{chronos2,decomp_acf,dlinear,naive,windowshap}.json` record
model, quantiles, context/horizon, seeds, DGP hash (`82057932bd10`), parameter
SHA-256, generator version (`hr-dgp-mmash-1.3`) and runtime for every run.
`data/synthetic/dgp_config_mmash.json` and `data/params/dgp_params_mmash.json`
pin the data-generating process; `data/synthetic/metadata_{scenario}.jsonl`
carry the per-series ground truth (components + injected anomaly windows)
used for the event/clean labels in the explainability evaluation.

## Uncertainty figure ↔ data

`plots/uq_*.png` (coverage, width, interval score, paired bootstrap CIs,
horizon effect) are rendered by `src/05_analysis/plot_uncertainty.py` from the
four CSVs in `results/uncertainty/` listed above.
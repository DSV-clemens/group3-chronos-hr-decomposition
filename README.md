# grpup3-chronos-hr-decomposition

# Pipeline — Runbook

> Get MMASH-raw data before: https://physionet.org/content/mmash/1.0.0/
> → `data/raw/mmash/`

## Setup

```bash
git clone <NEUES-REPO-URL> && cd chronos-hr-decomposition
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# ── Step 01: Daten aufbereiten ──
python src/01_data_prep/preprocess_rr.py --input data/raw/mmash --out-dir data/processed

# ── Step 02: DGP-Parameter kalibrieren ──
python src/02_calibration/extract_dgp_params.py --input data/processed --out data/params/dgp_params_mmash.json

# ── Step 03: Synthetische Daten generieren ──
python src/03_synthesis/generate_hr_mmash.py \
    --params data/params/dgp_params_mmash.json \
    --n-series 150 --seed 67 --out-dir data/synthetic

# ── Step 03b: DLinear-Trainingskorpus (held-out) ──
python src/03_synthesis/generate_dlinear_corpus.py \
    --params data/params/dgp_params_mmash.json \
    --n-series 1500 --seed 6677 --out-dir data/dlinear_corpus

# ── Step 04: Forecasts (alle Arme) ──
python src/04_forecast/run_forecast.py --input-dir data/synthetic --out-dir results
python src/04_forecast/run_forecast.py --input-dir data/synthetic --out-dir results --naive
python src/04_forecast/run_forecast_dlinear.py --corpus-dir data/dlinear_corpus --input-dir data/synthetic --out-dir results
python src/04_forecast/run_forecast_decomp.py --input-dir data/synthetic --out-dir results

# ── Step 05a: Plots + Horizon-Metriken ──
python src/05_analysis/plot_forecast_results.py --input-dir results --data-dir data/synthetic --out-dir plots
python src/05_analysis/metrics_by_horizon.py --results-dir results

# ── Step 05b: Flat-Median-Diagnose (chronos2) ──
python src/05_analysis/diagnose_flat_median.py --results-dir results --data-dir data/synthetic --label chronos2

# ── Step 06: WindowSHAP (GPU empfohlen!) ──
python src/06_xai/window_shap.py --input-dir data/synthetic --out-dir results

# ── Step 05c: Event Attribution (braucht Step 06!) ──
python src/05_analysis/plot_event_attribution.py --results-dir results --data-dir data/synthetic --out-dir plots

# ── Step 05d: Report ──
python src/05_analysis/collect_metrics.py --results-dir results --data-dir data/synthetic

# ── Step 05e: Metrics Dashboard ──
python src/05_analysis/view_metrics_dashboard.py \
    --results-dir results --data-dir data/synthetic --out metrics_dashboard.html

python src/05_analysis/view_hr_parquet.py --input-dir data/synthetic --out hr_viewer.html
python src/05_analysis/view_forecasts.py --forecast-dir results --data-dir data/synthetic --out forecast_viewer.html
python src/05_analysis/view_decomp.py --forecast-dir results --data-dir data/synthetic --out decomp_viewer.html
python src/05_analysis/view_windowshap.py --forecast-dir results --data-dir data/synthetic --out windowshap_viewer.html
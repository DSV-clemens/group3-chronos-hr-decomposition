# STL Decomposition for Explainable and Uncertainty-Aware Zero-Shot Heart-Rate Forecasting

Code and data accompanying our research project paper. We compare Chronos-2
zero-shot forecasting on raw heart-rate signals against an STL + Chronos-2
decomposition pipeline, on a synthetic benchmark calibrated to the MMASH
dataset — evaluating **accuracy** (MAE, WQL), **uncertainty** (PICP, MPIW,
Interval Score) and **explainability** (WindowSHAP window attributions with a
position-controlled contrast).

**Authors:** Alexandra Kordali, Clemens Pretzler, Mikaela Rantza,
_Department of Computer and Systems Sciences, Stockholm University_ (Group 3).

> **AI usage:** This project was completed with AI support as permitted by
> [DSV's AI policy](https://www.su.se/english/divisions/department-of-computer-and-systems-sciences/education/during-your-studies/dsvs-ai-policy).
> The use of AI tools is declared in [`AI_USAGE.md`](AI_USAGE.md).

## Key findings (TL;DR)

- **Accuracy:** unaffected. Median MAE of the STL pipeline matches direct
  Chronos-2 within 0.3–1.7 bpm in all scenarios; no paired difference is
  significant.
- **Uncertainty:** degraded. STL lowered empirical 80% coverage in every
  scenario and increased the Interval Score significantly on clean signals
  (+6.54 [1.94, 11.27]).
- **Explainability:** the attribution contrast gain (3.8x → 6.4x) does not
  survive position control, it is a recency-weighting artifact, not improved
  anomaly recognition.

Conclusion: for zero-shot forecasting today, direct Chronos-2 on raw signals
remains the better default. See [docs/metrics_map.md](docs/metrics_map.md) for where every number lives in the repository.

## Repository structure

```
src/01_data_prep      # RR-interval cleaning (MMASH) → 1-min bpm series
src/02_calibration    # DGP parameter extraction from processed MMASH
src/03_synthesis      # synthetic benchmark (seed 67) + DLinear corpus (seed 6677)
src/04_forecast       # forecast arms: chronos2, naive, dlinear, decomp_acf
src/05_analysis       # metrics, uncertainty evaluation, plots, viewers
src/06_xai            # WindowSHAP (exact Shapley over 8 context windows)
data/params           # committed DGP calibration (SHA-256 traceable)
data/synthetic        # committed benchmark series + event metadata
results/              # committed forecasts, WindowSHAP outputs, metrics, manifests
plots/                # committed figures
```

## Reproducibility

- **Committed artifacts:** all benchmark data, all forecast parquets, all
  WindowSHAP outputs, trained DLinear weights and every metrics table are in
  the repo. Every number in the paper can be re-derived from them **on a
  laptop CPU, without any GPU and without regenerating data**.
- **Provenance:** every run writes a manifest (`results/manifest_*.json`)
  recording the DGP hash (`82057932bd10`), parameter SHA-256, seeds and
  runtime. Generator version: `hr-dgp-mmash-1.3`.
- **Seeds:** benchmark series use seed 67 (150 per scenario, paired design);
  the DLinear training corpus uses seed 6677 (1,500 series, strictly held out
  from the benchmark).
- Only Steps 01–03 (raw data → calibration → synthesis) require a one-time
  MMASH download; everything from Step 04 onward runs directly on committed
  data.

## Setup

```bash
git clone https://github.com/DSV-clemens/group3-chronos-hr-decomposition.git
cd group3-chronos-hr-decomposition
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# GPU support (recommended before running the full pipeline, see below):
# pip installs the CPU-only PyTorch wheel by default. To run the forecast
# arms and WindowSHAP on a CUDA GPU, install the CUDA build FIRST —
# see https://pytorch.org/get-started/locally/ for your CUDA version, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

> **Note on GPU usage:** `pip install -r requirements.txt` alone gives you a
> fully working **CPU** setup — enough to run all analysis steps on the
> committed results. The forecast arms (Step 04) and especially WindowSHAP
> (Step 06, 256 model calls per series and pipeline) benefit massively from a
> CUDA GPU. Chronos-2 picks the device automatically (`cuda` if available,
> else CPU), so installing the CUDA-enabled PyTorch wheel **before**
> `requirements.txt` is the only thing needed to enable it.

Raw data (Steps 01–03 only): download MMASH from PhysioNet →
<https://physionet.org/content/mmash/1.0.0/> → `data/raw/mmash/`.

## Pipeline

### Steps 01–03 — data preparation (one-time, needs MMASH download)

```bash
# 01 - RR cleaning and 1-minute aggregation
python src/01_data_prep/preprocess_rr.py --input data/raw/mmash --out-dir data/processed

# 02 - DGP calibration from processed data
python src/02_calibration/extract_dgp_params.py --input data/processed --out data/params/dgp_params_mmash.json

# 03 - synthetic benchmark (150 series/scenario, seed 67)
python src/03_synthesis/generate_hr_mmash.py \
    --params data/params/dgp_params_mmash.json \
    --n-series 150 --seed 67 --out-dir data/synthetic
```

### Step 03b — DLinear training corpus (optional, CPU-only, minutes)

Not committed to the repo (regenerable). Only needed for re-fitting the DLinear
weights (`--train`); forecasting works without it because the fitted weights
are committed under `results/dlinear_weights/`.

```bash
python src/03_synthesis/generate_dlinear_corpus.py \
    --params data/params/dgp_params_mmash.json \
    --n-series 1500 --seed 6677 --out-dir data/dlinear_corpus
```

### Step 04 — forecast arms (GPU recommended; committed results allow skipping)

```bash
python src/04_forecast/run_forecast.py --input-dir data/synthetic --out-dir results           # Chronos-2
python src/04_forecast/run_forecast.py --input-dir data/synthetic --out-dir results --naive  # persistence
python src/04_forecast/run_forecast_dlinear.py --corpus-dir data/dlinear_corpus \
    --input-dir data/synthetic --out-dir results                                             # loads committed weights
python src/04_forecast/run_forecast_decomp.py --input-dir data/synthetic --out-dir results   # STL + Chronos-2
```

Reference runtimes (recorded in the manifests): Chronos-2 ≈ 36 s, STL + Chronos-2 ≈ 49 s,
DLinear ≈ 106 s (GPU).

### Steps 05a–e — analysis, metrics and plots (CPU, runs on committed results)

```bash
python src/05_analysis/plot_forecast_results.py --input-dir results --data-dir data/synthetic --out-dir plots
python src/05_analysis/metrics_by_horizon.py --results-dir results
python src/05_analysis/analyze_uncertainty.py --results-dir results          # → results/uncertainty/*.csv
python src/05_analysis/plot_uncertainty.py --results-dir results --out-dir plots
python src/05_analysis/diagnose_flat_median.py --results-dir results --data-dir data/synthetic --label chronos2
python src/05_analysis/collect_metrics.py --results-dir results --data-dir data/synthetic
```

### Steps 06a–c — WindowSHAP explainability

`window_shap.py` requires 256 model calls per series, pipeline and scenario,
**GPU strongly recommended, expect hours**. Its outputs are committed, so the
analysis steps below also run directly on committed data.

```bash
# 06a - WindowSHAP (GPU!)
python src/06_xai/window_shap.py --input-dir data/synthetic --out-dir results

# 06b - event attribution (needs 06a) + position-controlled contrast
python src/05_analysis/plot_event_attribution.py --results-dir results --data-dir data/synthetic --out-dir plots
python src/05_analysis/attribution_contrast.py --results-dir results
```

### Interactive viewers (optional, CPU, seconds)

Self-contained HTML; regenerated from committed results. Frozen snapshots are
attached to the
[v1.0.0-paper release](https://github.com/DSV-clemens/group3-chronos-hr-decomposition/releases/tag/v1.0.0-paper).

```bash
python src/05_analysis/view_hr_parquet.py --input-dir data/synthetic --out hr_viewer.html
python src/05_analysis/view_forecasts.py --forecast-dir results --data-dir data/synthetic --out forecast_viewer.html
python src/05_analysis/view_decomp.py --forecast-dir results --data-dir data/synthetic --out decomp_viewer.html
python src/05_analysis/view_windowshap.py --forecast-dir results --data-dir data/synthetic --out windowshap_viewer.html
python src/05_analysis/view_metrics_dashboard.py --results-dir results --data-dir data/synthetic --out metrics_dashboard.html
```

## Data and model attribution

- MMASH dataset (Rossi et al., 2020), PhysioNet, separate license, not redistributed here.
- Chronos-2 (Ansari et al., 2025), pre-trained weights via `chronos-forecasting`.

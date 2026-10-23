"""
Phase 2: Extract DGP parameters from MMASH HR data (Phase-1 output).

Reads all hr_1min_*.csv in a folder (1 file = 1 person = 1 day,
columns: minute,bpm) and calibrates the ranges of the synthetic HR
generator from them. Result: dgp_params_mmash.json
(+ mmash_per_subject.csv as a transparency table).

Statistics per person:
- Median/std of heart rate -> baseline range
- Linear trend + circadian harmonic (1440 min) -> slope range
- Ultradian period via ACF peak (60-480 min) -> period range
- Harmonic amplitude at the ultradian period -> seasonal amplitude
- AR(1) phi and sigma on the cleaned residual -> inertia/noise
- Jump amplitudes (|minute diff| > 10) -> spike severities
- Sustained level shifts (>15 bpm, difference of 30-min means)
  -> level-shift severities

Usage:
    python extract_dgp_params.py --input data/processed --out data/params/dgp_params_mmash.json
""" 

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

EXTRACTOR_VERSION = "2.0"
CIRC = 1440.0  # circadian period [min]

# Fallback values (NOT calibrated from data; flagged as such in the provenance block)
DEFAULT_SPIKE_LEN = {"subtle": [10, 30], "moderate": [15, 45], "severe": [20, 60]}
DEFAULT_DRIFT_TOTAL = {"subtle": [4, 8], "moderate": [8, 16], "severe": [16, 30]}


# ------------------------------------------------------------------ building blocks

def _mad_sigma(x: np.ndarray) -> float:
    """Robust std via MAD."""
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def _harmonic_design(t: np.ndarray, period: float) -> np.ndarray:
    return np.column_stack([
        np.sin(2 * np.pi * t / period),
        np.cos(2 * np.pi * t / period),
    ])


def _fit(t: np.ndarray, y: np.ndarray, periods: list[float]) -> dict:
    """Regression: y ~ 1 + t + sum_p sin/cos(p), fitted jointly.
    Returns: slope, one amplitude per period (same order), residual."""
    cols = [np.ones_like(t, dtype=float), t.astype(float)]
    for p in periods:
        cols.append(_harmonic_design(t, p))
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    amps = [float(np.hypot(beta[2 + 2 * k], beta[3 + 2 * k])) for k in range(len(periods))]
    return {
        "slope_bpm_per_min": float(beta[1]),
        "amps": amps,
        "resid": resid,
    }


def _acf_first_peak(x: np.ndarray, lo: int = 60, hi: int = 480, min_r: float = 0.1,
                    min_prominence: float = 0.05):
    """First ACF peak (lag in minutes) in the range [lo, hi].

    v2: the peak must also rise at least `min_prominence` above the lowest ACF
    value at smaller lags. Without this, a slowly decaying ACF (persistent AR
    noise) returns a tiny wiggle right after `lo`, i.e. the period is an artifact
    of the search bound rather than a real ultradian rhythm."""
    x = x - x.mean()
    n = len(x)
    hi = min(hi, n // 2)
    if hi <= lo:
        return None
    acf = np.array([np.corrcoef(x[:-lag], x[lag:])[0, 1] for lag in range(1, hi)])
    for lag in range(lo - 1, len(acf) - 1):
        if acf[lag] > acf[lag - 1] and acf[lag] >= acf[lag + 1] and acf[lag] > min_r:
            if acf[lag] - float(acf[:lag].min()) >= min_prominence:
                return lag + 1
    return None


def _phi_sigma(resid: np.ndarray, robust: bool = True) -> tuple[float, float]:
    """AR(1) parameters on the residual: phi = lag-1 autocorrelation,
    sigma = scale * sqrt(1 - phi^2) (stationary AR(1) innovation).

    robust=True (v2 default): scale = MAD-sigma and phi is estimated on the residual
    winsorised at +-4 MAD-sigma. Reason: jumps and sustained level shifts in MMASH
    are modelled SEPARATELY as anomalies (spike_amp / shift_amp below). Estimating
    the background noise with the classical std counts them a second time, so the
    'none' scenario would already contain anomaly-sized variance.
    robust=False reproduces v1 (classical std, raw residual)."""
    r = resid - np.median(resid)
    if robust:
        scale = _mad_sigma(r)
        r = np.clip(r, -4 * scale, 4 * scale)
    else:
        r = r - r.mean()
        scale = float(r.std())
    phi = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    sigma = float(scale * np.sqrt(max(1e-9, 1.0 - phi ** 2)))
    return phi, sigma


def _sustained_shifts(y: np.ndarray, contiguous: np.ndarray | None = None,
                      win: int = 30, jump: float = 15.0,
                      min_shift: float = 8.0) -> list[float]:
    """Level shifts: |minute diff| > jump AND the 30-min means before/after
    differ by > min_shift (keeps single sensor spikes out).
    `contiguous[i]` is True if samples i and i+1 are adjacent minutes."""
    d = np.diff(y)
    mask = np.abs(d) > jump
    if contiguous is not None:
        mask &= contiguous
    idx = np.where(mask)[0]
    shifts = []
    for i in idx:
        a = y[max(0, i - win):i]
        b = y[i + 1: i + 1 + win]
        if len(a) >= win // 2 and len(b) >= win // 2:
            s = float(b.mean() - a.mean())
            if abs(s) > min_shift:
                shifts.append(abs(s))
    return shifts


def _pct(x: list[float] | np.ndarray, q: float, floor: float) -> float:
    """Percentile with a lower bound (prevents empty/nonsensical bands)."""
    if len(x) == 0:
        return floor
    return max(floor, float(np.percentile(x, q)))


def analyse_file(path: Path, robust: bool = True) -> dict:
    df = pd.read_csv(path)
    df.columns = ["minute", "bpm"]
    n_raw = len(df)
    df = df.dropna(subset=["bpm"]).sort_values("minute").reset_index(drop=True)
    t = df["minute"].to_numpy(dtype=float)
    y = df["bpm"].to_numpy(dtype=float)
    contiguous = np.diff(t) == 1

    # (a) Location/dispersion
    median, std = float(np.median(y)), float(y.std())

    # (b) Trend + circadian
    fit_circ = _fit(t, y, [CIRC])

    # (c) Ultradian period in the circadian-cleaned residual
    per = _acf_first_peak(fit_circ["resid"])

    # (d) JOINT fit trend + circadian + ultradian -> final noise residual
    if per is not None:
        fit_full = _fit(t, y, [CIRC, float(per)])
        resid = fit_full["resid"]
        circ_amp, ultra_amp, ultra_period = fit_full["amps"][0], fit_full["amps"][1], per
        slope = fit_full["slope_bpm_per_min"]
    else:
        resid = fit_circ["resid"]
        circ_amp, ultra_amp, ultra_period = fit_circ["amps"][0], None, None
        slope = fit_circ["slope_bpm_per_min"]

    # (e) AR(1) on the cleaned residual (robust = default; classical kept for comparison)
    phi, sigma = _phi_sigma(resid, robust=robust)
    phi_cls, sigma_cls = _phi_sigma(resid, robust=False)

    # (f) Jumps & level shifts (contiguous minutes only)
    d = np.diff(y)[contiguous]
    jumps = np.abs(d[np.abs(d) > 10])
    shifts = _sustained_shifts(y, contiguous)

    return {
        "file": path.name,
        "n": len(df),
        "n_missing": n_raw - len(df),
        "median_hr": median,
        "std_hr": std,
        "slope_bpm_per_min": slope,
        "circadian_amp": circ_amp,
        "ultradian_period_min": ultra_period,
        "ultradian_amp": ultra_amp,
        "ar_phi": phi,
        "ar_sigma": sigma,
        "noise_std": float(sigma / np.sqrt(max(1e-9, 1.0 - phi ** 2))),
        "ar_phi_classical": phi_cls,
        "ar_sigma_classical": sigma_cls,
        "resid_std": float(resid.std()),
        "resid_mad_sigma": _mad_sigma(resid),
        "n_jumps_gt10": int(len(jumps)),
        "jump_p50/p90/p99": [float(np.percentile(jumps, q)) for q in (50, 90, 99)] if len(jumps) else [],
        "n_sustained_shifts": len(shifts),
        "_shifts": shifts,
    }


# ------------------------------------------------------------ aggregation

def aggregate(stats: list[dict], file_hashes: dict, robust: bool = True) -> dict:
    medians = np.array([s["median_hr"] for s in stats])
    slopes = np.array([s["slope_bpm_per_min"] for s in stats])
    phis = np.array([s["ar_phi"] for s in stats])
    sigmas = np.array([s["ar_sigma"] for s in stats])
    resid_stds = np.array([s["noise_std"] for s in stats])
    circ_amps = np.array([s["circadian_amp"] for s in stats])
    periods = [s["ultradian_period_min"] for s in stats if s["ultradian_period_min"] is not None]
    amps = [s["ultradian_amp"] for s in stats if s["ultradian_amp"] is not None]

    # Jump bands from the per-person percentiles (mean across persons)
    j50 = [s["jump_p50/p90/p99"][0] for s in stats if s["jump_p50/p90/p99"]]
    j90 = [s["jump_p50/p90/p99"][1] for s in stats if s["jump_p50/p90/p99"]]
    j99 = [s["jump_p50/p90/p99"][2] for s in stats if s["jump_p50/p90/p99"]]

    mean_j50 = float(np.mean(j50)) if j50 else 12.0
    mean_j90 = float(np.mean(j90)) if j90 else 28.0
    mean_j99 = float(np.mean(j99)) if j99 else 60.0

    spike_amp = {
        "subtle":   [round(0.8 * mean_j50, 1), round(1.2 * mean_j50, 1)],
        "moderate": [round(mean_j90 * 0.8, 1), round(mean_j90 * 1.15, 1)],
        "severe":   [round(mean_j99 * 0.6, 1), round(mean_j99, 1)],
    }

    params = {
        "provenance": {
            "dataset": "MMASH (HR per minute, 1 day per person)",
            "extractor_version": EXTRACTOR_VERSION,
            "noise_scale": "robust (MAD, winsorised +-4)" if robust else "classical std (v1)",
            "n_subjects_with_ultradian_peak": len(periods),
            "n_subjects": len(stats),
            "files": file_hashes,
            "extracted_on": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "method": "Median/trend/circadian/ACF/joint harmonic fit/AR(1)/jump percentiles "
                      "per person, ranges = min/max across persons",
            "calibrated_from_data": {
                "baseline_range": True, "slope_range": True,
                "trend.circadian_amp_range": True,
                "seasonal.period_range": len(periods) >= 2,
                "seasonal.amp_range": len(amps) >= 2,
                "ar": True,
                "anomaly.spike_amp": bool(j50),
                "anomaly.shift_amp": None,  # set in attach_shift_amp
                "anomaly.spike_len": False,
                "anomaly.drift_total": False,
            },
        },
        "hr_bounds": [40, 200],
        "baseline_range": [round(float(medians.min()), 1), round(float(medians.max()), 1)],
        "slope_range": [
            round(float(np.clip(np.percentile(slopes, 10), -0.03, 0.0)), 4),
            round(float(np.clip(np.percentile(slopes, 90), 0.0, 0.05)), 4),
        ],
        "trend": {
            "circadian_amp_range": [round(float(circ_amps.min()), 1), round(float(circ_amps.max()), 1)],
            "note": "Amplitude of the 1440-min harmonic (joint fit). Over a 768-min window this "
                    "wave appears as smooth curvature and belongs to the TREND channel, not to "
                    "the residual. Not consumed by generator v1.",
        },
        "seasonal": {
            "period_range": [int(min(periods)), int(max(periods))] if len(periods) >= 2 else [55, 90],
            "amp_range": [round(float(np.min(amps)), 1), round(float(np.max(amps)), 1)] if len(amps) >= 2 else [2.0, 8.0],
            "note": "Ultradian component; the circadian wave (1440 min) does not fit the 768-min "
                    "window and is represented by the trend channel.",
        },
        "ar": {
            "phi_range": [round(max(0.5, float(phis.min()) - 0.05), 3),
                          round(min(0.97, float(phis.max())), 3)],
            "sigma_range": [round(0.8 * float(sigmas.min()), 2), round(1.2 * float(sigmas.max()), 2)],
            "noise_std_range": [round(float(resid_stds.min()), 2), round(float(resid_stds.max()), 2)],
            "note": "phi_range and sigma_range are marginal ranges. To stay inside the observed "
                    "stationary variance, sample (phi, sigma) jointly from "
                    "subject_level.ar_phi_sigma_pairs, or cap sigma/sqrt(1-phi^2) at "
                    "noise_std_range[1].",
        },
        "anomaly": {
            "spike_amp": spike_amp,
            "spike_len": DEFAULT_SPIKE_LEN,
            "shift_amp": None,  # set below
            "drift_total": DEFAULT_DRIFT_TOTAL,
        },
        "subject_level": {
            "median_hr_per_subject": [round(float(m), 1) for m in medians],
            "ar_phi_per_subject": [round(float(p), 3) for p in phis],
            "ar_sigma_per_subject": [round(float(s), 2) for s in sigmas],
            "ar_phi_sigma_pairs": [[round(float(p), 3), round(float(s), 2)] for p, s in zip(phis, sigmas)],
            "noise_std_per_subject": [round(float(s), 2) for s in resid_stds],
            "circadian_amp_per_subject": [round(float(a), 1) for a in circ_amps],
        },
        "notes": [
            "Sensor artifacts (single spikes, dropouts, interpolated segments) are assumed "
            "removed in Phase 1; this script does not filter them itself.",
            "Ranges are min/max across persons -> preserves heterogeneity; with more "
            "persons, q10/q90 can be used instead.",
            "spike_len and drift_total are hard-coded defaults, not estimated from MMASH.",
        ],
    }
    return params


def attach_shift_amp(stats: list[dict], params: dict) -> None:
    """Level-shift bands from the sustained level shifts of all files."""
    all_shifts = [s for st in stats for s in st["_shifts"]]
    calibrated = len(all_shifts) >= 5
    if calibrated:
        s = np.array(all_shifts)
        params["anomaly"]["shift_amp"] = {
            "subtle":   [round(_pct(s, 40, 4), 1), round(_pct(s, 60, 6), 1)],
            "moderate": [round(_pct(s, 70, 8), 1), round(_pct(s, 85, 10), 1)],
            "severe":   [round(_pct(s, 92, 15), 1), round(_pct(s, 98, 20), 1)],
        }
    else:
        params["anomaly"]["shift_amp"] = {
            "subtle": [5, 10], "moderate": [10, 20], "severe": [20, 35],
        }
        params["notes"].append("Too few level shifts for data calibration -> fallback shift bands.")
    params["provenance"]["calibrated_from_data"]["anomaly.shift_amp"] = calibrated
    params["provenance"]["n_sustained_shifts_total"] = len(all_shifts)


# ----------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed", help="Folder with hr_1min_*.csv")
    ap.add_argument("--out", default="data/params/dgp_params_mmash.json")
    ap.add_argument("--noise-scale", choices=["robust", "classical"], default="robust",
                    help="AR(1) noise scale: robust MAD (v2 default) or classical std (v1). "
                         "Run both and report the classical one as a sensitivity analysis.")
    args = ap.parse_args()
    robust = args.noise_scale == "robust"
    print(f"[Version] extract_dgp_params v{EXTRACTOR_VERSION} "
          f"(branch: DGP v2.0 calibration — joint fit, robust MAD noise, gap-aware shifts)")

    paths = sorted(Path(args.input).glob("hr_1min_*.csv"))
    if not paths:
        raise SystemExit(f"No hr_1min_*.csv found in {args.input}.")

    stats, hashes = [], {}
    for p in paths:
        stats.append(analyse_file(p, robust=robust))
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
        s = stats[-1]
        print(f"[OK] {p.name}: median={s['median_hr']:.1f} phi={s['ar_phi']:.3f} "
              f"sigma={s['ar_sigma']:.2f} noise_std={s['noise_std']:.2f} "
              f"(classical resid_std={s['resid_std']:.2f}) "
              f"circ_amp={s['circadian_amp']:.1f} ultra={s['ultradian_period_min']}")

    params = aggregate(stats, hashes, robust=robust)
    attach_shift_amp(stats, params)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(params, indent=2, ensure_ascii=False))

    per_subject = pd.DataFrame([{k: v for k, v in s.items()
                                 if k not in ("jump_p50/p90/p99", "_shifts")} for s in stats])
    per_subject.to_csv(out.with_name("mmash_per_subject.csv"), index=False)

    print(f"\n{len(stats)} persons analyzed -> {out}")
    print(json.dumps({k: params[k] for k in ("baseline_range", "slope_range", "trend", "seasonal", "ar", "anomaly")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

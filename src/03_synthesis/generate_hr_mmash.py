"""
Phase 3: MMASH-calibrated synthetic HR generator.

Project: Decomposition, explainability & uncertainty of Chronos-2.

Differences from the first generator (hr-dgp-generator):
- All dynamic parameters come from dgp_params_mmash.json, which was derived
  from real MMASH HR data (1 file = 1 person/day) with extract_dgp_params.py
  -> external validity, easy to motivate in the report.
- Per series, a "subject profile" is drawn (baseline/phi/sigma as a JOINT
  tuple from the per-person estimates + jitter), because feedback and noise
  are correlated in real data (sluggish subjects = high sigma). Independent
  uniform draws from the ranges would create unphysiological combinations.
- Seasonal = ultradian ~1h component (period 61-82 min, from ACF).
  The circadian wave (1440 min) does not fit the 768-min window and is
  represented by the trend channel (slope_range).
- The 3 scenarios are kept:
    "residual": spike/level-shift/drift in the residual channel
                -> decomposition SHOULD help.
    "trend":    ONLY a drift anomaly in the trend channel (adversarial)
                -> decomposition should NOT help much.
    "none":     no anomaly (baseline reference).
- Scientific integrity: the config hash now covers the generator version +
  the complete parameter file (including its SHA-256) -> every generated
  series is traceable back to the calibration set. The same seed across all
  scenarios -> paired design: series i has identical base dynamics in every
  scenario; only the anomalies differ.

Usage:
    python generate_hr_mmash.py --params data/params/dgp_params_mmash.json --n-series 150
    python generate_hr_mmash.py --n-series 5 --out-dir data_demo   # without params file: embedded defaults
    python generate_hr_mmash.py --scenarios residual none         # only specific scenarios
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

GENERATOR_VERSION = "hr-dgp-mmash-1.3"  # 1.3: trend-scenario budget pre-check tested the residual budget (never exhausted there) instead of the trend-drift budget -> ghost events with amplitude ~0 after the budget was spent; now breaks like the residual scenario

# ----------------------------------------------------------------------------
# Embedded defaults: test run from 2026-10-10 on 6 MMASH persons.
# Automatically overridden by --params; generate with all 22 persons
# (then provenance.n_subjects matches as well).
# ----------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "provenance": {
        "dataset": "MMASH (HR per minute, 1 day per person)",
        "n_subjects": 6,
        "extracted_on": "2026-10-10T14:14:47+00:00",
        "method": "Median/trend/circadian/ACF/AR(1)/jump percentiles per person, ranges = min/max across persons",
    },
    "hr_bounds": [40, 200],
    "baseline_range": [63.9, 89.4],
    "slope_range": [-0.0054, 0.0351],
    "seasonal": {
        "period_range": [61, 82],
        "amp_range": [1.1, 3.3],
        "note": "Ultradian (~1 h) component; the circadian wave (1440 min) does not fit the 768-min window and is represented by the trend channel.",
    },
    "ar": {
        "phi_range": [0.768, 0.962],
        "sigma_range": [3.79, 10.47],
    },
    "anomaly": {
        "spike_amp": {"subtle": [11.5, 17.3], "moderate": [24.4, 35.1], "severe": [33.9, 56.4]},
        "spike_len": {"subtle": [10, 30], "moderate": [15, 45], "severe": [20, 60]},
        "shift_amp": {"subtle": [12.2, 17.9], "moderate": [21.1, 26.9], "severe": [29.9, 60.0]},
        "drift_total": {"subtle": [4, 8], "moderate": [8, 16], "severe": [16, 30]},
    },
    "subject_level": {
        "median_hr_per_subject": [86.2, 72.7, 89.4, 69.4, 73.8, 63.9],
        "ar_phi_per_subject": [0.846, 0.960, 0.953, 0.818, 0.840, 0.962],
        "ar_sigma_per_subject": [7.55, 4.73, 8.72, 6.18, 5.84, 7.72],
    },
}

Scenario = Literal["residual", "trend", "none"]

SEVERITIES = ("subtle", "moderate", "severe")


@dataclass(frozen=True)
class DGPConfig:
    """Only structure parameters; all dynamic parameters come from params."""
    length: int = 768          # points, 1-min grid (~12.8 h)
    circadian: bool = True     # add circadian curvature to the trend (if calibrated)
    context_length: int = 512  # model context; anomalies must land here
    horizon: int = 256          # forecast horizon (768 - 512), documentation only
    n_anomalies_range: tuple = (1, 3)  # incl. upper bound


def load_params(path: str | None) -> tuple[dict, str, str]:
    """Loads the calibration parameters. Returns: (params, source, sha256)."""
    if path and Path(path).exists():
        text = Path(path).read_text(encoding="utf-8")
        source = str(path)
    else:
        text = json.dumps(DEFAULT_PARAMS, sort_keys=True)
        source = "embedded defaults (6-person test run)"
        if path:
            print(f"[WARN] {path} not found -> using embedded defaults.")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return json.loads(text), source, sha


def effective_config_hash(cfg: DGPConfig, params: dict, params_sha: str) -> str:
    """Hash over generator version + structure + the COMPLETE calibration
    parameters. Every series is thereby traceable back to the calibration
    set (integrity)."""
    payload = json.dumps({
        "generator_version": GENERATOR_VERSION,
        "config": asdict(cfg),
        "params": params,
        "params_sha256": params_sha,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# ------------------------------------------------------------------ building blocks

def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    """AR(1) noise: x_t = phi * x_{t-1} + eps_t."""
    eps = rng.normal(0.0, sigma, size=n)
    out = np.empty(n)
    out[0] = eps[0]
    for t in range(1, n):
        out[t] = phi * out[t - 1] + eps[t]
    return out


def _reflect(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Reflects values at the bounds instead of hard clipping.

    Hard clipping produces plateaus at the boundary (exactly the artifact
    look we deliberately do NOT take over from the MMASH data). Reflection
    keeps the series within the band and looks like a value bouncing off.
    """
    y = np.where(y < lo, 2 * lo - y, y)
    y = np.where(y > hi, 2 * hi - y, y)
    return np.clip(y, lo, hi)  # safety net for multiple reflections


def _apply_bounds(trend: np.ndarray, seasonal: np.ndarray, residual: np.ndarray,
                  lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Keeps the series within [lo, hi] by folding the required correction
    into the RESIDUAL channel.

    Returns (hr, residual) with the corrected residual, so that
    hr == trend + seasonal + residual holds EXACTLY for the stored channels.
    The ground-truth identity is the basis of the channel attribution
    (Phase 06); applying the reflection only to the total (old behaviour)
    broke it at every reflected point. Here the correction from the
    reflection is added to the residual, so the sum of the stored
    components always equals the emitted `hr`.
    """
    for _ in range(20):
        y = trend + seasonal + residual
        corrected = _reflect(y, lo, hi)
        delta = corrected - y
        if not np.any(delta):
            break
        residual = residual + delta
    return trend + seasonal + residual, residual


def draw_subject_profile(rng: np.random.Generator, P: dict) -> dict:
    """Draws baseline/phi/sigma as a joint subject tuple + jitter.

    Rationale: in MMASH, inertia (phi) and noise (sigma) correlate with the
    level; independent uniform draws would create combinations that do not
    occur in the real data. Jitter spreads the discrete profiles into a
    continuous population.
    """
    subj = P.get("subject_level", {})
    meds = subj.get("median_hr_per_subject") or []
    phis = subj.get("ar_phi_per_subject") or []
    sigs = subj.get("ar_sigma_per_subject") or []
    lo_b, hi_b = P["baseline_range"]
    lo_p, hi_p = P["ar"]["phi_range"]
    lo_s, hi_s = P["ar"]["sigma_range"]

    if len(meds) == len(phis) == len(sigs) and len(meds) > 0:
        i = int(rng.integers(len(meds)))
        baseline = meds[i] + rng.uniform(-3.0, 3.0)
        # Jitter volatility (stationary residual std) and inertia SEPARATELY,
        # then derive sigma. If phi and sigma were both jittered upwards, the
        # series std would grow beyond the true within-subject std.
        std_stat = sigs[i] / np.sqrt(max(1e-6, 1.0 - phis[i] ** 2))
        std_jit = std_stat * rng.uniform(0.85, 1.15)
        phi = phis[i] + rng.uniform(-0.05, 0.05)
        phi = float(np.clip(phi, max(0.5, lo_p - 0.05), min(0.97, hi_p + 0.02)))
        sigma = std_jit * np.sqrt(max(1e-6, 1.0 - phi ** 2))
        src = f"subject#{i}"
        subj_idx = i
    else:  # fallback: independent draw from the ranges
        baseline = rng.uniform(lo_b, hi_b)
        phi = rng.uniform(lo_p, hi_p)
        sigma = rng.uniform(lo_s, hi_s)
        src = "range"
        subj_idx = None

    return {
        "baseline": float(np.clip(baseline, lo_b, hi_b)),
        "ar_phi": phi,
        "ar_sigma": float(np.clip(sigma, lo_s * 0.5, hi_s * 1.2)),
        "profile_source": src,
        "subject_idx": subj_idx,
    }


def build_severity_params(anomaly: dict) -> dict:
    """Transposes anomaly[param][severity] -> severity[param] (as tuples)."""
    return {sev: {k: tuple(v[sev]) for k, v in anomaly.items()} for sev in SEVERITIES}


def _inject_anomaly(rng, cfg: DGPConfig, sev_params: dict, residual, trend,
                    a_type: str, severity: str, scenario: Scenario,
                    shift_cap: float = np.inf, drift_cap: float = np.inf):
    """Injects ONE anomaly; metadata for later metrics."""
    p = sev_params[severity]
    n = cfg.length
    lo_start = int(0.55 * cfg.context_length)

    if a_type == "spike":
        amp = rng.uniform(*p["spike_amp"])
        length = int(rng.integers(p["spike_len"][0], p["spike_len"][1] + 1))
        # 1.2: the whole spike must lie inside the context window
        start = int(rng.integers(lo_start, max(lo_start + 1, cfg.context_length - length)))
        end = min(start + length, cfg.context_length)
        residual[start:end] += amp * np.hanning(max(2, end - start))
        event = {"type": "spike", "start": start, "end": end,
                 "amplitude": float(amp), "severity": severity, "channel": "residual"}

    elif a_type == "level_shift":
        start = int(rng.integers(lo_start, cfg.context_length - 10))
        amp = min(rng.uniform(*p["shift_amp"]), shift_cap)  # respect the shift budget
        residual[start:] += amp
        event = {"type": "level_shift", "start": start, "end": n,
                 "amplitude": float(amp), "severity": severity, "channel": "residual"}

    elif a_type == "drift":
        start = int(rng.integers(lo_start, cfg.context_length - 10))
        total = min(rng.uniform(*p["drift_total"]), drift_cap)  # respect the drift budget
        if scenario == "trend":
            trend[start:] += np.linspace(0.0, total, n - start)   # slow channel
            channel = "trend"
        else:
            # fast channel: short ramp, then the offset PERSISTS. A purely
            # transient ramp would vanish before the forecast horizon and
            # the residual scenario would degenerate to 'none' for Phase 04.
            ramp_len = min(120, n - start)
            ramp = np.linspace(0.0, total, ramp_len)
            offset = np.concatenate([ramp, np.full(n - start - ramp_len, total)])
            residual[start:] += offset
            channel = "residual"
        event = {"type": "drift", "start": start, "end": n,
                 "amplitude": float(total), "severity": severity, "channel": channel}
    else:
        raise ValueError(f"Unknown anomaly type: {a_type}")

    return residual, trend, event


# ------------------------------------------------------------- series / dataset

def generate_series(rng_base: np.random.Generator, rng_anom: np.random.Generator,
                    cfg: DGPConfig, P: dict, scenario: Scenario,
                    series_id: int) -> dict:
    """Generates one series.

    Paired design: `rng_base` depends ONLY on (seed, series_id) -> identical
    base dynamics (profile, slope, seasonality, AR(1)) for series i in every
    scenario. `rng_anom` additionally depends on the scenario and is used
    exclusively for anomaly draws, so scenario differences cannot desync
    the base-dynamics RNG stream.
    """
    n = cfg.length

    # (a) Subject profile (coupled draw!) + (b) trend channel.
    # CENTERED: the slope acts symmetrically around the profile median so the
    # series median stays at the true person median (baseline*slope combos
    # like "highest median + steepest slope" would be unphysiological because
    # both come from the same very active persons).
    prof = draw_subject_profile(rng_base, P)
    slope = rng_base.uniform(*P["slope_range"])
    trend = prof["baseline"] + slope * (np.arange(n) - (n - 1) / 2.0)

    # (c) Seasonal: ultradian ~1h wave, period PER series from the band
    per_lo, per_hi = P["seasonal"]["period_range"]
    period = int(rng_base.integers(per_lo, per_hi + 1))
    amp = rng_base.uniform(*P["seasonal"]["amp_range"])
    phase = rng_base.uniform(0, 2 * np.pi)
    t = np.arange(n)
    seasonal = amp * np.sin(2 * np.pi * t / period + phase) \
             + 0.3 * amp * np.sin(4 * np.pi * t / period + phase)

    # (d) AR(1) with coupled (phi, sigma) from the profile
    residual = _ar1(rng_base, n, prof["ar_phi"], prof["ar_sigma"])

    # (d2) Circadian curvature in the trend channel (1.2). Drawn AFTER the AR
    # noise so the earlier base-dynamics draws stay in the same RNG order.
    circ_amps = P.get("subject_level", {}).get("circadian_amp_per_subject") or []
    circ_amp, circ_phase = 0.0, None
    if cfg.circadian and circ_amps and prof["subject_idx"] is not None \
            and prof["subject_idx"] < len(circ_amps):
        circ_amp = float(circ_amps[prof["subject_idx"]]) * rng_base.uniform(0.85, 1.15)
        circ_phase = float(rng_base.uniform(0, 2 * np.pi))
        circ = circ_amp * np.sin(2 * np.pi * np.arange(n) / 1440.0 + circ_phase)
        trend = trend + (circ - circ.mean())   # centred: keeps the person median

    # (e) Anomalies
    events = []
    sev_params = build_severity_params(P["anomaly"])
    # Persistent-offset budget: level shifts AND persistent residual drifts
    # both move the series level PERMANENTLY. They therefore share ONE budget
    # (the severe upper bound of the shift amplitudes); without it, stacked
    # shifts + drifts could lift the level by up to 90 bpm -> unphysiological
    # and ground-truth events would overlap impossibly sharply.
    # Spikes also draw from this pool when injected: a spike does not move the
    # level permanently, but capping its amplitude by the remaining budget
    # keeps the TOTAL injected amplitude per series bounded and prevents a
    # severe spike from stacking on top of a severe shift. (This is a design
    # choice: an alternative would be to exempt spikes entirely, which would
    # allow spike+shift combinations up to ~85 bpm total.)
    shift_budget = float(P["anomaly"]["shift_amp"]["severe"][1])
    shift_floor = float(P["anomaly"]["shift_amp"]["subtle"][0])
    # Drift budget (trend channel): sum of trend drifts must stay
    # physiological (severe upper bound). Without it, three severe drifts
    # could move the series by up to 90 bpm.
    drift_budget = float(P["anomaly"]["drift_total"]["severe"][1])
    drift_floor = float(P["anomaly"]["drift_total"]["subtle"][0])
    if scenario != "none":
        n_anom = int(rng_anom.integers(cfg.n_anomalies_range[0], cfg.n_anomalies_range[1] + 1))
        # In the trend scenario ONLY drift in the trend channel (adversarial check).
        types = ["drift"] if scenario == "trend" else ["spike", "level_shift", "drift"]
        for _ in range(n_anom):
            a_type = str(rng_anom.choice(types))
            severity = str(rng_anom.choice(SEVERITIES))
            # Budget pre-check BEFORE injection. Each scenario checks the budget
            # that the drawn anomaly would actually consume: the residual
            # scenario draws from the shared persistent-offset pool (shifts +
            # residual drifts), the trend scenario only from the trend-drift
            # pool. Checking the wrong budget (1.2 bug: always shift_budget,
            # which the trend scenario never decrements) let the loop inject
            # ghost events with amplitude ~0 once the real budget was spent.
            if scenario == "trend":
                budget_left = drift_budget < drift_floor
            elif a_type in ("level_shift", "drift"):
                budget_left = shift_budget < shift_floor
            else:
                # spike: transient, but its amplitude is still capped by the
                # remaining shift_budget below (see budget comment above) --
                # the pre-check does not fire for spikes.
                budget_left = False
            if budget_left:
                if scenario == "trend":
                    break  # trend scenario: ONLY trend drifts, no fallback channel
                a_type = "spike"  # persistent budget exhausted -> transient anomaly
            cap = drift_budget if (a_type == "drift" and scenario == "trend") else shift_budget
            residual, trend, event = _inject_anomaly(
                rng_anom, cfg, sev_params, residual, trend, a_type, severity, scenario,
                shift_cap=shift_budget, drift_cap=cap)
            used = min(event["amplitude"], cap)
            if a_type == "drift" and scenario == "trend":
                drift_budget -= used
            else:
                shift_budget -= used
            events.append(event)

    # (f) Total series + bounds: reflection, with the correction folded into
    # the residual -> hr == trend + seasonal + residual ALWAYS holds.
    hr, residual = _apply_bounds(trend, seasonal, residual, *P["hr_bounds"])

    return {
        "series_id": series_id, "scenario": scenario, "t": t, "hr": hr,
        "true_trend": trend, "true_seasonal": seasonal, "true_residual": residual,
        "events": events,
        "params": {**prof, "slope": float(slope), "seasonal_period": period,
                   "seasonal_amp": float(amp), "circadian_amp": circ_amp,
                   "circadian_phase": circ_phase},
    }


def generate_dataset(n_series: int, seed: int, scenario: Scenario,
                     cfg: DGPConfig, P: dict, params_sha: str,
                     dgp_hash: str) -> tuple[pd.DataFrame, list[dict]]:
    scen_idx = {"residual": 0, "trend": 1, "none": 2}[scenario]
    frames, metadata = [], []
    for i in range(n_series):
        rng_base = np.random.default_rng([seed, i])
        rng_anom = np.random.default_rng([seed, i, scen_idx])
        s = generate_series(rng_base, rng_anom, cfg, P, scenario, i)
        frames.append(pd.DataFrame({
            "series_id": s["series_id"], "t": s["t"], "hr": s["hr"],
            "true_trend": s["true_trend"], "true_seasonal": s["true_seasonal"],
            "true_residual": s["true_residual"],
        }))
        metadata.append({
            "series_id": s["series_id"], "scenario": s["scenario"], "seed": seed,
            "events": s["events"], "params": s["params"],
            "dgp_hash": dgp_hash, "params_sha256": params_sha[:12],
        })
    df = pd.concat(frames, ignore_index=True)
    df["scenario"] = scenario
    _plausibility_check(df, P)
    return df, metadata


def _plausibility_check(df: pd.DataFrame, P: dict) -> None:
    """Sanity check against the MMASH reference. Fails -> adjust the DGP."""
    g = df.groupby("series_id")["hr"]
    lo, hi = P["hr_bounds"]
    assert (df["hr"] >= lo).all() and (df["hr"] <= hi).all(), "HR outside the bounds!"
    assert g.mean().between(40, 140).all(), "Implausible series mean!"
    lengths = df.groupby("series_id").size()
    assert lengths.nunique() == 1, "Series have different lengths!"
    identity = (df["hr"] - (df["true_trend"] + df["true_seasonal"] + df["true_residual"])).abs()
    assert (identity <= 1e-9).all(), "Ground-truth identity hr = trend + seasonal + residual broken!"
    print(f"[Plausibility] {g.ngroups} series | series mean {g.mean().min():.1f}-"
          f"{g.mean().max():.1f} bpm | span {df['hr'].min():.1f}-{df['hr'].max():.1f} bpm "
          f"(MMASH reference: median 63.9-89.4)")


# ------------------------------------------------------------------ persistence

def _save(df: pd.DataFrame, meta: list[dict], scenario: str, out_dir: str) -> None:
    """Parquet (preferred) with CSV fallback, + JSONL metadata per scenario."""
    base = os.path.join(out_dir, f"synthetic_hr_mmash_{scenario}")
    try:
        df.to_parquet(base + ".parquet")
        saved = base + ".parquet"
    except ImportError:
        saved = base + ".csv.gz"
        df.to_csv(saved, index=False)
        print("[Note] pyarrow missing -> CSV fallback (.csv.gz). "
              "pip install pyarrow for parquet.")
    with open(os.path.join(out_dir, f"metadata_{scenario}.jsonl"), "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    print(f"[Saving] scenario='{scenario}': {df.shape[0]} points, "
          f"{df['series_id'].nunique()} series -> {saved}")


def main() -> None:
    ap = argparse.ArgumentParser(description="MMASH-calibrated HR generator")
    ap.add_argument("--params", default="data/params/dgp_params_mmash.json",
                    help="Parameter file from extract_dgp_params.py")
    ap.add_argument("--n-series", type=int, default=5,
                    help="Series per scenario (final run: 150)")
    ap.add_argument("--seed", type=int, default=67,
                    help="Seed; identical across all scenarios -> paired design")
    ap.add_argument("--out-dir", default="data/synthetic")
    ap.add_argument("--scenarios", nargs="+",
                    default=["residual", "trend", "none"],
                    choices=["residual", "trend", "none"],
                    help="Scenarios to generate (default: all three)")
    ap.add_argument("--no-circadian", action="store_true",
                    help="Do not add circadian curvature to the trend (reproduces 1.1)")
    args = ap.parse_args()
    print(f"[Version] {GENERATOR_VERSION} "
          f"(branch: DGP v1.2 — circadian trend curvature, context-safe spikes)")

    P, source, params_sha = load_params(args.params)
    cfg = DGPConfig(circadian=not args.no_circadian)
    dgp_hash = effective_config_hash(cfg, P, params_sha)
    print(f"[DGP] version={GENERATOR_VERSION} | parameters from: {source}")
    print(f"[DGP] calibration SHA256={params_sha[:12]} | config hash={dgp_hash}")
    print(f"[DGP] {P['provenance'].get('n_subjects', '?')} calibration persons "
          f"({P['provenance'].get('dataset', 'unknown')})")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "dgp_config_mmash.json"), "w") as f:
        json.dump({
            "generator_version": GENERATOR_VERSION,
            "config": asdict(cfg),
            "params": P,
            "params_sha256": params_sha,
            "dgp_hash": dgp_hash,
            "seed": args.seed, "n_series": args.n_series,
        }, f, indent=2, ensure_ascii=False)

    # Same seed per scenario: series i has identical base dynamics in all
    # scenarios (paired design); only the anomalies differ.
    for scenario in args.scenarios:
        df, meta = generate_dataset(args.n_series, args.seed, scenario,
                                    cfg, P, params_sha, dgp_hash)
        _save(df, meta, scenario, args.out_dir)


if __name__ == "__main__":
    main()

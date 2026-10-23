"""
Phase 3b: training corpus for the DLinear baseline arm.

Generates a SEPARATE synthetic corpus (default 1500 series per scenario,
seed 6677) from the same DGP calibration as the benchmark. The corpus is
used exclusively to fit the global linear (DLinear-style) model in
`src/04_forecast/run_forecast_dlinear.py`; the benchmark series (seed 67)
are never part of it.

Why a separate corpus: the DLinear reference (Zeng et al., AAAI 2023) wins
in the literature because it TRAINS on a large corpus. Chronos-2 likewise
saw a large corpus during pretraining. Giving the linear arm its own
training data makes the information budgets comparable - and keeps the
benchmark (seed 67) strictly held out.

Reproducibility: the corpus is fully determined by
  (dgp_params_mmash.json, generator_version, n_series, seed)
and can always be regenerated with this script; the trained weights are
committed to the repo (results/dlinear_weights/) so the forecast step
also runs without regenerating the corpus.

Usage:
    python src/03_synthesis/generate_dlinear_corpus.py            # 1500, seed 6677
    python src/03_synthesis/generate_dlinear_corpus.py --n-series 500
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_hr_mmash import (DGPConfig, Scenario, effective_config_hash,  # noqa: E402
                               generate_dataset, load_params, _save,
                               GENERATOR_VERSION)

SCENARIOS: tuple[str, ...] = ("residual", "trend", "none")
DEFAULT_SEED = 6677
DEFAULT_N = 1500


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Training corpus for the DLinear baseline (separate seed)")
    ap.add_argument("--params", default="data/params/dgp_params_mmash.json")
    ap.add_argument("--n-series", type=int, default=DEFAULT_N,
                    help="series per scenario (default: 1500)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="generator seed (default: 6677; benchmark uses 67)")
    ap.add_argument("--out-dir", default="data/dlinear_corpus")
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS),
                    choices=list(SCENARIOS))
    args = ap.parse_args()

    if args.seed == 67:
        raise SystemExit(
            "Refusing seed 67: that is the BENCHMARK seed. The training corpus "
            "must use a different seed so the benchmark series are never seen "
            "in training.")

    print(f"[Version] {GENERATOR_VERSION} | corpus for the DLinear baseline")
    P, source, params_sha = load_params(args.params)
    cfg = DGPConfig(circadian=True)
    dgp_hash = effective_config_hash(cfg, P, params_sha)
    print(f"[DGP] calibration SHA256={params_sha[:12]} | config hash={dgp_hash}")
    print(f"[Corpus] {args.n_series} series/scenario | seed {args.seed} "
          f"-> {args.out_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    for scenario in args.scenarios:
        df, meta = generate_dataset(args.n_series, args.seed,
                                    scenario, cfg, P, params_sha, dgp_hash)
        _save(df, meta, scenario, args.out_dir)


if __name__ == "__main__":
    main()

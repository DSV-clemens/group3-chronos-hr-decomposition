"""
Phase 1: MMASH RR intervals -> bpm time series (1 & 5 minutes) + QC plots.

Reads all user_*/RR.csv from --input, cleans IBI artifacts and writes
hr_1min_{uid}.csv / hr_5min_{uid}.csv (1 file = 1 person = 1 day) to
--out-dir. These files are the basis for DGP calibration (Phase 2).

Usage:
    python preprocess_rr.py --input data/raw/mmash --out-dir data/processed
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------------- Settings ----------------
WINDOW = 5                # rolling median over this many beats
TOLERANCE = 0.20          # IBI may deviate at most +-20 % from the local median
INTERVALS_MIN = (1, 5)    # minute intervals for the CSVs
MAX_HOURS = 24            # everything after this is cut off (0 = do not cut)
MIN_LOG_GAP = 5           # gaps of at least this length are listed individually in the log
# ------------------------------------------------


def clean_artifacts(df):
    """Remove artifacts: plausible range + deviation from the local median."""
    df = df[df.ibi_s.between(0.3, 2.0)]
    med = df.ibi_s.rolling(WINDOW, center=True, min_periods=1).median()
    df = df[(df.ibi_s - med).abs() <= TOLERANCE * med].copy()
    return df


def bpm_series(df, res_min):
    """Median bpm per res_min interval; gaps are linearly interpolated.
    Returns: series, number of interpolated values, missing mask (before
    interpolation)."""
    hr = (60 / df.ibi_s).groupby(df.slot // res_min).median()
    hr.name, hr.index.name = "bpm", "minute"
    hr.index = hr.index * res_min  # interval start as index
    hr = hr.reindex(range(int(hr.index.min()), int(hr.index.max()) + 1, res_min))
    missing = hr.isna()
    return hr.interpolate(method="linear"), int(missing.sum()), missing


def gaps_text(missing):
    """Gaps of at least MIN_LOG_GAP as text, e.g. '23:52-23:57 (6 min)'."""
    clock = lambda m: f"{int(m) // 60 % 24:02d}:{int(m) % 60:02d}"
    out, start, length = [], None, 0
    for minute, m in missing.items():
        if m:
            if start is None:
                start, length = int(minute), 1
            else:
                length += 1
        elif start is not None:
            if length >= MIN_LOG_GAP:
                out.append(f"{clock(start)}-{clock(start + length - 1)} ({length} min)")
            start, length = None, 0
    if start is not None and length >= MIN_LOG_GAP:
        out.append(f"{clock(start)}-{clock(start + length - 1)} ({length} min)")
    return "; ".join(out)


def plot_user(out, title, uid, hr1, hr5):
    ax = hr1.plot(figsize=(12, 4), lw=0.8, color="lightsteelblue", label="1 min")
    hr5.plot(ax=ax, lw=2, color="tab:blue", label="5 min")
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(m) // 60 % 24:02d}:{int(m) % 60:02d}" for m in ticks])
    ax.set_xlabel("Minute of day (HH:MM)")
    ax.set_ylabel("Beats per minute (bpm)")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(out / f"hr_overview_{uid}.png", dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="MMASH user_*/RR.csv -> bpm time series (1 & 5 min) + QC plots")
    ap.add_argument("--input", default="data/raw/mmash",
                    help="Folder with the user_* subfolders (MMASH)")
    ap.add_argument("--out-dir", default="data/processed",
                    help="Output folder for the bpm CSVs and plots")
    args = ap.parse_args()

    root = Path(args.input)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    users = [u for u in root.glob("user_*") if (u / "RR.csv").exists()]
    users.sort(key=lambda p: int("".join(filter(str.isdigit, p.name))))
    if not users:
        raise SystemExit(f"No user_*/RR.csv found in {root}.")
    print(f"{len(users)} users found in '{root}'\n")

    all_5min = {}
    for u in users:
        uid = "".join(filter(str.isdigit, u.name))
        df = pd.read_csv(u / "RR.csv")
        n_raw = len(df)
        df = clean_artifacts(df)
        if df.empty:
            print(f"{u.name}: EMPTY AFTER FILTERING - skipped")
            continue
        # Chronology: use the day column if it is plausible (never decreasing).
        # For data errors like user_8 (day jumps to -29 at night while
        # timestamps continue normally 23:59:59 -> 00:00:00), count midnight
        # transitions instead.
        t = pd.to_datetime(df.time, format="%H:%M:%S")
        if (df.day.diff() < 0).any():
            df["slot"] = (t.diff() < pd.Timedelta(0)).cumsum() * 1440 \
                + t.dt.hour * 60 + t.dt.minute
        else:
            df["slot"] = (df.day - df.day.min()) * 1440 \
                + t.dt.hour * 60 + t.dt.minute

        # Cut off after MAX_HOURS (24-h protocol; sensor is off afterwards)
        n_cut = 0
        if MAX_HOURS:
            end = int(df.slot.min()) + int(MAX_HOURS * 60)
            n_cut = int((df.slot >= end).sum())
            df = df[df.slot < end]

        hr_by_res, n_interp = {}, {}
        for res in INTERVALS_MIN:
            hr, n_interp[res], missing = bpm_series(df, res)
            hr.to_csv(out / f"hr_{res}min_{uid}.csv")
            hr_by_res[res] = hr
            if res == INTERVALS_MIN[0]:
                missing_1min = missing

        plot_user(out, u.name, uid, hr_by_res[INTERVALS_MIN[0]],
                  hr_by_res[INTERVALS_MIN[-1]])
        all_5min[u.name] = hr_by_res[INTERVALS_MIN[-1]]
        print(f"{u.name}: {n_raw} beats, {n_raw - len(df) - n_cut} filtered, "
              + "interpolated: "
              + ", ".join(f"{res}min={n_interp[res]}" for res in INTERVALS_MIN))
        if n_cut:
            print(f"    cut off after {MAX_HOURS} h: {n_cut} beats")
        large_gaps = gaps_text(missing_1min)
        if large_gaps:
            print(f"    Gaps >= {MIN_LOG_GAP} min: {large_gaps}")

    # ---- Overall comparison: all users, EACH with its own x-axis ----
    n = len(all_5min)
    fig, axes = plt.subplots(n, 1, sharey=True, figsize=(12, max(1.2 * n, 4)))
    if n == 1:
        axes = [axes]
    for ax, (name, hr) in zip(axes, all_5min.items()):
        x = (hr.index - hr.index[0]) / 60  # hours since start of the recording
        ax.plot(x, hr.values, lw=1.2, color="tab:blue")
        ax.set_ylabel(name.replace("user_", ""), rotation=0, ha="right",
                      labelpad=28, fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("Hours since start of the recording (each user has its own interval)")
    axes[0].set_title("MMASH - all users (5-minute median)")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out / "all_users_comparison.png", dpi=150)
    plt.close()
    print(f"\nOverall comparison saved: {out / 'all_users_comparison.png'}")


if __name__ == "__main__":
    main()

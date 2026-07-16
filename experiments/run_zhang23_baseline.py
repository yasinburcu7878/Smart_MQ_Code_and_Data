"""
Baseline evaluation: Zhang23 (adaptive threshold) and the classical
threshold baselines vs SmartMQ.

Provides two selectivity estimators so the same yardstick can be
applied to every method:

  * selectivity_reference_coupled(sig, dec):
        semantic change measured against each method's OWN last published
        embedding (parallels SmartMQ's internal metric). Use only as a
        per-system self-consistency description -- NOT as a cross-method
        scoreboard (it is unfair to methods that do not decide on this
        score; e.g. VBT can fall below 1.0 as an artifact).

  * selectivity_decision_independent(sig, dec):
        semantic change measured against the PREVIOUS SAMPLE (fixed,
        decision-independent). This is the fair common ruler for
        cross-method comparison and the recommended estimator.

Run as a script for the full matched-rate comparison:
    python experiments/run_zhang23_baseline.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from collections import deque

from core import run_smartmq, summarize, generate_signal, PARAMS
from core.smartmq import compute_embedding, cosine_sim, embedding_series, kappa_of
from core.baseline_zhang23 import run_zhang23, matched_zhang23, mae_zhang23


# ----------------------------------------------------------------------
# Classical baselines (kept self-contained so importing this module has
# no side effects; mirrors the definitions used in figures/).
# ----------------------------------------------------------------------
def run_sod(sig, delta=0.30):
    ref = sig[0]; dec = ["PUBLISH"]
    for x in sig[1:]:
        if abs(x - ref) > delta:
            dec.append("PUBLISH"); ref = x
        else:
            dec.append("SUPPRESS")
    return dec


def run_msod(sig, delta=0.30, max_interval=30):
    ref = sig[0]; last = 0; dec = ["PUBLISH"]
    for t, x in enumerate(sig[1:], 1):
        if abs(x - ref) > delta or (t - last) >= max_interval:
            dec.append("PUBLISH"); ref = x; last = t
        else:
            dec.append("SUPPRESS")
    return dec


def run_vbt(sig, thr=0.040, window=10, max_interval=30):
    buf = deque(maxlen=window); last = 0; dec = []
    for t, x in enumerate(sig):
        buf.append(x)
        lv = float(np.var(list(buf))) if len(buf) >= 3 else 0.0
        if t == 0 or lv > thr or (t - last) >= max_interval:
            dec.append("PUBLISH"); last = t
        else:
            dec.append("SUPPRESS")
    return dec


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def pr(dec):
    return sum(1 for d in dec if d == "PUBLISH") / len(dec)


def mae_zoh(sig, dec):
    last = sig[0]; s = 0.0
    for i, d in enumerate(dec):
        if d == "PUBLISH":
            last = sig[i]
        s += abs(sig[i] - last)
    return s / len(dec)


def aoi_peak(dec):
    last = 0; peak = 0
    for t, d in enumerate(dec):
        if d == "PUBLISH":
            last = t
        peak = max(peak, t - last)
    return peak


def selectivity_reference_coupled(sig, dec, P=PARAMS):
    """Semantic change vs each method's OWN last published embedding.
    Self-consistency description only; not a fair cross-method scoreboard."""
    buf = deque(maxlen=25); diffs = deque(maxlen=P["NOISE_W"])
    prev_emb = None; prev_x = None; smoothed = 1.0
    pub, sup = [], []
    for x, d in zip(sig, dec):
        buf.append(x)
        if prev_x is not None: diffs.append(x - prev_x)
        prev_x = x
        emb = compute_embedding(buf, P, kappa_of(diffs, P))
        if prev_emb is None:
            smoothed = 1.0
        else:
            raw = 1.0 - cosine_sim(emb, prev_emb)
            smoothed = P["SEM_ALPHA"] * smoothed + (1 - P["SEM_ALPHA"]) * raw
        (pub if d == "PUBLISH" else sup).append(smoothed)
        if d == "PUBLISH":
            prev_emb = list(emb)
    if not pub or not sup:
        return float("nan")
    return float(np.mean(pub) / (np.mean(sup) + 1e-12))


def selectivity_decision_independent(sig, dec, P=PARAMS):
    """Semantic change vs the PREVIOUS SAMPLE (fixed reference, independent
    of any decisions). Fair common ruler for cross-method comparison."""
    prev_emb = None; smoothed = 1.0
    Z = []
    for emb in embedding_series(sig, P):
        if prev_emb is None:
            smoothed = 1.0
        else:
            raw = 1.0 - cosine_sim(emb, prev_emb)
            smoothed = P["SEM_ALPHA"] * smoothed + (1 - P["SEM_ALPHA"]) * raw
        Z.append(smoothed); prev_emb = list(emb)
    pub = [z for z, d in zip(Z, dec) if d == "PUBLISH"]
    sup = [z for z, d in zip(Z, dec) if d == "SUPPRESS"]
    if not pub or not sup:
        return float("nan")
    return float(np.mean(pub) / (np.mean(sup) + 1e-12))


def matched_sod(sig, target):
    """Binary-search SOD delta to match a target publish rate."""
    lo, hi = 0.0, 20.0
    for _ in range(60):
        m = (lo + hi) / 2
        if pr(run_sod(sig, m)) > target:
            lo = m
        else:
            hi = m
    return run_sod(sig, (lo + hi) / 2)


# ----------------------------------------------------------------------
# Script entry point
# ----------------------------------------------------------------------

def matched_msod(sig, target, max_interval=30):
    lo, hi = 0.0, 20.0
    for _ in range(60):
        m = (lo + hi) / 2
        if pr(run_msod(sig, m, max_interval)) > target: lo = m
        else: hi = m
    return run_msod(sig, (lo + hi) / 2, max_interval)


def matched_vbt(sig, target, window=10, max_interval=30):
    lo, hi = 0.0, 10.0
    for _ in range(60):
        m = (lo + hi) / 2
        if pr(run_vbt(sig, m, window, max_interval)) > target: lo = m
        else: hi = m
    return run_vbt(sig, (lo + hi) / 2, window, max_interval)


def main():
    scenarios = [("flat", 1000), ("step", 1000), ("periodic", 1000), ("drift", 500)]
    N_SEEDS = 20
    METHODS = ["SmartMQ", "Zhang23", "SOD", "MSOD", "VBT"]
    print("=" * 74)
    print("Decision-independent selectivity Z "
          "(20-seed mean, all baselines matched to SmartMQ publish rate)")
    print("Reproduces the Fig. 3(c) selectivity values of the paper")
    print("=" * 74)
    hdr = f"{'scenario':9s} | " + " | ".join(f"{m:>8s}" for m in METHODS)
    print(hdr); print("-" * len(hdr))

    for sc, _ in scenarios:
        sel = {m: [] for m in METHODS}
        for seed in range(N_SEEDS):
            ss = generate_signal(sc, seed=seed)
            tr = run_smartmq(ss, seed=seed)
            tgt = summarize(tr)["pub_rate"]
            decs = {
                "SmartMQ": [r["decision"] for r in tr],
                "Zhang23": matched_zhang23(ss, tgt),
                "SOD":     matched_sod(ss, tgt),
                "MSOD":    matched_msod(ss, tgt),
                "VBT":     matched_vbt(ss, tgt),
            }
            for mname, dec in decs.items():
                sel[mname].append(selectivity_decision_independent(ss, dec))
        means = {m: float(np.nanmean(sel[m])) for m in METHODS}
        print(f"{sc:9s} | " + " | ".join(f"{means[m]:8.3f}" for m in METHODS))
    print("-" * len(hdr))

    # Hardware (single real deployment; no seed averaging)
    import pandas as pd
    csv = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                       "sht30_temperature_humidity.csv")
    if os.path.exists(csv):
        print("\nHARDWARE (SHT30):")
        df = pd.read_csv(csv)
        for sensor in ["temperature", "humidity"]:
            sig = df[sensor].values
            m_sm = summarize(run_smartmq(sig))
            dec_sm = [r["decision"] for r in run_smartmq(sig)]
            dec_z = matched_zhang23(sig, m_sm["pub_rate"], e_min=0.05, e_max_cap=2.0)
            print(f"  {sensor:11s} SmartMQ Z="
                  f"{selectivity_decision_independent(sig, dec_sm):.3f}"
                  f"  Zhang23 Z={selectivity_decision_independent(sig, dec_z):.3f}")


if __name__ == "__main__":
    main()

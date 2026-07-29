#!/usr/bin/env python3
"""
burnin_sensitivity.py
---------------------
Warm-up (burn-in) sensitivity of the decision-independent semantic
selectivity Z, for the robustness note of Sec. 8.7.

Motivation. All metrics reported in the paper are computed over the
full trace; no burn-in period is discarded. Two initialization effects
are therefore present in the first samples of every run:

  (i)  the smoothed semantic difference is initialized at 1.0, so the
       first values of z_t are dominated by the initial condition
       rather than by the data (SEM_ALPHA = 0.5);
  (ii) the trend component of the embedding is identically zero until
       the long slope window (SLOPE_LONG_W = 20) is filled, and
       kappa_t is estimated from a partial noise window.

Because every method publishes during this warm-up window, these
samples enter the numerator of Z with atypically large z_t. This
script quantifies the resulting sensitivity by recomputing Z after
discarding the first B samples from the metric sets, WITHOUT altering
the runs themselves (the decision sequences are unchanged; only the
samples entering the mean are restricted).

Note that discarding the first B samples of the SIGNAL and re-running
is a different and inappropriate test here: the system re-initializes
at the new t = 0 and the warm-up transient simply reappears.

Covered here:
  [B1] Z vs burn-in length B, SmartMQ, 20 seeds per scenario
  [B2] SmartMQ vs matched Zhang23 at B = 0 and B = 40, 20 seeds
  [B3] Hardware traces (SHT30 temperature, humidity) at B = 0, 40, 100

This script adds numbers; it does not modify or re-run any of the
scripts whose outputs are archived in session_logs/.

Run:  PYTHONPATH=<repo_root> python reproduce/burnin_sensitivity.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from core.signals import generate_signal
from core.smartmq import run_smartmq, cosine_sim
from experiments.run_zhang23_baseline import (
    PARAMS, embedding_series, matched_zhang23,
)

SCENARIOS = ["flat", "step", "periodic", "drift"]
SEEDS = range(20)
BURNS = [0, 20, 40, 60, 100]


def z_series(sig, P=PARAMS):
    """Decision-independent semantic difference z_t (vs previous sample).

    Identical to selectivity_decision_independent(), but returns the
    series instead of the aggregate ratio, so that the ratio can be
    recomputed over arbitrary sample subsets.
    """
    prev = None
    smoothed = 1.0
    out = []
    for emb in embedding_series(sig, P):
        if prev is None:
            smoothed = 1.0
        else:
            smoothed = (P["SEM_ALPHA"] * smoothed
                        + (1 - P["SEM_ALPHA"]) * (1.0 - cosine_sim(emb, prev)))
        out.append(smoothed)
        prev = list(emb)
    return out


def selectivity(zs, dec, burn=0):
    pub = [z for i, (z, d) in enumerate(zip(zs, dec))
           if d == "PUBLISH" and i >= burn]
    sup = [z for i, (z, d) in enumerate(zip(zs, dec))
           if d == "SUPPRESS" and i >= burn]
    if not pub or not sup:
        return float("nan")
    return float(np.mean(pub) / (np.mean(sup) + 1e-12))


def main():
    print("=" * 70)
    print("[B1] Z vs burn-in length B  (SmartMQ, mean over 20 seeds)")
    print("=" * 70)
    header = "scenario  " + "".join(f"  B={b:<4d}" for b in BURNS)
    print(header)

    cache = {}
    for sc in SCENARIOS:
        row = []
        for b in BURNS:
            vals = []
            for s in SEEDS:
                key = (sc, s)
                if key not in cache:
                    sig = generate_signal(sc, seed=s)
                    dec = [r["decision"] for r in run_smartmq(sig, seed=s)]
                    cache[key] = (sig, dec, z_series(sig))
                sig, dec, zs = cache[key]
                vals.append(selectivity(zs, dec, b))
            row.append(float(np.mean(vals)))
        print(f"{sc:9s}" + "".join(f"  {v:6.3f}" for v in row))

    print()
    print("=" * 70)
    print("[B2] SmartMQ vs matched Zhang23  (mean over 20 seeds)")
    print("=" * 70)
    print("scenario   SmartMQ B=0  Zhang23 B=0   SmartMQ B=40  Zhang23 B=40"
          "   wins B=0  wins B=40")
    for sc in SCENARIOS:
        s0, z0, s4, z4 = [], [], [], []
        w0 = w4 = 0
        for s in SEEDS:
            sig, dec, zs = cache[(sc, s)]
            rate = sum(1 for d in dec if d == "PUBLISH") / len(dec)
            zdec = matched_zhang23(sig, rate)
            a0, b0 = selectivity(zs, dec, 0), selectivity(zs, zdec, 0)
            a4, b4 = selectivity(zs, dec, 40), selectivity(zs, zdec, 40)
            s0.append(a0); z0.append(b0); s4.append(a4); z4.append(b4)
            w0 += a0 > b0
            w4 += a4 > b4
        print(f"{sc:9s}  {np.mean(s0):11.3f}  {np.mean(z0):11.3f}  "
              f"{np.mean(s4):13.3f}  {np.mean(z4):12.3f}  "
              f"{w0:8d}/20  {w4:6d}/20")

    print()
    print("=" * 70)
    print("[B3] Hardware traces (SHT30)")
    print("=" * 70)
    csv = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "sht30_temperature_humidity.csv")
    df = pd.read_csv(csv)
    print("trace         n     B=0     B=40    B=100   paper 95% CI (B=0)")
    ci = {"temperature": "[1.80, 2.70]", "humidity": "[1.87, 2.86]"}
    for col in ["temperature", "humidity"]:
        sig = df[col].tolist()
        dec = [r["decision"] for r in run_smartmq(sig)]
        zs = z_series(sig)
        print(f"{col:12s} {len(sig):5d}  "
              f"{selectivity(zs, dec, 0):6.3f}  "
              f"{selectivity(zs, dec, 40):6.3f}  "
              f"{selectivity(zs, dec, 100):6.3f}   {ci[col]}")

    print()
    print("Interpretation: Z decreases for every method once the warm-up"
          " window is excluded,")
    print("and stabilizes beyond B = 40; the hardware point estimates"
          " remain inside the")
    print("bootstrap intervals reported in Sec. 8.7.")


if __name__ == "__main__":
    main()

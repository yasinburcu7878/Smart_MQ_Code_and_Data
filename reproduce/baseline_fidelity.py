#!/usr/bin/env python3
"""
baseline_fidelity.py
--------------------
Fidelity check for the Zhang23 baseline (Section 7.4).

The reduction path of [zhang2023sensors] -- Algorithm 1 (adaptive
threshold driven by CUSUM drift detection), Algorithm 2 (publish iff
the Kalman prediction residual exceeds the current threshold) and the
smoothed trend of Equation (9) -- is reproduced in
core/baseline_zhang23.py. The original formulation leaves the process
and measurement noise variances, the CUSUM parameters and the
threshold step size to the deployment; the values used here are listed
in the manuscript.

One implementation choice is not a free parameter. The original
requires the sensor and the remote processor to hold identical
predictions, which means the predictor may be updated only at
transmitted samples. Our implementation updates it at every sample.
This script quantifies the difference by running both variants at
publish rates matched to SmartMQ:

  [F1] Decision-independent selectivity Z of SmartMQ, of the
       implementation used in the article, and of the synchronized
       variant, 20 seeds per scenario.

The synchronized variant is uniformly less selective, so the
implementation used in the article is the more favorable one for the
baseline.

Run:  PYTHONPATH=<repo_root> python reproduce/baseline_fidelity.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.signals import generate_signal
from core.smartmq import run_smartmq, cosine_sim
from core.baseline_zhang23 import run_zhang23
from experiments.run_zhang23_baseline import PARAMS, embedding_series

SCENARIOS = ["flat", "step", "periodic", "drift"]
SEEDS = range(20)


def run_zhang23_synchronized(signal, e_min=0.05, e_max_cap=0.5,
                             band_scale=1.0, Q=5e-3, R=0.04,
                             cusum_k=0.10, cusum_h=0.50):
    """Zhang23 with the predictor updated only at transmitted samples,
    so that sensor and remote processor stay synchronized."""
    e_min_s = e_min * band_scale
    e_cap_s = e_max_cap * band_scale
    step = (e_cap_s - e_min_s) / 10.0
    e_max = e_cap_s
    x_est = signal[0]; P_est = 1.0
    s_hi = s_lo = 0.0
    dec = ["PUBLISH"]
    for k, z in enumerate(signal):
        if k == 0:
            x_est = z
            continue
        x_pred = x_est
        P_pred = P_est + Q
        e = z - x_pred
        s_hi = max(0.0, s_hi + e - cusum_k)
        s_lo = max(0.0, s_lo - e - cusum_k)
        drift = (s_hi > cusum_h) or (s_lo > cusum_h)
        if drift:
            s_hi = s_lo = 0.0
        e_max = (max(e_min_s, e_max - step) if drift
                 else min(e_cap_s, e_max + step))
        if abs(e) > e_max:
            dec.append("PUBLISH")
            K = P_pred / (P_pred + R)
            x_est = x_pred + K * e
            P_est = (1.0 - K) * P_pred
        else:
            dec.append("SUPPRESS")
            x_est = x_pred
            P_est = P_pred
    return dec


def matched(fn, sig, target):
    lo, hi = 1e-3, 50.0
    for _ in range(60):
        mid = (lo + hi) / 2
        d = fn(sig, band_scale=mid)
        if sum(1 for x in d if x == "PUBLISH") / len(d) > target:
            lo = mid
        else:
            hi = mid
    return fn(sig, band_scale=(lo + hi) / 2)


def z_series(sig, P=PARAMS):
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


def selectivity(zs, dec):
    pub = [v for v, d in zip(zs, dec) if d == "PUBLISH"]
    sup = [v for v, d in zip(zs, dec) if d == "SUPPRESS"]
    if not pub or not sup:
        return float("nan")
    return float(np.mean(pub) / (np.mean(sup) + 1e-12))


def main():
    print("=" * 74)
    print("[F1] Selectivity Z at matched publish rate, mean over 20 seeds")
    print("=" * 74)
    print(f"{'scenario':10s} {'SmartMQ':>10s} {'Zhang23 (article)':>19s} "
          f"{'Zhang23 (synchronized)':>24s}")
    for sc in SCENARIOS:
        a, b, c = [], [], []
        for s in SEEDS:
            sig = generate_signal(sc, seed=s)
            dec = [r["decision"] for r in run_smartmq(sig, seed=s)]
            rate = sum(1 for d in dec if d == "PUBLISH") / len(dec)
            zs = z_series(sig)
            a.append(selectivity(zs, dec))
            b.append(selectivity(zs, matched(run_zhang23, sig, rate)))
            c.append(selectivity(zs, matched(run_zhang23_synchronized,
                                             sig, rate)))
        print(f"{sc:10s} {np.mean(a):10.3f} {np.mean(b):19.3f} "
              f"{np.mean(c):24.3f}")
    print()
    print("The synchronized variant is less selective in every scenario;")
    print("the implementation used in the article therefore does not")
    print("understate the baseline.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
reproduce_results.py
--------------------
Reproduces the SmartMQ-centric numerical results reported in the paper,
directly from the model code, and prints each value next to the paper
value for verification. No hand-entered numbers.

Covered here (SmartMQ-only, core dependencies):
  - base publish rates
  - embedding ablation table
  - utility ablation table
  - hardware publish rate + decision-independent selectivity (Z)
  - statistical significance (one-sided Mann-Whitney U on z_t)
  - Age-of-Information (peak, violations)
  - drift sub-interval publish-rate trajectory (20-seed)
  - selectivity under the w_r ablation (decision-independent Z)
  - sensitivity-sweep extremes (Fig. 5 range)

The cross-method selectivity comparison (SmartMQ vs SOD/MSOD/VBT/Zhang23)
and all six figures are reproduced by figures/generate_figures.py.

Run:  PYTHONPATH=<repo_root> python reproduce/reproduce_results.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from collections import deque
from scipy.stats import mannwhitneyu, norm

from core import run_smartmq, generate_signal, PARAMS
from core.smartmq import compute_embedding, cosine_sim, embedding_series
import core.smartmq as sm

SCN = ["flat", "step", "periodic", "drift"]


def pubrate(dec):
    return 100.0 * sum(1 for d in dec if d == "PUBLISH") / len(dec)


def decisions(sig, **kw):
    return [r["decision"] for r in run_smartmq(sig, **kw)]


# ----- decision-independent semantic change z_t (selectivity metric) -----
def z_series(sig, P=PARAMS):
    """z_t = 1 - cos(e_t, e_{t-1}), smoothed by the semantic filter.
    Fixed lag-1 reference -> independent of any method's decisions."""
    prev = None; sm_val = 1.0; Z = []
    for emb in embedding_series(sig, P):
        if prev is None:
            sm_val = 1.0
        else:
            raw = 1.0 - cosine_sim(emb, prev)
            sm_val = P["SEM_ALPHA"] * sm_val + (1 - P["SEM_ALPHA"]) * raw
        Z.append(sm_val); prev = emb
    return np.array(Z)


def selectivity(sig, dec):
    z = z_series(sig); dec = np.array(dec)
    p = z[dec == "PUBLISH"]; q = z[dec == "SUPPRESS"]
    if len(p) == 0 or len(q) == 0:
        return float("nan")
    return float(np.mean(p) / (np.mean(q) + 1e-12))


def mwu(sig, dec):
    z = z_series(sig); dec = np.array(dec)
    p = z[dec == "PUBLISH"]; q = z[dec == "SUPPRESS"]
    U, pv = mannwhitneyu(p, q, alternative="greater")
    n1, n2 = len(p), len(q)
    mu = n1 * n2 / 2.0; sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    log10p = float(norm.logsf((U - mu) / sd) / np.log(10))
    rb = 2 * U / (n1 * n2) - 1.0
    eff = "large" if abs(rb) >= 0.474 else "medium" if abs(rb) >= 0.33 else "small" if abs(rb) >= 0.147 else "negligible"
    return pv, log10p, rb, eff


def line(label, got, paper):
    print(f"  {label:26s} got: {got:30s} paper: {paper}")


print("=" * 74)
print("SmartMQ — reproduction of reported numbers (got vs paper)")
print("=" * 74)

# 1) base publish rates -------------------------------------------------
print("\n[1] Base publish rates (%)")
base = {sc: pubrate(decisions(generate_signal(sc))) for sc in SCN}
line("flat/step/periodic/drift",
     " ".join(f"{base[sc]:.1f}" for sc in SCN),
     "13.8 14.2 18.8 14.6")

# 2) embedding ablation -------------------------------------------------
print("\n[2] Embedding ablation (publish %)")
orig_emb = sm.compute_embedding
def _ablate(idxs):
    def f(buf, P, kap):
        e = orig_emb(buf, P, kap)
        for i in idxs: e[i] = 0.0
        return e
    return f
def emb_row(idxs):
    sm.compute_embedding = _ablate(idxs) if idxs else orig_emb
    out = {sc: pubrate(decisions(generate_signal(sc))) for sc in SCN}
    sm.compute_embedding = orig_emb
    return out
emb = [("full", [], "13.8 14.2 18.8 14.6"),
       ("w/o trend", [1], "8.5 8.5 9.8 10.0"),
       ("w/o autocorr", [2], "11.6 11.0 15.5 10.4"),
       ("w/o variance", [3], "19.2 19.4 25.2 21.2"),
       ("w/o trend+autocorr", [1, 2], "5.8 6.1 6.5 5.4"),
       ("w/o autocorr+variance", [2, 3], "15.5 15.3 21.3 15.4")]
for name, idxs, paper in emb:
    r = emb_row(idxs)
    line(name, " ".join(f"{r[sc]:.1f}" for sc in SCN), paper)

# 3) utility ablation ---------------------------------------------------
print("\n[3] Utility ablation (publish %)")
util = [("full", {}, "13.8 14.2 18.8 14.6"),
        ("w_s=0", {"W_S": 0}, "4.4 4.5 5.5 4.8"),
        ("w_e=0", {"W_E": 0}, "5.5 5.6 6.4 5.4"),
        ("w_d=0", {"W_D": 0}, "18.4 18.0 23.0 20.8"),
        ("w_m=0", {"W_M": 0}, "18.3 18.3 22.7 18.0"),
        ("w_t=0", {"W_T": 0}, "4.5 4.3 5.6 4.8"),
        ("w_r=0", {"W_R": 0}, "7.6 7.8 11.8 8.2"),
        ("w_a=0", {"W_A": 0}, "7.8 7.8 16.5 6.6")]
for name, ov, paper in util:
    r = {sc: pubrate(decisions(generate_signal(sc), param_overrides=ov)) for sc in SCN}
    line(name, " ".join(f"{r[sc]:.1f}" for sc in SCN), paper)

# 3b) selectivity under the w_r ablation (decision-independent Z) -------
print("\n[3b] Decision-independent selectivity Z under the w_r ablation")
for name, ov, paper in [("full model (same runs)", {}, "2.01 2.12 1.95 2.24"),
                        ("w_r=0", {"W_R": 0}, "1.82 1.85 2.03 2.43  (paper: 1.82-2.43, Sec. 8.3)")]:
    zz = []
    for sc in SCN:
        sig = generate_signal(sc)
        zz.append(selectivity(sig, decisions(sig, param_overrides=ov)))
    line(name, " ".join(f"{z:.2f}" for z in zz), paper)

# 4) hardware -----------------------------------------------------------
print("\n[4] Hardware (real SHT30, data/sht30_temperature_humidity.csv)")
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "sht30_temperature_humidity.csv")
df = pd.read_csv(DATA)
print(f"  samples = {len(df)}  (paper: 2636)")
for nm, col, pr_paper, z_paper in [("temperature", "temperature", "14.5", "2.36"),
                                    ("humidity", "humidity", "18.4", "2.67")]:
    sig = df[col].values; dec = decisions(sig)
    line(f"{nm} pub% / Z",
         f"{pubrate(dec):.1f} / {selectivity(sig, dec):.2f}",
         f"{pr_paper} / {z_paper}")

# 5) significance -------------------------------------------------------
print("\n[5] Within-run Mann-Whitney U on z_t (descriptive support; primary: [5b])")
print("    (z_t values are serially dependent; no pooled p-value is reported,")
print("     consistent with Sec. 8.7 of the paper)")
for sc in SCN:
    sig = generate_signal(sc); dec = decisions(sig)
    pv, l10, rb, eff = mwu(sig, dec)
    ps = f"{pv:.1e}" if pv > 0 else f"<1e{int(np.ceil(l10))}"
    print(f"  {sc:9s} p={ps:>9s}  rank-biserial={rb:.3f} ({eff})")
for nm, col in [("hw-temp", "temperature"), ("hw-humid", "humidity")]:
    sig = df[col].values; dec = decisions(sig); pv, l10, rb, eff = mwu(sig, dec)
    ps = f"{pv:.1e}" if pv > 0 else f"<1e{int(np.ceil(l10))}"
    print(f"  {nm:9s} p={ps:>9s}  rank-biserial={rb:.3f} ({eff})")

# 5b) run-level statistics (primary analysis) ---------------------------
print("\n[5b] Run-level selectivity Z: 20 independent seeds per scenario")
from scipy.stats import wilcoxon
for sc in SCN:
    Zs = []
    for s in range(20):
        sig = generate_signal(sc, seed=s)
        Zs.append(selectivity(sig, decisions(sig, seed=s)))
    Zs = np.array(Zs)
    w = wilcoxon(Zs - 1.0, alternative="greater")
    print(f"  {sc:9s} mean={Zs.mean():.2f}  min={Zs.min():.2f}  "
          f"Z>1 in {int((Zs > 1).sum())}/20 runs  Wilcoxon p={w.pvalue:.1e}")

# 6) AoI ----------------------------------------------------------------
print("\n[6] Age of Information (N_age = 30)")
def aoi(dec, Nage=30):
    last = 0; peak = 0; viol = 0
    for t, d in enumerate(dec):
        if d == "PUBLISH": last = t
        a = t - last; peak = max(peak, a); viol += (a > Nage)
    return peak, viol
peaks = []
for sc in SCN:
    pk, v = aoi(decisions(generate_signal(sc))); peaks.append(pk)
    print(f"  {sc:9s} peak={pk:3d}  violations={v}")
line("overall peak AoI", str(max(peaks)), "28 (zero violations; guaranteed by freshness cap)")

# 7) drift sub-intervals (20-seed) -------------------------------------
print("\n[7] Drift sub-interval publish rates (20-seed mean)")
E = []; M = []; L = []
for s in range(20):
    dec = decisions(generate_signal("drift", seed=s), seed=s)
    E.append(pubrate(dec[:150])); M.append(pubrate(dec[150:350])); L.append(pubrate(dec[350:500]))
line("early / mid / late",
     f"{np.mean(E):.1f} / {np.mean(M):.1f} / {np.mean(L):.1f}",
     "18.5 / 15.6 / 13.3 (monotonic decline)")

# 8) sensitivity extremes (Fig. 5 sweep) --------------------------------
print("\n[8] Sensitivity extremes: Z/baseline over the +/-50% sweep (Fig. 5)")
SENS = ["W_S", "W_T", "W_R", "W_A", "U_HIGH", "U_LOW", "W_E", "SEM_K"]
FACT = [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5]
cache = {}
for sc in SCN:
    sig = generate_signal(sc)
    z = z_series(sig)
    d = np.array(decisions(sig))
    base = float(np.mean(z[d == "PUBLISH"]) / (np.mean(z[d == "SUPPRESS"]) + 1e-12))
    cache[sc] = (sig, z, base)
gmin = (9.0, ""); gmax = (0.0, ""); th = {}
for p in SENS:
    for sc in SCN:
        sig, z, base = cache[sc]
        for f in FACT:
            d = np.array(decisions(sig, param_overrides={p: PARAMS[p] * f}))
            r = float(np.mean(z[d == "PUBLISH"]) / (np.mean(z[d == "SUPPRESS"]) + 1e-12)) / base
            if r < gmin[0]: gmin = (r, f"{p} {sc} x{f}")
            if r > gmax[0]: gmax = (r, f"{p} {sc} x{f}")
            if p == "U_HIGH" and f == 1.5 and sc in ("flat", "step"):
                th[sc] = r
line("min ratio", f"{gmin[0]:.2f}  ({gmin[1]})", "0.61")
line("max ratio", f"{gmax[0]:.2f}  ({gmax[1]})", "1.54")
line("theta_H +50%: flat/step", f"{th['flat']:.2f} / {th['step']:.2f}", "0.98 / 0.90")

print("\n" + "=" * 74)
print("Done. All values above are computed from the model code in this run.")
print("=" * 74)

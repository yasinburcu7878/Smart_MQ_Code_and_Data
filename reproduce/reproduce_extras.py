#!/usr/bin/env python3
"""
reproduce_extras.py
-------------------
Reproduces the remaining paper numbers NOT covered by
reproduce_results.py, and prints each value next to the paper value.
Together with reproduce_results.py, figures/generate_figures.py and
experiments/statistical_tests.py, this makes every number reported in
the paper regenerable from this repository.

Covered here:
  [E1] MAE overhead at matched publish rate vs SOD / MSOD   (Fig. 3b, Sec. 8.1.2)
  [E2] Freshness-cap activations, hw peak AoI, kappa medians (Secs. 8.5, 8.6)
  [E3] Event-detection delay on step, matched rate, 20 seeds (Sec. 8.1.3)
  [E4] Zhang23 at its DEFAULT operating point: AoI           (Sec. 8.5)
  [E5] Scale/offset-invariance on the real temperature trace (Table 4, Sec. 8.6)
  [E6] Dimensionless-constant robustness: C_TR, C_VAR sweep  (Sec. 8.4)
  [E7] Noise-window L in {50,100,200}: 20-run mean Z shift   (Sec. 8.4)
  [E8] Hardware moving-block bootstrap 95% CI for Z          (Secs. 8.6, 8.7)
  [E9] Paired per-run comparison vs matched Zhang23           (Sec. 8.7)

All sections are deterministic exact reproductions except [E8]: the
bootstrap CI bounds depend on the resampling RNG stream (point
estimates are deterministic); any seed reproduces the paper intervals
to within ~+/-0.05.

Run:  PYTHONPATH=<repo_root> python reproduce/reproduce_extras.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from core import run_smartmq, summarize, generate_signal, PARAMS
from core.baseline_zhang23 import run_zhang23, matched_zhang23
from experiments.run_zhang23_baseline import (
    run_sod, run_msod, run_vbt, matched_sod, matched_msod, matched_vbt,
    selectivity_decision_independent as sel_z, mae_zoh)
from experiments.statistical_tests import z_series

SCN = ["flat", "step", "periodic", "drift"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "sht30_temperature_humidity.csv")


def decisions(sig, **kw):
    return [r["decision"] for r in run_smartmq(sig, **kw)]


def line(label, got, paper):
    print(f"  {label:28s} got: {got:34s} paper: {paper}")


def forced_cap_count(trace, P=PARAMS):
    """Publishes forced by the freshness cap: age >= N_age AND U < theta_H.
    (At age >= N_age the previous decision is necessarily SUPPRESS, so
    without the cap these samples would not have been published.)"""
    lp = 0
    forced = 0
    for r in trace:
        if r["decision"] == "PUBLISH":
            if (r["t"] - lp) >= P["AGE_MAX"] and r["U"] < P["U_HIGH"]:
                forced += 1
            lp = r["t"]
    return forced


def aoi_stats(dec, Nage=30):
    lp = 0
    vals = []
    for t, d in enumerate(dec):
        if d == "PUBLISH":
            lp = t
        vals.append(t - lp)
    v = np.array(vals)
    return float(v.mean()), int(v.max()), 100.0 * float(np.mean(v > Nage))


def first_pub_delay(dec, onset=120):
    for t in range(onset, len(dec)):
        if dec[t] == "PUBLISH":
            return t - onset
    return None


print("=" * 76)
print("SmartMQ — reproduction of the remaining reported numbers (got vs paper)")
print("=" * 76)

# [E1] matched-rate MAE overhead --------------------------------------------
print("\n[E1] MAE overhead at matched publish rate (seed 42), x1e-2 degC")
paper_e1 = {"flat": "+2.9 / +2.3", "step": "+3.4 / +3.4",
            "periodic": "+4.9 / +4.9", "drift": "+0.5 / -0.1"}
for sc in SCN:
    sig = generate_signal(sc)
    tr = run_smartmq(sig)
    m = summarize(tr)
    o_sod = (m["mae"] - mae_zoh(sig, matched_sod(sig, m["pub_rate"]))) * 100
    o_ms = (m["mae"] - mae_zoh(sig, matched_msod(sig, m["pub_rate"]))) * 100
    line(f"{sc} (vs SOD / vs MSOD)", f"{o_sod:+.1f} / {o_ms:+.1f}",
         paper_e1[sc])

# [E2] freshness-cap activations --------------------------------------------
print("\n[E2] Freshness-cap activations (forced publish: age>=30 with U<theta_H)")
tot = 0
per = []
for sc in SCN:
    c = sum(forced_cap_count(run_smartmq(generate_signal(sc, seed=s), seed=s))
            for s in range(20))
    per.append(c)
    tot += c
line("20-seed totals f/s/p/d", " ".join(str(c) for c in per), "3 2 0 0")
line("total / samples", f"{tot} / 70000 = {100.0*tot/70000:.3f}%",
     "5 / 70000 = 0.007%")
c42 = [forced_cap_count(run_smartmq(generate_signal(sc))) for sc in SCN]
line("default seed 42 (all scn)", " ".join(str(c) for c in c42), "0 0 0 0")
df = pd.read_csv(DATA)
for nm, col, pk_p, kap_p in [("hw temperature", "temperature", "28", "0.010"),
                             ("hw humidity", "humidity", "24", "0.052")]:
    tr = run_smartmq(df[col].values.astype(float))
    kmed = float(np.median([r["kappa"] for r in tr]))
    pk = summarize(tr)["aoi_peak"]
    line(f"{nm} cap/peakAoI/kappa",
         f"{forced_cap_count(tr)} / {pk} / {kmed:.4f}",
         f"0 / {pk_p} / ~{kap_p}")

# [E3] event-detection delay -------------------------------------------------
print("\n[E3] Event-detection delay on STEP (onset t=120), matched rate, 20 seeds")
paper_e3 = {"SmartMQ": "mean 0.80  max 9  misses 0",
            "SOD": "mean 0.00  max 0  misses 0",
            "MSOD": "mean 0.00  max 0  misses 0",
            "VBT": "mean 0.05  max 1  misses 0",
            "Zhang23": "mean 0.00  max 0  misses 0"}
delays = {m: [] for m in paper_e3}
for s in range(20):
    sig = generate_signal("step", seed=s)
    tr = run_smartmq(sig, seed=s)
    tgt = summarize(tr)["pub_rate"]
    decs = {"SmartMQ": [r["decision"] for r in tr],
            "SOD": matched_sod(sig, tgt),
            "MSOD": matched_msod(sig, tgt),
            "VBT": matched_vbt(sig, tgt),
            "Zhang23": matched_zhang23(sig, tgt)}
    for mname, dec in decs.items():
        delays[mname].append(first_pub_delay(dec))
for mname, dl in delays.items():
    miss = sum(1 for d in dl if d is None or d > 60)
    ok = [d for d in dl if d is not None]
    line(mname,
         f"mean {np.mean(ok):.2f}  max {max(ok)}  misses {miss}",
         paper_e3[mname])

# [E4] Zhang23 default-operating-point AoI -----------------------------------
print("\n[E4] Zhang23 at DEFAULT operating point: AoI (seed 42)")
paper_e4 = {"flat": "2.3% / 41.8 / 148 / 52.2%",
            "step": "2.7% / 40.5 / 148 / 50.4%",
            "periodic": "9.0% / 9.3 / 40 / 2.8%",
            "drift": "2.0% / 29.4 / 82 / 45.8%"}
for sc in SCN:
    sig = generate_signal(sc)
    dec = run_zhang23(sig)
    mean_a, peak_a, viol = aoi_stats(dec)
    pub = 100.0 * sum(1 for d in dec if d == "PUBLISH") / len(dec)
    line(f"{sc} pub/mean/peak/viol",
         f"{pub:.1f}% / {mean_a:.1f} / {peak_a} / {viol:.1f}%",
         paper_e4[sc])

# [E5] scale/offset invariance (Table 4) --------------------------------------
print("\n[E5] Invariance on the real temperature trace (Table 4)")
temp = df["temperature"].values.astype(float)
base = decisions(temp)
for a, b, lab in [(1.0, 100.0, "x+100"), (1.8, 32.0, "1.8x+32"),
                  (10.0, 0.0, "10x"), (0.1, 0.0, "0.1x"),
                  (0.01, 5.0, "0.01x+5")]:
    d2 = decisions(a * temp + b)
    agr = 100.0 * float(np.mean([x == y for x, y in zip(base, d2)]))
    line(lab, f"{agr:.2f}% decision agreement", "100.00%")

# [E6] C_TR / C_VAR robustness (Fig-5 protocol: fixed metric, seed 42) --------
print("\n[E6] Dimensionless-constant robustness: C_TR, C_VAR x{0.5,0.75,1.25,1.5}")
cache = {}
for sc in SCN:
    sig = generate_signal(sc)
    cache[sc] = (sig, sel_z(sig, decisions(sig)))
gmin = (9.0, ""); gmax = (0.0, "")
for pname in ["C_TR", "C_VAR"]:
    for sc in SCN:
        sig, baseZ = cache[sc]
        for f in [0.5, 0.75, 1.25, 1.5]:
            r = sel_z(sig, decisions(sig,
                      param_overrides={pname: PARAMS[pname] * f})) / baseZ
            if r < gmin[0]: gmin = (r, f"{pname} x{f}, {sc}")
            if r > gmax[0]: gmax = (r, f"{pname} x{f}, {sc}")
line("min ratio", f"{gmin[0]:.2f}  ({gmin[1]})", "0.84 (C_TR x0.75, drift)")
line("max ratio", f"{gmax[0]:.2f}  ({gmax[1]})", "1.17 (C_VAR x0.5, drift)")

# [E7] noise-window L sweep ----------------------------------------------------
# Two protocols are reported for transparency:
#   (a) fixed ruler: the z_t metric keeps the default L=100 normalization
#       while the model's L is swept (same convention as [E6] / Fig. 5);
#   (b) co-varied:   the z_t normalization uses the same L as the model
#       (the self-consistent reading of Sec. 7.5).
# The paper states the bound "less than 0.05", which holds for both.
print("\n[E7] Noise-scale window L in {50,100,200}: 20-run mean selectivity Z")


def z_ratio(z, d):
    d = np.asarray(d)
    p, q = z[d == "PUBLISH"], z[d == "SUPPRESS"]
    return float(np.mean(p) / (np.mean(q) + 1e-12))


sig_cache = {(sc, s): generate_signal(sc, seed=s)
             for sc in SCN for s in range(20)}
z_fixed = {k: z_series(v) for k, v in sig_cache.items()}
meansA, meansB = {}, {}
for L in [50, 100, 200]:
    Pov = dict(PARAMS)
    Pov["NOISE_W"] = L
    for sc in SCN:
        va, vb = [], []
        for s in range(20):
            sig = sig_cache[(sc, s)]
            d = decisions(sig, seed=s, param_overrides={"NOISE_W": L})
            va.append(z_ratio(z_fixed[(sc, s)], d))
            vb.append(z_ratio(z_series(sig, Pov), d))
        meansA[(L, sc)] = float(np.mean(va))
        meansB[(L, sc)] = float(np.mean(vb))
devA = max(abs(meansA[(L, sc)] - meansA[(100, sc)])
           for L in [50, 200] for sc in SCN)
devB = max(abs(meansB[(L, sc)] - meansB[(100, sc)])
           for L in [50, 200] for sc in SCN)
line("mean Z at L=100 (f/s/p/d)",
     " ".join(f"{meansA[(100, sc)]:.2f}" for sc in SCN), "1.92 1.90 1.86 2.16")
line("max |dZ|, fixed ruler", f"{devA:.3f}", "< 0.05")
line("max |dZ|, co-varied norm.", f"{devB:.3f}", "< 0.05")

# [E8] hardware moving-block bootstrap CI --------------------------------------
print("\n[E8] Hardware moving-block bootstrap 95% CI for Z (block=50, 2000 reps)")
print("     (CI bounds depend on the resampling RNG stream; point estimates")
print("      are deterministic. Paper: temp [1.80, 2.70], humid [1.87, 2.86])")
rng = np.random.default_rng(42)
for nm, col, paper_ci in [("temperature", "temperature", "2.36  [1.80, 2.70]"),
                          ("humidity", "humidity", "2.67  [1.87, 2.86]")]:
    sig = df[col].values.astype(float)
    dec = np.array(decisions(sig))
    z = z_series(sig)
    N = len(z); block = 50
    nblocks = int(np.ceil(N / block))
    stats = []
    for _ in range(2000):
        starts = rng.integers(0, N - block + 1, nblocks)
        idx = np.concatenate([np.arange(s0, s0 + block) for s0 in starts])[:N]
        zz, dd = z[idx], dec[idx]
        p, q = zz[dd == "PUBLISH"], zz[dd == "SUPPRESS"]
        if len(p) and len(q):
            stats.append(float(p.mean() / (q.mean() + 1e-12)))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    point = float(z[dec == "PUBLISH"].mean() / (z[dec == "SUPPRESS"].mean() + 1e-12))
    line(nm, f"{point:.2f}  [{lo:.2f}, {hi:.2f}]", paper_ci)

# [E9] paired per-run comparison vs the strongest baseline --------------------
print("\n[E9] Paired per-run selectivity vs matched Zhang23 (20 seeds per scenario)")
from scipy.stats import wilcoxon
paper_e9 = {"flat": "19/20, p <= 1.3e-4", "step": "19/20, p <= 1.3e-4",
            "periodic": "15/20, p <= 1.3e-4", "drift": "19/20, p <= 1.3e-4"}
for sc in SCN:
    wins = 0; d_sm = []; d_zh = []
    for s in range(20):
        sig = sig_cache[(sc, s)]
        tr = run_smartmq(sig, seed=s)
        zs = z_ratio(z_fixed[(sc, s)], [r["decision"] for r in tr])
        zz = z_ratio(z_fixed[(sc, s)],
                     matched_zhang23(sig, summarize(tr)["pub_rate"]))
        d_sm.append(zs); d_zh.append(zz); wins += zs > zz
    w = wilcoxon(np.array(d_sm) - np.array(d_zh), alternative="greater")
    line(sc, f"{wins}/20 runs, p={w.pvalue:.1e}", paper_e9[sc])

print("\n" + "=" * 76)
print("Done. All values above are computed from the model code in this run.")
print("=" * 76)

"""
Statistical significance tests for the SmartMQ paper.

One-sided Mann-Whitney U on the DECISION-INDEPENDENT semantic change z_t
at PUBLISHED vs SUPPRESSED samples (H1: published > suppressed).

    z_t = 1 - cos(e_t, e_{t-1})   (smoothed by the semantic filter)

i.e. the embedding change vs the PREVIOUS sample -- a fixed reference
that no method optimizes, so the test is fair across methods and is not
an artifact of SmartMQ's own objective. This is the metric reported in
the paper. (The reference-coupled smoothed score in the run_smartmq
trace, 'smoothed_sem', is SmartMQ's INTERNAL decision signal, not the
evaluation metric, and is intentionally not used here.)

NOTE (v2): consecutive z_t values are serially dependent (EMA smoothing,
overlapping windows), so the within-trace Mann-Whitney tests below are
DESCRIPTIVE only. The PRIMARY significance analysis is the run-level
Wilcoxon over 20 independent seeds (independent observations by design).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np, pandas as pd
from collections import deque
from scipy import stats
from scipy.stats import norm
from core import run_smartmq, generate_signal, PARAMS
from core.smartmq import compute_embedding, cosine_sim, embedding_series


def z_series(sig, P=PARAMS):
    prev = None; sm = 1.0; Z = []
    for emb in embedding_series(sig, P):
        if prev is None:
            sm = 1.0
        else:
            raw = 1.0 - cosine_sim(emb, prev)
            sm = P["SEM_ALPHA"] * sm + (1 - P["SEM_ALPHA"]) * raw
        Z.append(sm); prev = emb
    return np.array(Z)


def split(sig):
    dec = np.array([r["decision"] for r in run_smartmq(sig)])
    z = z_series(sig)
    return z[dec == "PUBLISH"], z[dec == "SUPPRESS"]


def report(pub, sup, label):
    stat, p = stats.mannwhitneyu(pub, sup, alternative="greater")
    n1, n2 = len(pub), len(sup)
    r = 2 * stat / (n1 * n2) - 1
    effect = "large" if abs(r) >= 0.474 else ("medium" if abs(r) >= 0.33 else "small")
    mu = n1 * n2 / 2; sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    l10 = norm.logsf((stat - mu) / sd) / np.log(10)
    ps = f"p={p:.1e}" if p > 0 else f"p<10^{int(np.floor(l10))}"
    print(f"  {label:<14} {ps}  r={r:.3f} ({effect})")


if __name__ == "__main__":
    print("=" * 68)
    print("SmartMQ significance — Mann-Whitney U on decision-independent z_t")
    print("=" * 68)
    print("\n--- Simulation (within-run, descriptive; see run-level below) ---")
    for sc in ["flat", "step", "periodic", "drift"]:
        p, s = split(generate_signal(sc))
        report(p, s, sc.capitalize())
    print("\n--- Run-level (PRIMARY): 20 independent seeds per scenario ---")
    from scipy.stats import wilcoxon
    for sc in ["flat", "step", "periodic", "drift"]:
        Zs = []
        for s in range(20):
            sig = generate_signal(sc, seed=s)
            dec = np.array([r["decision"] for r in run_smartmq(sig, seed=s)])
            z = z_series(sig)
            Zs.append(float(np.mean(z[dec == "PUBLISH"]) /
                            (np.mean(z[dec == "SUPPRESS"]) + 1e-12)))
        Zs = np.array(Zs)
        w = wilcoxon(Zs - 1.0, alternative="greater")
        print(f"  {sc:<10} mean Z={Zs.mean():.2f}  min={Zs.min():.2f}  "
              f"Z>1 in {int((Zs > 1).sum())}/20 runs  Wilcoxon p={w.pvalue:.1e}")

    print("\n--- Hardware (descriptive, single deployment trace) ---")
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  "data", "sht30_temperature_humidity.csv"))
    for sensor, col in [("Temperature", "temperature"), ("Humidity", "humidity")]:
        p, s = split(df[col].values); report(p, s, sensor)

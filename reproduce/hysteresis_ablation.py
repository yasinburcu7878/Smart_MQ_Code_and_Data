#!/usr/bin/env python3
"""
hysteresis_ablation.py
----------------------
Ablation of the dual-threshold hysteresis band (Section 8.3).

The band is disabled by collapsing the two thresholds onto their
midpoint, theta_L = theta_H = (0.40 + 0.24) / 2, which turns the
dual-threshold rule of Equation (15) into a single-threshold rule
while leaving every other component untouched.

Reported here:
  [H1] Publish rate with and without the band (default configuration,
       matching the basis of Table 3)
  [H2] Number of transitions between the publish and suppress states,
       which is the quantity the band is intended to reduce

This script adds numbers; it does not modify or re-run any of the
scripts whose outputs are archived in session_logs/.

Run:  PYTHONPATH=<repo_root> python reproduce/hysteresis_ablation.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.signals import generate_signal
from core.smartmq import run_smartmq
from core.params import PARAMS

SCENARIOS = ["flat", "step", "periodic", "drift"]


def transitions(decisions):
    """Number of changes between consecutive publish/suppress states."""
    return sum(1 for a, b in zip(decisions, decisions[1:]) if a != b)


def publish_rate(decisions):
    return 100.0 * sum(1 for d in decisions if d == "PUBLISH") / len(decisions)


def main():
    mid = (PARAMS["U_HIGH"] + PARAMS["U_LOW"]) / 2.0
    print("=" * 72)
    print("[H1/H2] Hysteresis band ablation, default configuration")
    print(f"        band disabled by setting theta_L = theta_H = {mid:.2f}")
    print("=" * 72)
    print(f"{'scenario':10s} {'publish %':>10s} {'publish % (no band)':>21s} "
          f"{'transitions':>12s} {'transitions (no band)':>22s}")

    for sc in SCENARIOS:
        signal = generate_signal(sc)
        full = [r["decision"] for r in run_smartmq(signal)]
        flat_rule = [r["decision"] for r in
                     run_smartmq(signal, {"U_HIGH": mid, "U_LOW": mid})]
        print(f"{sc:10s} {publish_rate(full):10.1f} "
              f"{publish_rate(flat_rule):21.1f} "
              f"{transitions(full):12d} {transitions(flat_rule):22d}")

    print()
    print("paper: publish rates without the band are 16.1, 15.8, 15.1, 17.2")
    print("paper: transitions 111, 113, 145, 59 against 291, 285, 277, 151")


if __name__ == "__main__":
    main()

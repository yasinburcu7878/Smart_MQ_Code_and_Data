"""Signal generation for SmartMQ evaluation scenarios."""

import random, math, numpy as np
from .params import SCENARIOS

def generate_signal(scenario, seed=42):
    """Generate synthetic sensor signal for given scenario."""
    random.seed(seed); np.random.seed(seed)
    cfg = SCENARIOS[scenario]; n = cfg["n"]; sig = []
    for t in range(n):
        base = 20.0
        if cfg["step"] and t >= 120: base += 1.0
        if cfg["periodic"]: base += 0.5 * math.sin(2*math.pi*t/40)
        base += t * cfg["drift"]
        sig.append(base + random.gauss(0, 0.2))
    return sig

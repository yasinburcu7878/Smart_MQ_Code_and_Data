"""
Baseline: Zhang, Na & Zhang (2023) — Autonomous IoT Data Reduction
based on Adaptive Threshold  [ref 17], Sensors 23(23):9427.

Faithful re-implementation adapted to the SmartMQ publish/suppress
harness. Same interface as run_sod / run_msod / run_vbt:
    run_zhang23(signal, ...) -> ["PUBLISH"/"SUPPRESS", ...]

Mechanism (per the paper):
  - A basic (local-level) Kalman filter predicts each sample; the
    prediction error e_k = z_k - x_pred drives the decision:
    PUBLISH iff |e_k| > e_max.                         (Algorithm 2)
  - CUSUM concept-drift detection on the innovation triggers an
    ADAPTIVE threshold: on drift e_max is lowered (publish more),
    otherwise raised, clipped to [e_min, e_max_cap], step = band/10.
                                                       (Algorithm 1)
  - A smoothed first-difference trend d_k is maintained for
    trend-aware reconstruction at the receiver.        (Eq. 9 / Alg. 3)

This is a magnitude/prediction-error method: it ADAPTS its threshold
but still decides on instantaneous prediction error, not on the
temporal-character embedding used by SmartMQ.
"""

import numpy as np


def run_zhang23(signal,
                e_min=0.05, e_max_cap=0.5, band_scale=1.0,
                Q=5e-3, R=0.04,
                cusum_k=0.10, cusum_h=0.50,
                trend_alpha=0.6,
                return_state=False):
    """Return per-step PUBLISH/SUPPRESS decisions for the Zhang23 method.

    band_scale multiplies [e_min, e_max_cap] and is the knob used by
    matched_zhang23() to match a target publish rate (higher -> fewer
    publishes), exactly analogous to the SOD delta search.
    """
    e_min_s = e_min * band_scale
    e_cap_s = e_max_cap * band_scale
    step = (e_cap_s - e_min_s) / 10.0
    e_max = e_cap_s                      # start permissive (max reduction)

    x_est = signal[0]; P_est = 1.0       # Kalman local-level state
    s_hi = s_lo = 0.0                     # two-sided CUSUM accumulators
    d_k = 0.0; prev_z = signal[0]         # smoothed trend (Eq. 9)

    dec = ["PUBLISH"]                     # k=0 always transmitted
    e_max_trace = [e_max]
    for k, z in enumerate(signal):
        if k == 0:
            x_est = z; prev_z = z
            continue

        # --- Kalman prediction (random-walk / local-level) ---
        x_pred = x_est
        P_pred = P_est + Q
        e = z - x_pred                    # innovation / prediction error

        # --- CUSUM concept-drift detection on the innovation ---
        s_hi = max(0.0, s_hi + e - cusum_k)
        s_lo = max(0.0, s_lo - e - cusum_k)
        drift = (s_hi > cusum_h) or (s_lo > cusum_h)
        if drift:
            s_hi = s_lo = 0.0

        # --- Adaptive threshold (Algorithm 1) ---
        if drift:
            e_max = max(e_min_s, e_max - step)
        else:
            e_max = min(e_cap_s, e_max + step)

        # --- Decision (Algorithm 2) ---
        if abs(e) > e_max:
            dec.append("PUBLISH")
        else:
            dec.append("SUPPRESS")

        # --- Kalman correction ---
        K = P_pred / (P_pred + R)
        x_est = x_pred + K * e
        P_est = (1.0 - K) * P_pred

        # --- Smoothed trend d_k (Eq. 9) ---
        diff = z - prev_z
        d_k = diff if k == 1 else trend_alpha * diff + (1 - trend_alpha) * d_k
        prev_z = z
        e_max_trace.append(e_max)

    if return_state:
        return dec, e_max_trace
    return dec


def reconstruct_zhang23(signal, dec, trend_alpha=0.6):
    """Receiver-side trend-aware reconstruction (Algorithm 3).

    On PUBLISH the true value is received; on SUPPRESS the receiver
    extrapolates with the last known smoothed trend d_k instead of a
    flat hold. This is [17]'s native reconstruction; it is provided so
    that the baseline can also be evaluated under its own reconstruction
    rule, not only under zero-order hold.
    """
    recon = []
    last_val = signal[0]; d_k = 0.0; prev_z = signal[0]
    for k, (z, d) in enumerate(zip(signal, dec)):
        if d == "PUBLISH":
            # received: update trend from the newly known sample
            diff = z - prev_z
            d_k = diff if k <= 1 else trend_alpha * diff + (1 - trend_alpha) * d_k
            prev_z = z
            last_val = z
            recon.append(z)
        else:
            # not received: extrapolate along the last known trend
            last_val = last_val + d_k
            recon.append(last_val)
    return recon


def mae_zhang23(signal, dec, mode="zoh"):
    """MAE for Zhang23 under a chosen receiver-side reconstruction.

    This function is not used for any number reported in the paper: the
    reconstruction-fidelity comparison there (Fig. 3b) is made against
    SOD and MSOD under zero-order hold and is computed by mae_zoh().

    mode='zoh' (default): zero-order hold, the like-for-like setting
        also applied to SOD and MSOD.
    mode='native': trend extrapolation via reconstruct_zhang23(). Note
        that faithful Alg. 3 applies the trend only inside intervals
        where drift has been detected, whereas this unconditional
        variant also extrapolates across long flat runs; 'zoh' is
        therefore the default until drift gating is implemented.
    """
    if mode == "zoh":
        last = signal[0]; s = 0.0
        for i, d in enumerate(dec):
            if d == "PUBLISH": last = signal[i]
            s += abs(signal[i] - last)
        return s / len(dec)
    recon = reconstruct_zhang23(signal, dec)
    return float(np.mean([abs(a - b) for a, b in zip(signal, recon)]))


def _pr(dec):
    return sum(1 for d in dec if d == "PUBLISH") / len(dec)


def matched_zhang23(signal, target, **kw):
    """Binary-search band_scale so the publish rate matches `target`
    (same protocol as matched_delta for SOD). Returns the decision list."""
    lo, hi = 1e-3, 50.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _pr(run_zhang23(signal, band_scale=mid, **kw)) > target:
            lo = mid          # need higher thresholds -> fewer publishes
        else:
            hi = mid
    return run_zhang23(signal, band_scale=(lo + hi) / 2, **kw)

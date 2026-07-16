SmartMQ — Reproducibility Package
===================================

Paper: "SmartMQ: Utility-Driven Semantic-Difference-Based Transmission
Filtering for MQTT-Based IoT Systems"

Requirements:
  pip install numpy scipy pandas matplotlib

Structure:
  core/           SmartMQ simulation (single source of truth)
    params.py     All system parameters
    smartmq.py    Core run_smartmq(), summarize()
    signals.py    Signal generation
    baseline_zhang23.py  Adaptive-threshold baseline (see below)
  experiments/
    statistical_tests.py     Mann-Whitney U on decision-independent z_t
    run_zhang23_baseline.py  Matched-rate baseline comparison
  figures/
    generate_figures.py  Regenerates all six figures and prints the
                         cross-method selectivity table (Z)
  reproduce/
    reproduce_results.py Recomputes every reported SmartMQ number and
                         prints it next to the paper value (got vs paper)
    reproduce_extras.py  Matched-rate MAE overhead, freshness-cap counts,
                         event-detection delay, Zhang23 default AoI,
                         invariance (Table 4), C_TR/C_VAR and L sweeps,
                         hardware bootstrap CIs (got vs paper)
  data/
    sht30_temperature_humidity.csv Hardware deployment data (SHT30, 2h51min, 2636 samples)

Reproduce paper results:
  python reproduce/reproduce_results.py     # all SmartMQ numbers, got vs paper
  python reproduce/reproduce_extras.py      # remaining numbers incl. Table 4, got vs paper
  python figures/generate_figures.py        # six figures + cross-method table
  python experiments/statistical_tests.py   # significance (decision-independent)

Key parameters (Table 1 in paper):
  W = [0.22, 0.08, 0.05, 0.05, 0.20, 0.25, 0.23]
  U_HIGH=0.40, U_LOW=0.24, seed=42

-----------------------------------------------------------------
Adaptive-threshold baseline (Zhang23) and the selectivity metric
-----------------------------------------------------------------
  core/baseline_zhang23.py
    Faithful re-implementation of Zhang, Na & Zhang (2023),
    "Autonomous IoT data reduction based on adaptive threshold",
    Sensors 23(23):9427 -- Kalman prediction + CUSUM concept-drift
    + adaptive threshold (+ trend reconstruction). Same interface as
    the classical baselines:
        from core import run_zhang23, matched_zhang23
        dec = matched_zhang23(signal, target_publish_rate)

  experiments/run_zhang23_baseline.py
    Matched-rate comparison of SmartMQ vs Zhang23/SOD/MSOD/VBT.
    Provides two selectivity estimators:
      - selectivity_decision_independent()  <-- fair, recommended
            (semantic change vs the previous SAMPLE; common ruler)
      - selectivity_reference_coupled()
            (vs each method's own last published embedding; self-
            consistency description only, not a cross-method scoreboard)
    Run:  python experiments/run_zhang23_baseline.py

  METRIC (semantic selectivity Z): the paper reports DECISION-INDEPENDENT
  selectivity, denoted Z, where
      z_t = 1 - cos(e_t, e_{t-1})   (smoothed by the semantic filter)
  is the embedding change vs the PREVIOUS sample -- a fixed reference no
  method optimizes, so Z is non-circular and fair across methods. Under
  this ruler the baselines do NOT sit at ~1.0x: matched to SmartMQ's
  publish rate (20-seed mean), SmartMQ ~1.84-2.10x, Zhang23 ~1.68-1.84x,
  SOD ~1.58-1.66x, MSOD ~1.56-1.65x, VBT ~1.06-1.19x. The SmartMQ lead is
  genuine but modest; selectivity is a self-selectivity property, not the
  primary differentiator (the contribution is the integrated, utility-
  driven decision). selectivity_decision_independent() implements Z;
  selectivity_reference_coupled() is retained ONLY as a self-
  consistency description of SmartMQ's internal decision signal and must
  not be used as a cross-method scoreboard.

  NOTE on mae_zhang23(): default mode is zero-order hold (apples-to-apples
  with SOD/MSOD). The native trend reconstruction (mode="native") must be
  drift-gated before it is fair to the baseline (see code comment).

-----------------------------------------------------------------
HARDWARE DATA
-----------------------------------------------------------------
  data/sht30_temperature_humidity.csv contains the raw SHT30 hardware
  streams with the columns:
      timestamp, sample_index, temperature, humidity
  All reported hardware results are computed from these raw streams by
  run_smartmq() (see figures/generate_figures.py, hardware section).

  The dataset contains 2636 samples collected over 2h51min. sample_index
  spans 5..2730; the gap between this index range and the row count
  corresponds to dropped sensor reads during collection. All hardware
  results are computed over these 2636 samples.

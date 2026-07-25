# SmartMQ — Reproducibility Package

Reference implementation, hardware dataset, and reproduction scripts for the paper:

> **SmartMQ: Utility-Driven Semantic-Difference-Based Transmission Filtering for Message Queuing Telemetry Transport-Based Internet of Things Systems**
> Ecem İlayda Kay, Yasin Ünal, Volkan Rodoplu

Every number reported in the paper is recomputed by the scripts in `reproduce/`,
which print each value next to the published one (`got` vs `paper`).

## Requirements

```bash
pip install -r requirements.txt
```

Results were produced with Python 3.12, `numpy==2.4.4` and `scipy==1.17.1`.
These two versions are pinned because the reported figures are bit-reproducible
under them: running the scripts in a clean environment reproduces
`session_logs/` byte for byte.

## Reproduce the paper results

```bash
python3 reproduce/reproduce_results.py     # all SmartMQ numbers, got vs paper
python3 reproduce/reproduce_extras.py      # remaining numbers incl. Table 4, got vs paper
python3 figures/generate_figures.py        # six figures + cross-method table
python3 experiments/statistical_tests.py   # significance (decision-independent)
```

Reference outputs of all four scripts are stored in `session_logs/`.

## Structure

```
core/                          SmartMQ simulation (single source of truth)
  params.py                    All system parameters (= Table 1 of the paper)
  smartmq.py                   Core run_smartmq(), summarize()
  signals.py                   Signal generation
  baseline_zhang23.py          Adaptive-threshold baseline (see below)
experiments/
  statistical_tests.py         Mann-Whitney U / Wilcoxon on decision-independent z_t
  run_zhang23_baseline.py      Matched-rate baseline comparison
figures/
  generate_figures.py          Regenerates all six figures and prints the
                               cross-method selectivity table (Z)
  output/                      figure_3..figure_8 (.pdf vector + .png)
reproduce/
  reproduce_results.py         Recomputes every reported SmartMQ number
  reproduce_extras.py          Matched-rate MAE overhead, freshness-cap counts,
                               event-detection delay, Zhang23 default AoI,
                               invariance (Table 4), C_TR/C_VAR and L sweeps,
                               hardware bootstrap CIs
data/
  sht30_temperature_humidity.csv  Hardware deployment data (SHT30, 2h51min, 2636 samples)
session_logs/                  Reference outputs of the scripts above
```

## Key parameters (Table 1 in the paper)

```
w = [w_s, w_e, w_d, w_m, w_t, w_r, w_a] = [0.22, 0.08, 0.05, 0.05, 0.20, 0.25, 0.23]
theta_H = 0.40, theta_L = 0.24, L = 100, N_age = 30, seed = 42
```

`core/params.py` is the single source of truth; every script imports from it.

## Adaptive-threshold baseline (Zhang23) and the selectivity metric

**`core/baseline_zhang23.py`** — faithful re-implementation of

> Zhang, H.; Na, J.; Zhang, B. *Autonomous Internet of Things (IoT) Data
> Reduction Based on Adaptive Threshold.* Sensors 2023, 23(23), 9427.
> https://doi.org/10.3390/s23239427

(Kalman prediction + CUSUM concept-drift detection + adaptive threshold,
plus trend reconstruction). Same interface as the classical baselines:

```python
from core import run_zhang23, matched_zhang23
dec = matched_zhang23(signal, target_publish_rate)
```

**`experiments/run_zhang23_baseline.py`** — matched-rate comparison of SmartMQ
against Zhang23/SOD/MSOD/VBT. It provides two selectivity estimators:

- `selectivity_decision_independent()` — **fair, recommended**: semantic change
  measured against the previous *sample*, a common ruler for all methods.
- `selectivity_reference_coupled()` — against each method's own last published
  embedding; a self-consistency description only, **not** a cross-method
  scoreboard.

**Metric (semantic selectivity Z).** The paper reports the
decision-independent selectivity Z, where

```
z_t = 1 - cos(e_t, e_{t-1})     (smoothed by the semantic filter)
```

is the embedding change against the *previous sample* — a fixed reference that
no method optimizes, so Z is non-circular and fair across methods. Under this
ruler the baselines do **not** sit at ~1.0x. Matched to SmartMQ's publish rate
(20-seed mean, per scenario):

| Method  | Selectivity Z |
|---------|---------------|
| SmartMQ | 1.86x – 2.16x |
| Zhang23 | 1.65x – 1.77x |
| SOD     | 1.56x – 1.66x |
| MSOD    | 1.55x – 1.66x |
| VBT     | 1.05x – 1.17x |

The SmartMQ lead is genuine but modest; selectivity evidences that the
embedding front-end is semantically grounded, and is not the primary
differentiator — the contribution is the integrated, utility-driven decision.

**Note on `mae_zhang23()`.** The default mode is zero-order hold
(apples-to-apples with SOD/MSOD). The native trend reconstruction
(`mode="native"`) must be drift-gated before it is fair to the baseline
(see the code comment).

## Hardware data

`data/sht30_temperature_humidity.csv` contains the raw SHT30 streams with the
columns `timestamp, sample_index, temperature, humidity`. All reported hardware
results are computed from these raw streams by `run_smartmq()`.

The dataset contains 2636 samples logged at a nominal 3 s interval over
2h51min. `sample_index` spans 5..2730; the gap between this index range and the
row count corresponds to dropped sensor reads during collection
(17 interruptions, the largest ≈31 minutes, ≈23% of nominal slots missing).
Timestamps are retained so that the gap structure is fully inspectable. As
described in the paper, the analysis treats the logged stream as a contiguous
discrete-time sequence, with AoI defined in the sample-index domain.

## Determinism

Point estimates are deterministic: with the pinned versions above, all four
scripts reproduce `session_logs/` byte for byte. The only resampling-dependent
quantity is the hardware moving-block bootstrap in `reproduce_extras.py` [E8];
its confidence-interval bounds depend on the resampling RNG stream and may
differ from the published bounds by about ±0.02, while the point estimates
(2.36x and 2.67x) are exact. The script prints both.

## License

Released under the MIT License — see [LICENSE](LICENSE).

## Citation

If you use this code or dataset, please cite the paper (details will be updated
upon publication) and the archived release:

```
Kay, E. İ., Ünal, Y., & Rodoplu, V. SmartMQ: Utility-Driven
Semantic-Difference-Based Transmission Filtering for Message Queuing Telemetry Transport-Based Internet of Things Systems.
```

Archived release: [Zenodo DOI — to be added]

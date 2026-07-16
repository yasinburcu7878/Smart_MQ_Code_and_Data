"""
SmartMQ Master Parameters — Single source of truth.
All experiments and figures must use these values.

v2 (scale-normalized model): the unit-bearing scale parameters of v1
(SLOPE_LONG_SCALE, VAR_SCALE, DRIFT_MAX — all in sensor units) are
replaced by DIMENSIONLESS constants C_TR, C_VAR, C_R that multiply the
online noise-scale estimate kappa_t. At the simulation noise level
(sigma = 0.2), C_TR*kappa = 0.25*0.2 = 0.05, C_VAR*kappa = 0.2 and
C_R*kappa = 2.0 reproduce the v1 operating point exactly.
"""

PARAMS = dict(
    W_S=0.22, W_E=0.08, W_D=0.05, W_M=0.05,
    W_T=0.20, W_R=0.25, W_A=0.23,
    U_HIGH=0.40, U_LOW=0.24,
    SEM_ALPHA=0.5, SEM_K=5.0,
    SLOPE_LONG_W=20, AUTOCORR_W=8, VAR_W=10,
    C_TR=0.25, C_VAR=1.0, C_R=10.0,          # dimensionless scale constants
    NOISE_W=100, KAPPA_EPS=1e-9,             # noise-scale estimator window / floor
    AGE_MAX=30,
    DELAY_MIN_MS=20.0, DELAY_MAX_MS=150.0, DELAY_STD_MS=5.0,
    BATTERY_DECAY_PUB_MIN=0.0015, BATTERY_DECAY_PUB_MAX=0.0025,
    BATTERY_DECAY_SUP_MIN=0.00005, BATTERY_DECAY_SUP_MAX=0.00015,
)

SCENARIOS = {
    "flat":     {"n": 1000, "drift": 0.0,   "step": False, "periodic": False},
    "step":     {"n": 1000, "drift": 0.0,   "step": True,  "periodic": False},
    "periodic": {"n": 1000, "drift": 0.0,   "step": False, "periodic": True},
    "drift":    {"n": 500,  "drift": 0.012, "step": False, "periodic": False},
}

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VAR with Shifting Endpoint (VAR-SE)
====================================
BCB Estudo Especial 19/2018 — Kozicki & Tinsley (2012)

Replication of the BCB boxe methodology:
  - 4-variable monthly VAR(2): IPCA livres, IPCA admin, Δ USD/BRL, juros real
  - Shifting endpoint: inflation intercept μ_t follows a random walk
  - Anchoring: μ_t pinned to Focus 48-month-ahead survey expectations
  - Estimation: two-step (OLS VAR → Kalman filter MLE for noise variances)
  - Evaluation: recursive OOS with Diebold-Mariano (1995) and Clark-West (2007)

References
----------
Areosa & Gaglianone (2018) — BCB Estudo Especial 19/2018
Kozicki & Tinsley (2012) — JMCB 44: 145-169
Diebold & Mariano (1995) — JBES 13: 253-265
Clark & West (2007) — JE 138: 291-311
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
from scipy.optimize import minimize
import statsmodels.api as sm

from bcb import currency, sgs, Expectativas

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

START  = '2002-01-01'
END    = '2026-04-01'
H      = 48   # Shifting endpoint horizon (months)
P_VAR  = 3    # VAR lags

# SGS series codes
SGS = {
    'ipca_total'  : 433,    # IPCA total / headline (% a.m.) — J1 selects this
    'selic_am'    : 4390,   # Selic realizada acumulada no mês (% a.m.)
}

# Endogenous variables: [pi_total, dfx, r_real]
# Using headline IPCA so that J1 and the Focus anchor (also headline) are consistent.
# The BCB paper used [pi_livres, pi_admin, dfx, r_real] but anchored to headline
# Focus — this variant removes the mismatch.
ENDO_COLS = ['pi_total', 'dfx', 'r_real']

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def download_data(start: str, end: str) -> pd.DataFrame:
    print("Downloading BCB data...")

    # Monthly macro series
    macro = sgs.get(SGS, start=start, end=end)
    macro.index = pd.to_datetime(macro.index)
    macro.index = macro.index.to_period('M').to_timestamp('M')

    # Real interest rate (ex-post): Selic a.m. minus IPCA a.m.
    real_rate = macro['selic_am'] - macro['ipca_total']
    real_rate.name = 'r_real'

    # FX: USD/BRL end-of-month, monthly % change
    usdbrl = currency.get('USD', start=start, end=end)
    usdbrl_m = usdbrl.resample('ME').last()
    usdbrl_m.index = usdbrl_m.index.to_period('M').to_timestamp('M')
    dfx = usdbrl_m['USD'].pct_change() * 100
    dfx.name = 'dfx'

    # Focus: 4-year-ahead headline IPCA expectations (% a.a.)
    focus = _download_focus(start)

    df = pd.concat(
        [macro['ipca_total'], dfx, real_rate, focus],
        axis=1, join='inner'
    ).dropna()
    df.columns = ['pi_total', 'dfx', 'r_real', 'focus_48m']
    df = df.loc[start:end]

    print(f"  Sample: {df.index[0].strftime('%Y-%m')} → {df.index[-1].strftime('%Y-%m')}"
          f"  ({len(df)} obs)")
    return df


def _download_focus(start: str) -> pd.Series:
    """4-year-ahead Focus IPCA expectation, interpolated across calendar years."""
    ep = Expectativas().get_endpoint('ExpectativasMercadoAnuais')
    raw = (ep.query()
           .filter(ep.Indicador == 'IPCA')
           .filter(ep.Data >= start)
           .filter(ep.baseCalculo == 0)
           .select(ep.Data, ep.Media, ep.DataReferencia)
           .orderby(ep.Data.asc())
           .collect())

    raw['Data'] = pd.to_datetime(raw['Data'])
    raw['DataReferencia'] = raw['DataReferencia'].astype(int)

    pivot = raw.pivot_table(index='Data', columns='DataReferencia',
                            values='Media', aggfunc='last')

    def _interp(row):
        survey_date = row.name
        target_year = survey_date.year + 4
        w = survey_date.month / 12.0
        val_lo = row.get(target_year, np.nan)
        val_hi = row.get(target_year + 1, np.nan)
        if np.isnan(val_lo) and np.isnan(val_hi):
            return np.nan
        if np.isnan(val_hi):
            return val_lo
        if np.isnan(val_lo):
            return val_hi
        return (1 - w) * val_lo + w * val_hi

    series = pivot.apply(_interp, axis=1).resample('ME').last()
    series.index = series.index.to_period('M').to_timestamp('M')
    series.name = 'focus_48m'
    return series

# ══════════════════════════════════════════════════════════════════════════════
# 2. OLS VAR(p)
# ══════════════════════════════════════════════════════════════════════════════

def ols_var(Y: np.ndarray, p: int):
    """
    Estimate VAR(p) by OLS equation-by-equation.

    Parameters
    ----------
    Y : (T, k) array
    p : number of lags

    Returns
    -------
    intercept : (k,)
    Phis      : list of p matrices, each (k, k)
    Sigma     : (k, k) residual covariance
    resid     : (T-p, k) residuals
    """
    T, k = Y.shape
    X_list = [np.ones(T - p)]
    for lag in range(1, p + 1):
        X_list.append(Y[p - lag: T - lag])  # shape (T-p, k)
    X = np.column_stack(X_list)             # (T-p, 1 + k*p)
    Y_dep = Y[p:]                           # (T-p, k)

    coef = np.linalg.lstsq(X, Y_dep, rcond=None)[0]  # (1+k*p, k)
    resid = Y_dep - X @ coef

    intercept = coef[0]                     # (k,)
    Phis = [coef[1 + i*k: 1 + (i+1)*k].T  # (k, k)
            for i in range(p)]
    dof = T - p - 1 - k * p
    Sigma = resid.T @ resid / dof

    return intercept, Phis, Sigma, resid


def enforce_stability(Phis: list, k: int, intercept: np.ndarray,
                      Y_mean: np.ndarray,
                      max_modulus: float = 0.96) -> tuple[list, np.ndarray]:
    """
    If the VAR companion's spectral radius exceeds max_modulus, rescale
    all lag matrices uniformly so the largest eigenvalue equals max_modulus.

    After rescaling, the intercept is recomputed so the VAR unconditional
    mean equals Y_mean (the sample mean).  Without this correction the
    implied long-run level would shift, producing explosive-looking forecasts.

    Returns (Phis_scaled, intercept_corrected).
    """
    n_comp = k * len(Phis)
    companion = np.zeros((n_comp, n_comp))
    for i, Phi in enumerate(Phis):
        companion[:k, i*k:(i+1)*k] = Phi
    if len(Phis) > 1:
        companion[k:, :k*(len(Phis)-1)] = np.eye(k * (len(Phis)-1))

    rho = np.abs(np.linalg.eigvals(companion)).max()
    if rho > max_modulus:
        scale = max_modulus / rho
        Phis  = [Phi * scale for Phi in Phis]
        warnings.warn(
            f"VAR companion spectral radius {rho:.4f} > {max_modulus}; "
            f"lag matrices rescaled by {scale:.4f} to enforce stationarity."
        )

    # Recompute intercept so long-run mean = Y_mean regardless of scaling
    intercept_corrected = (np.eye(k) - sum(Phis)) @ Y_mean
    return Phis, intercept_corrected


# ══════════════════════════════════════════════════════════════════════════════
# 3. STATE-SPACE MATRICES
# ══════════════════════════════════════════════════════════════════════════════
#
# State X_t  (n = k*p + 2):
#   [0:k]       = y_t   (current endogenous)
#   [k:2k]      = y_{t-1}
#   ...
#   [xbar_idx]  = x̄_t  (constant state, always 1)
#   [mu_idx]    = μ_t   (random walk intercept for inflation)
#
# Observables Y_t  (k+1):
#   [0:k]  = y_t  (directly observed, near-zero noise)
#   [k]    = f_{t+H|t}  (Focus survey, % a.a. ≈ sum of 12 monthly forecasts)
#
# Observation constraint:
#   C_survey @ X_t = J1 @ sum_{h=H-11}^{H} A^h @ X_t
#   This sum of 12 monthly forecasts ≈ annual % (same units as Focus)

def build_state_noise(Sigma: np.ndarray, sigma2_mu: float, k: int, p: int) -> np.ndarray:
    """
    State noise covariance Q for X_t = A X_{t-1} + u_t.

    Shocks hit:
      - the current VAR block y_t through Sigma
      - the shifting endpoint mu_t through sigma2_mu
    """
    n = k * p + 2
    mu_idx = k * p + 1

    Q = np.zeros((n, n))
    Q[:k, :k] = Sigma          # VAR innovations
    Q[mu_idx, mu_idx] = sigma2_mu
    return Q

def build_transition(k: int, p: int, Phis: list, intercept: np.ndarray,
                     varse: bool = False) -> np.ndarray:
    """
    Companion-form transition matrix A (n × n).

    varse=False (unrestricted VAR):
        A[0, mu_idx] = 1.0  →  μ_t acts as a raw additive intercept.
        X0[mu_idx] = intercept_ols[0] gives the correct OLS long-run level.

    varse=True (VAR-SE, deviation-from-endpoint form, Kozicki & Tinsley 2012):
        A[0, mu_idx] = 1 - Σ Φ_j[0,0]  →  long-run inflation = μ_T directly,
        so the forecast stabilises at Focus without slow amplification drift.
    """
    n = k * p + 2
    xbar_idx = k * p
    mu_idx   = k * p + 1

    A = np.zeros((n, n))

    # VAR lag matrices
    for i, Phi in enumerate(Phis):
        A[:k, i*k:(i+1)*k] = Phi

    # Companion shift (y_{t-1} ← y_t, etc.)
    if p > 1:
        A[k:k*p, :k*(p-1)] = np.eye(k * (p - 1))

    # x̄_t column: fixed intercepts for non-inflation equations
    c_col = intercept.copy()
    c_col[0] = 0.0           # inflation: no fixed intercept (μ_t handles it)
    A[:k, xbar_idx] = c_col

    # μ_t column
    if varse:
        phi_sum_row0 = sum(Phi[0, 0] for Phi in Phis)
        A[0, mu_idx] = 1.0 - phi_sum_row0   # deviation-from-endpoint
    else:
        A[0, mu_idx] = 1.0                   # raw intercept (correct for OLS VAR)

    # x̄ and μ are random walks (diagonal = 1)
    A[xbar_idx, xbar_idx] = 1.0
    A[mu_idx,   mu_idx]   = 1.0

    return A


def build_obs_matrix(A: np.ndarray, k: int, n: int, H: int) -> np.ndarray:
    """
    Observation matrix C  (k+1, n).

    Row 0..k-1  : identity — directly observe y_t
    Row k       : survey constraint — sum_{h=H-11}^{H} J1 @ A^h
                  Sum of 12 monthly forecasts ≈ annual inflation (% a.a.)
    """
    C = np.zeros((k + 1, n))
    C[:k, :k] = np.eye(k)

    J1 = np.zeros(n)
    J1[0] = 1.0

    # Compute A^{H-11} once, then iterate
    Ah = np.linalg.matrix_power(A, H - 11)
    survey_row = np.zeros(n)
    for _ in range(12):                # h = H-11, H-10, ..., H
        survey_row += J1 @ Ah
        Ah = Ah @ A

    C[k, :] = survey_row
    return C

# ══════════════════════════════════════════════════════════════════════════════
# 4. KALMAN FILTER
# ══════════════════════════════════════════════════════════════════════════════

def kalman_filter(
    Y_endo:        np.ndarray,   # (T, k) endogenous
    focus:         np.ndarray,   # (T,) Focus series (% a.a.)
    A:             np.ndarray,   # (n, n) transition
    C:             np.ndarray,   # (k+1, n) observation — last row is c_survey
    mu0:           float,        # initial μ
    sigma2_mu:     float,        # variance of μ innovation
    sigma2_survey: float,        # survey measurement error variance
    p:             int,
) -> tuple[np.ndarray, float]:
    """
    Scalar Kalman filter for VAR-SE.

    y_t is treated as exactly observed — state components X[:k] and X[xbar_idx]
    are overridden directly from data at every step.  Only the Focus survey
    (a scalar linear function of the state) is treated as a stochastic
    observation, yielding a rank-1 (scalar) Kalman update for μ.

    Returns filtered states (n, T-p) and log-likelihood.
    """
    k = Y_endo.shape[1]
    n = A.shape[0]
    xbar_idx = k * p
    mu_idx   = k * p + 1
    T_eff    = len(Y_endo) - p

    # Process noise: only μ random walk
    Q = np.zeros((n, n))
    Q[mu_idx, mu_idx] = sigma2_mu

    # Survey observation row (scalar): last row of C
    c_survey = C[-1]   # shape (n,)

    # Initial state
    X = np.zeros(n)
    for lag in range(p):
        X[lag * k:(lag + 1) * k] = Y_endo[p - 1 - lag]
    X[xbar_idx] = 1.0
    X[mu_idx]   = mu0

    # Initial covariance — only μ is uncertain
    P = np.zeros((n, n))
    P[mu_idx, mu_idx] = 10.0    # diffuse prior on μ

    filtered = np.zeros((n, T_eff))
    log_lik  = 0.0

    for t in range(T_eff):
        # ── 1. Predict ────────────────────────────────────────────────────────
        X_p = A @ X
        P_p = A @ P @ A.T + Q

        # ── 2. Scalar survey update (Focus observation) ───────────────────────
        v   = focus[t + p] - c_survey @ X_p          # scalar
        s2  = c_survey @ P_p @ c_survey + sigma2_survey  # scalar innovation var
        K   = P_p @ c_survey / s2                    # (n,) Kalman gain

        X = X_p + K * v
        P = P_p - np.outer(K, K) * s2

        log_lik += -0.5 * (np.log(s2) + v * v / s2)

        # ── 3. Override exactly-observed VAR state components ─────────────────
        # y_t and all lags are observed without error; constant = 1 is exact.
        t_data = t + p          # index into Y_endo
        for lag in range(p):
            idx = t_data - lag
            if idx >= 0:
                X[lag * k:(lag + 1) * k] = Y_endo[idx]
        X[xbar_idx] = 1.0

        # Zero P rows/cols for all deterministic state components
        P[:k * p, :]       = 0.0
        P[:, :k * p]       = 0.0
        P[xbar_idx, :]     = 0.0
        P[:, xbar_idx]     = 0.0
        P = 0.5 * (P + P.T)   # keep symmetric

        filtered[:, t] = X

    return filtered, log_lik


def estimate_var_se(
    Y_endo:        np.ndarray,
    focus:         np.ndarray,
    intercept:     np.ndarray,
    A:             np.ndarray,
    C:             np.ndarray,
    p:             int,
    sigma2_survey: float = 0.25,   # calibrated: ±0.5 pp a.a. Focus dispersion
) -> tuple[float, float, np.ndarray, float]:
    """
    Kalman-based VAR-SE estimation with anchored survey equation.

    σ²_survey is calibrated externally (Kozicki-Tinsley convention) — the
    MLE for σ²_survey is degenerate in the scalar Kalman because the
    log-likelihood is monotonically increasing as σ²_survey → 0.

    The default 0.25 (pp a.a.)² corresponds to ±0.5 pp std across Focus
    forecasters at the 4yr horizon, consistent with Brazilian survey dispersion.

    Only σ²_μ (μ random-walk drift) is estimated by 1-D bounded MLE.

    Returns (sigma2_mu, sigma2_survey, filtered_states (n, T-p), log_lik).
    """
    from scipy.optimize import minimize_scalar

    mu0   = intercept[0]
    s2_srv = sigma2_survey

    def neg_ll(log_s2):
        _, ll = kalman_filter(Y_endo, focus, A, C, mu0,
                               np.exp(log_s2), s2_srv, p)
        return -ll

    res = minimize_scalar(neg_ll,
                          bounds=(np.log(1e-8), np.log(1e-2)),
                          method='bounded',
                          options={'xatol': 1e-7})
    s2_mu = np.exp(res.x)
    filtered, ll = kalman_filter(Y_endo, focus, A, C, mu0, s2_mu, s2_srv, p)
    return s2_mu, s2_srv, filtered, ll


def pin_mu(Y: np.ndarray, focus_t: float,
           A: np.ndarray, C: np.ndarray,
           k: int, p: int, n: int, xbar_idx: int, mu_idx: int) -> float:
    """
    Directly solve for μ_T so the VAR-SE H-step forecast equals focus_t.

    The survey constraint is an equality (BCB paper eq.):
        C_survey @ X_T = focus_t

    With X_T known except for μ_T, this is a single linear equation:
        (C_survey @ X_T|μ=0) + C_survey[mu_idx] * μ_T = focus_t
        → μ_T = (focus_t − C_survey @ X_T|μ=0) / C_survey[mu_idx]

    No Kalman filter, no MLE — exact, numerically stable.
    """
    survey_row = C[-1]               # C_survey is the last row of C
    c_mu = survey_row[mu_idx]        # scalar: sensitivity of forecast to μ

    # Build observed state with μ = 0
    X0 = np.zeros(n)
    for lag in range(p):
        X0[lag * k:(lag + 1) * k] = Y[-(1 + lag)]
    X0[xbar_idx] = 1.0              # x_bar is always 1

    mu_T = (focus_t - survey_row @ X0) / c_mu
    return mu_T


def compute_mu_path(Y: np.ndarray, focus: np.ndarray,
                    A: np.ndarray, C: np.ndarray,
                    k: int, p: int, n: int, xbar_idx: int, mu_idx: int) -> np.ndarray:
    """
    Compute the historical μ_t path by pinning at each date t.

    μ_t directly reflects Focus long-run expectations corrected for the
    current VAR state's contribution to the 37–48 month horizon.
    The attenuation of the current state over ~40 steps means μ_t is
    predominantly driven by focus_t, producing a smooth series.
    """
    T          = len(Y)
    survey_row = C[-1]
    c_mu       = survey_row[mu_idx]
    mu_path    = np.full(T, np.nan)

    for t in range(p, T):
        X0 = np.zeros(n)
        for lag in range(p):
            X0[lag * k:(lag + 1) * k] = Y[t - lag]
        X0[xbar_idx] = 1.0
        mu_path[t] = (focus[t] - survey_row @ X0) / c_mu

    return mu_path

# ══════════════════════════════════════════════════════════════════════════════
# 5. FORECASTING
# ══════════════════════════════════════════════════════════════════════════════

def build_initial_state(Y: np.ndarray, k: int, p: int, n: int,
                        xbar_idx: int, mu_idx: int, mu_val: float) -> np.ndarray:
    X0 = np.zeros(n)
    for lag in range(p):
        X0[lag*k:(lag+1)*k] = Y[-(1 + lag)]
    X0[xbar_idx] = 1.0
    X0[mu_idx]   = mu_val
    return X0


def var_forecast_monthly(A: np.ndarray, X0: np.ndarray,
                         h_max: int, k: int) -> np.ndarray:
    """Iterate state forward, return inflation (row 0) at each step."""
    fc = np.zeros(h_max)
    X  = X0.copy()
    for h in range(h_max):
        X      = A @ X
        fc[h]  = X[0]        # J1 @ X = X[0] (inflation)
    return fc


def monthly_to_12m_accum(fc_monthly: np.ndarray) -> np.ndarray:
    """
    Convert monthly % forecasts to 12m accumulated sum.
    Horizon h = 12 uses fc[0..11], h = 13 uses fc[1..12], etc.
    Returns NaN for h < 12.
    """
    h_max = len(fc_monthly)
    out   = np.full(h_max, np.nan)
    for h in range(12, h_max + 1):
        out[h - 1] = fc_monthly[h - 12: h].sum()
    return out



# ══════════════════════════════════════════════════════════════════════════════
# 6. OUT-OF-SAMPLE EVALUATION — TABLE 1
# ══════════════════════════════════════════════════════════════════════════════

def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray) -> tuple[float, float]:
    """
    Diebold-Mariano (1995) test: H0 = equal predictive accuracy.
    Uses Newey-West HAC variance with bandwidth = h^(1/3).
    Returns (DM stat, p-value) — two-sided.
    """
    from scipy import stats
    d   = e1**2 - e2**2
    n   = len(d)
    h   = int(np.ceil(n**(1/3)))
    d_m = d.mean()

    # Newey-West long-run variance
    lrv = np.var(d, ddof=1)
    for j in range(1, h + 1):
        gamma_j = np.cov(d[j:], d[:-j], ddof=1)[0, 1]
        lrv += 2 * (1 - j / (h + 1)) * gamma_j

    se   = np.sqrt(max(lrv, 1e-12) / n)
    dm   = d_m / se
    pval = 2 * (1 - stats.norm.cdf(abs(dm)))
    return dm, pval


def clark_west_test(e_bench: np.ndarray, e_varse: np.ndarray) -> tuple[float, float]:
    """
    Clark-West (2007): H0 = nested models have equal MSE.

    Adjusted series: f̂_t = e_bench² - e_varse² + (ŷ_bench - ŷ_varse)²
    where (ŷ_bench - ŷ_varse)² = (e_varse - e_bench)² = (e_bench - e_varse)²

    P-value is one-sided (right tail): reject when VAR-SE significantly better.

    Parameters
    ----------
    e_bench : forecast errors of the benchmark model (ARMA or VAR)
    e_varse : forecast errors of VAR-SE (the larger/unrestricted model)
    """
    from scipy import stats
    f_adj  = e_bench**2 - e_varse**2 + (e_bench - e_varse)**2
    n      = len(f_adj)
    t_stat = f_adj.mean() / (np.std(f_adj, ddof=1) / np.sqrt(n))
    pval   = 1 - stats.norm.cdf(t_stat)    # one-sided
    return t_stat, pval


def recursive_oos(
    data:          pd.DataFrame,
    horizons:      list[int],
    p:             int,
    H:             int,
    sigma2_mu:     float,
    sigma2_survey: float = 0.25,
    eval_start:    str   = '2013-06-01',
) -> dict:
    """
    Recursive (expanding window) OOS for ARMA, VAR, VAR-SE.

    Noise variances (sigma2_mu, sigma2_survey) are fixed at their
    full-sample estimates and held constant across recursive windows.
    Replicates BCB Table 1 columns.
    """
    Y_full    = data[ENDO_COLS].values
    focus_full = data['focus_48m'].values
    dates      = data.index

    eval_idx = data.index.get_indexer([eval_start], method='bfill')[0]
    T_total  = len(data)
    k        = Y_full.shape[1]
    n        = k * p + 2
    xbar_idx = k * p
    mu_idx   = k * p + 1

    results = {h: {'arma': [], 'var': [], 'var_se': [], 'actual': []}
               for h in horizons}

    print(f"\nRecursive OOS ({eval_start} onward)...")

    for t in range(eval_idx, T_total):
        Y_t      = Y_full[:t]
        focus_t  = focus_full[:t]

        if len(Y_t) < p + 20:
            continue

        # --- OLS VAR ---
        intercept, Phis, Sigma, _ = ols_var(Y_t, p)
        Phis, intercept = enforce_stability(Phis, k, intercept, Y_t.mean(axis=0))
        A_var   = build_transition(k, p, Phis, intercept, varse=False)
        A_varse = build_transition(k, p, Phis, intercept, varse=True)
        C       = build_obs_matrix(A_varse, k, n, H)

        # --- VAR-SE: Kalman filter ---
        mu0 = intercept[0]
        filtered_t, _ = kalman_filter(Y_t, focus_t, A_varse, C,
                                      mu0, sigma2_mu, sigma2_survey, p)
        mu_t_last = filtered_t[mu_idx, -1]

        # Build initial states
        X0_var    = build_initial_state(Y_t, k, p, n, xbar_idx, mu_idx, intercept[0])
        X0_var_se = build_initial_state(Y_t, k, p, n, xbar_idx, mu_idx, mu_t_last)

        # Monthly forecasts (max horizon needed)
        max_h = max(horizons)
        fc_var_m    = var_forecast_monthly(A_var,   X0_var,    max_h, k)
        fc_var_se_m = var_forecast_monthly(A_varse, X0_var_se, max_h, k)

        try:
            arma_mod = sm.tsa.ARIMA(Y_t[:, 0], order=(4, 0, 3)).fit()
            fc_arma_m = np.array(arma_mod.forecast(steps=max_h))
        except Exception:
            fc_arma_m = np.full(max_h, np.nan)

        for h in horizons:
            if t + h > T_total:
                continue
            # h-step-ahead realized = sum of next h monthly inflations
            actual_h = Y_full[t: t + h, 0].sum()

            fc_var_h    = fc_var_m[:h].sum()
            fc_var_se_h = fc_var_se_m[:h].sum()
            fc_arma_h   = fc_arma_m[:h].sum() if not np.isnan(fc_arma_m).any() else np.nan

            results[h]['actual'].append(actual_h)
            results[h]['var'].append(fc_var_h)
            results[h]['var_se'].append(fc_var_se_h)
            results[h]['arma'].append(fc_arma_h)

        if (t - eval_idx) % 10 == 0:
            print(f"  t = {dates[t].strftime('%Y-%m')} "
                  f"({t - eval_idx + 1}/{T_total - eval_idx})")

    return results


def compute_table1(oos_results: dict) -> pd.DataFrame:
    """
    Compute MSE and test statistics — BCB Table 1.

    Note on units: BCB Table 1 multiplies MSE by 10,000 and uses series in
    decimal proportions (/100). This code uses series in % a.m., so MSE values
    will be 10,000× larger than the paper's table. The DM/CW test p-values and
    significance stars are scale-invariant and directly comparable.
    """
    rows = []
    for h, d in sorted(oos_results.items()):
        actual  = np.array(d['actual'])
        f_arma  = np.array(d['arma'])
        f_var   = np.array(d['var'])
        f_varse = np.array(d['var_se'])

        valid = ~(np.isnan(f_arma) | np.isnan(f_var) | np.isnan(f_varse))
        a, fa, fv, fs = actual[valid], f_arma[valid], f_var[valid], f_varse[valid]

        e_arma  = a - fa
        e_var   = a - fv
        e_varse = a - fs

        mse_arma  = (e_arma**2).mean()
        mse_var   = (e_var**2).mean()
        mse_varse = (e_varse**2).mean()

        # DM test (ARMA vs VAR-SE) and (VAR vs VAR-SE)
        dm_arma,  pval_dm_arma  = diebold_mariano_test(e_arma,  e_varse)
        dm_var,   pval_dm_var   = diebold_mariano_test(e_var,   e_varse)

        # Clark-West (nested: VAR-SE nests VAR via μ_t; but used for ARMA comparison too)
        cw_arma, pval_cw_arma = clark_west_test(e_arma, e_varse)
        cw_var,  pval_cw_var  = clark_west_test(e_var,  e_varse)

        def stars(p):
            if p < 0.01: return '***'
            if p < 0.05: return '**'
            if p < 0.10: return '*'
            return ''

        rows.append({
            'Horizonte': h,
            'MSE_ARMA':  round(mse_arma  * 1e4, 3),
            'MSE_VAR':   round(mse_var   * 1e4, 3),
            'MSE_VARSE': round(mse_varse * 1e4, 3),
            'n_obs':     int(valid.sum()),
            'pDM_ARMA':  f"{pval_dm_arma:.3f}{stars(pval_dm_arma)}",
            'pDM_VAR':   f"{pval_dm_var:.3f}{stars(pval_dm_var)}",
            'pCW_ARMA':  f"{pval_cw_arma:.3f}{stars(pval_cw_arma)}",
            'pCW_VAR':   f"{pval_cw_var:.3f}{stars(pval_cw_var)}",
        })

    return pd.DataFrame(rows).set_index('Horizonte')

# ══════════════════════════════════════════════════════════════════════════════
# 7. PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_figure1(data, fc_arma_m, fc_var_m, fc_var_se_m, h_max=48):
    """
    Replicate BCB Gráfico 1: full-sample forecasts, acumulado 12 meses.

    The forecast lines are connected to the last observed 12m rolling sum:
    for h < 12, the rolling window blends the last (12-h) observed months with
    h forecast months, producing a smooth continuation.
    """
    pi_obs   = data['pi_total'].values        # monthly observed series
    pi_12m   = data['pi_total'].rolling(12).sum()
    last_d   = data.index[-1]
    fut      = pd.date_range(last_d, periods=h_max + 1, freq='ME')[1:]

    def blend_12m(fc_monthly):
        """12m rolling sum bridging observed → forecast."""
        tail = pi_obs[-11:]                    # last 11 observed months
        out  = np.empty(h_max)
        # h=1..11: (12-h) observed months + h forecast months
        for h in range(1, 12):
            out[h - 1] = tail[-(12 - h):].sum() + fc_monthly[:h].sum()
        # h=12..48: all-forecast 12m rolling window (no observed months left)
        for h in range(12, h_max + 1):
            out[h - 1] = fc_monthly[h - 12: h].sum()
        return out

    fc12_arma  = blend_12m(fc_arma_m)
    fc12_var   = blend_12m(fc_var_m)
    fc12_varse = blend_12m(fc_var_se_m)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pi_12m.index, pi_12m, 'k-', lw=1.5, label='IPCA (obs.)')
    ax.plot(fut, fc12_arma,   '--', color='steelblue',
            label=f"ARMA ({fc12_arma[-1]:.2f}%)")
    ax.plot(fut, fc12_var,    '--', color='orange',
            label=f"VAR ({fc12_var[-1]:.2f}%)")
    ax.plot(fut, fc12_varse,  '-',  color='seagreen', lw=2,
            label=f"VAR-SE ({fc12_varse[-1]:.2f}%)")

    focus_val = data['focus_48m'].iloc[-1]
    ax.axhline(focus_val, color='seagreen', ls=':', lw=0.8, alpha=0.6,
               label=f"Focus 48m ({focus_val:.2f}%)")

    ax.set_title('IPCA — Acumulado em 12 meses e projeções', fontsize=12)
    ax.set_ylabel('%')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    plt.tight_layout()
    return fig


def plot_figure2_3(data, intercept_ols,
                   k, p, n, xbar_idx, mu_idx,
                   sigma2_mu=1e-5, sigma2_survey=1e-6, h_max=48):
    """Replicate BCB Gráficos 2 and 3: recursive forecasts at 5 endpoints."""
    endpoints = ['2013-12', '2014-12', '2015-12', '2016-12', '2017-12']
    colors    = ['royalblue', 'orange', 'green', 'red', 'purple']

    pi_12m = data['pi_total'].rolling(12).sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax in axes:
        ax.plot(pi_12m.index, pi_12m, 'k-', lw=1.2, label='IPCA (obs.)')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))

    for ep_str, col in zip(endpoints, colors):
        ep_date = pd.Timestamp(ep_str)
        if ep_date not in data.index:
            ep_date = data.index[data.index.get_indexer([ep_date], method='ffill')[0]]

        sub = data.loc[:ep_str]
        Y_s = sub[ENDO_COLS].values
        focus_s = sub['focus_48m'].values

        # OLS VAR on subsample
        intr_s, Phis_s, _, _ = ols_var(Y_s, p)
        Phis_s, intr_s = enforce_stability(Phis_s, k, intr_s, Y_s.mean(axis=0))
        A_var_s   = build_transition(k, p, Phis_s, intr_s, varse=False)
        A_varse_s = build_transition(k, p, Phis_s, intr_s, varse=True)
        C_s = build_obs_matrix(A_varse_s, k, n, H)

        # VAR-SE: Kalman filter
        mu0_s = intr_s[0]
        filt_s, _ = kalman_filter(Y_s, focus_s, A_varse_s, C_s, mu0_s,
                                  sigma2_mu, sigma2_survey, p)
        mu_last_s = filt_s[mu_idx, -1]

        X0_var    = build_initial_state(Y_s, k, p, n, xbar_idx, mu_idx, intr_s[0])
        X0_var_se = build_initial_state(Y_s, k, p, n, xbar_idx, mu_idx, mu_last_s)

        fc_var_m    = var_forecast_monthly(A_var_s,   X0_var,    h_max, k)
        fc_var_se_m = var_forecast_monthly(A_varse_s, X0_var_se, h_max, k)

        # Blend observed tail with forecast for smooth continuation
        pi_tail = sub['pi_total'].values[-11:]

        def blend(fc_m):
            out = np.empty(h_max)
            for h in range(1, 12):
                out[h - 1] = pi_tail[-(12 - h):].sum() + fc_m[:h].sum()
            for h in range(12, h_max + 1):
                out[h - 1] = fc_m[h - 12: h].sum()
            return out

        fut = pd.date_range(ep_date, periods=h_max + 1, freq='ME')[1:]

        axes[0].plot(fut, blend(fc_var_m),    '--', color=col, lw=1.2,
                     label=ep_str[:7])
        axes[1].plot(fut, blend(fc_var_se_m), '--', color=col, lw=1.2,
                     label=ep_str[:7])

    axes[0].set_title('Projeções — VAR (irrestrito)', fontsize=11)
    axes[1].set_title('Projeções — VAR-SE', fontsize=11)
    axes[0].set_ylabel('%')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.set_xlim(pd.Timestamp('2012-01'), pd.Timestamp('2022-06'))
        ax.set_ylim(0, 11)
    plt.tight_layout()
    return fig


def plot_mu_path(data, filtered_states, intercept_ols, k, p):
    """Time-varying intercept μ_t vs Focus survey."""
    mu_idx     = k * p + 1
    dates_filt = data.index[p:]
    mu_path    = filtered_states[mu_idx, :]

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(dates_filt, mu_path, 'darkred', lw=1.5, label='μ_t (intercepto variável)')
    ax1.axhline(intercept_ols[0], color='k', ls='--', lw=1,
                label=f'OLS intercept ({intercept_ols[0]:.3f}%/m)')
    ax1.set_ylabel('% a.m.', color='darkred')
    ax1.legend(loc='upper left', fontsize=9)
    #ax1.set_ylim(0.25,0.4)

    ax2 = ax1.twinx()
    ax2.plot(data.index, data['focus_48m'], color='steelblue', alpha=0.6,
             lw=1.2, label='Focus 48m (% a.a.)')
    ax2.set_ylabel('Focus (% a.a.)', color='steelblue')
    ax2.legend(loc='upper right', fontsize=9)

    ax1.set_title('Intercepto variável da inflação μ_t vs. Expectativas Focus 48m', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # ── 1. Data ──────────────────────────────────────────────────────────────
    data = download_data(START, END)
    Y         = data[ENDO_COLS].values
    focus_arr = data['focus_48m'].values
    k = Y.shape[1]
    n = k * P_VAR + 2
    xbar_idx = k * P_VAR
    mu_idx   = k * P_VAR + 1

    # ── 2. OLS VAR(2) ────────────────────────────────────────────────────────
    print("\nEstimating OLS VAR(2)...")
    intercept_ols, Phis_ols, Sigma_ols, resid_ols = ols_var(Y, P_VAR)
    Phis_ols, intercept_ols = enforce_stability(Phis_ols, k, intercept_ols, Y.mean(axis=0))

    eig = np.abs(np.linalg.eigvals(
        build_transition(k, P_VAR, Phis_ols, intercept_ols)[:k*P_VAR, :k*P_VAR]
    ))
    print(f"  Companion spectral radius: {eig.max():.4f} "
          f"({'OK' if eig.max() < 1 else 'UNSTABLE'})")

    # ── Diagnostics: long-run levels ─────────────────────────────────────────
    Y_mean   = Y.mean(axis=0)
    print(f"\n  Sample means (should equal VAR long-run after correction):")
    for col, mu in zip(ENDO_COLS, Y_mean):
        print(f"    {col}: {mu:.4f}  ({mu*12:.2f}% a.a. if monthly %)")
    print(f"  Focus 48m at end of sample: {data['focus_48m'].iloc[-1]:.2f}% a.a.")
    print(f"  Current state (last obs):")
    for col, v in zip(ENDO_COLS, Y[-1]):
        print(f"    {col}: {v:.4f}")

    # ── 3. State-space matrices ───────────────────────────────────────────────
    A_var   = build_transition(k, P_VAR, Phis_ols, intercept_ols, varse=False)
    A_varse = build_transition(k, P_VAR, Phis_ols, intercept_ols, varse=True)
    C_fixed = build_obs_matrix(A_varse, k, n, H)   # survey constraint uses VAR-SE companion
    print(f"  Survey row norm: {np.linalg.norm(C_fixed[k, :]):.4f}")

    # ── 4. Kalman filter — anchored VAR-SE ───────────────────────────────────
    print("\nEstimating σ²_μ via anchored Kalman MLE...")
    s2_mu, s2_survey, filtered_states, ll = estimate_var_se(
        Y, focus_arr, intercept_ols, A_varse, C_fixed, P_VAR
    )
    mu_filtered = filtered_states[mu_idx, :]
    print(f"  σ²_μ (MLE)           = {s2_mu:.6f}")
    print(f"  σ²_survey (MLE)      = {s2_survey:.6f}  (std = {s2_survey**0.5:.4f} pp a.a.)")
    print(f"  log-likelihood       = {ll:.2f}")
    print(f"  μ_T  (last filtered) = {mu_filtered[-1]:.4f}%/m")
    print(f"  μ_0  (OLS intercept) = {intercept_ols[0]:.4f}%/m")
    f48 = data['focus_48m'].iloc[-1]
    f_monthly_compound = ((1 + f48/100)**(1/12) - 1) * 100
    print(f"  Focus 48m (compound) = {f_monthly_compound:.4f}%/m  "
          f"[linear: {f48/12:.4f}%/m]")

    # ── 5. Full-sample forecasts ──────────────────────────────────────────────
    print("\nComputing full-sample forecasts...")
    X0_var    = build_initial_state(Y, k, P_VAR, n, xbar_idx, mu_idx, intercept_ols[0])
    X0_var_se = build_initial_state(Y, k, P_VAR, n, xbar_idx, mu_idx, mu_filtered[-1])

    fc_var_m    = var_forecast_monthly(A_var,   X0_var,    H, k)
    fc_var_se_m = var_forecast_monthly(A_varse, X0_var_se, H, k)

    print("  Fitting ARMA(4,3)...")
    arma_mod   = sm.tsa.ARIMA(Y[:, 0], order=(4, 0, 3)).fit()
    fc_arma_m  = np.array(arma_mod.forecast(steps=H))

    # 12m accumulated for summary
    fc_arma_12m  = monthly_to_12m_accum(fc_arma_m)
    fc_var_12m   = monthly_to_12m_accum(fc_var_m)
    fc_var_se_12m = monthly_to_12m_accum(fc_var_se_m)

    print(f"\n{'─'*45}")
    print(f"  Long-run forecasts (12m accumulated, h=48):")
    print(f"  ARMA   : {fc_arma_12m[-1]:.2f}%")
    print(f"  VAR    : {fc_var_12m[-1]:.2f}%")
    print(f"  VAR-SE : {fc_var_se_12m[-1]:.2f}%")
    print(f"  Focus  : {data['focus_48m'].iloc[-1]:.2f}%")
    print(f"{'─'*45}")

    # ── 6. Plots ─────────────────────────────────────────────────────────────
    fig1 = plot_figure1(data, fc_arma_m, fc_var_m, fc_var_se_m)
    fig1.savefig('var_se_fig1_forecasts.png', dpi=150, bbox_inches='tight')

    fig_mu = plot_mu_path(data, filtered_states, intercept_ols, k, P_VAR)
    fig_mu.savefig('var_se_mu_path.png', dpi=150, bbox_inches='tight')

    print("\nComputing recursive subsample forecasts (Figs 2 & 3)...")
    fig23 = plot_figure2_3(data, intercept_ols, k, P_VAR, n, xbar_idx, mu_idx,
                           sigma2_mu=s2_mu, sigma2_survey=s2_survey)
    fig23.savefig('var_se_fig2_3_recursive.png', dpi=150, bbox_inches='tight')

    plt.show()

    # ── 7. OOS evaluation (slow — set RUN_OOS = True) ────────────────────────
    RUN_OOS = True   # set True to replicate Table 1 (takes ~5 min)

    if RUN_OOS:
        HORIZONS = [1, 3, 6, 12, 24, 36]
        oos = recursive_oos(data, HORIZONS, P_VAR, H,
                            sigma2_mu=s2_mu, sigma2_survey=s2_survey,
                            eval_start='2013-06-01')
        table1 = compute_table1(oos)
        print("\nTabela 1 — EQM (×10⁴) e testes de previsão:")
        print(table1.to_string())
        table1.to_excel('var_se_table1.xlsx')

    print("\nDone. Plots saved to var_se_fig*.png")

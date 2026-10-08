"""Core numerical routines for the regime-switching Black-Scholes bachelor thesis.

The module contains the main HMM, Black-Scholes, PDE, pricing, evaluation,
and plotting utilities used throughout the project.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import norm, t as student_t
from scipy.linalg import logm, lu_factor, lu_solve
from scipy.optimize import brentq
try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:  # Allows non-HMM utilities to be imported before installing requirements.
    GaussianHMM = None

try:
    from IPython.display import display
except ImportError:  # Plain-text fallback outside Jupyter/IPython.
    display = print

try:
    from mpl_bsic.apply_bsic_style import BSIC_COLORS, DEFAULT_COLOR_CYCLE, DEFAULT_TITLE_STYLE
except ImportError:
    BSIC_COLORS = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    DEFAULT_COLOR_CYCLE = plt.rcParams["axes.prop_cycle"]
    DEFAULT_TITLE_STYLE = {}

TRADING_DAYS_PER_YEAR = 252
OPTION_TYPE = "call"
ATM_MATURITY_DAYS = 30

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def apply_bsic_colors_only(fig, ax):
    """Apply the BSIC palette when available, without adding logos or source footers."""
    plt.rcParams["axes.prop_cycle"] = DEFAULT_COLOR_CYCLE
    if isinstance(ax, np.ndarray):
        axes = ax.ravel()
    elif isinstance(ax, (list, tuple)):
        axes = ax
    else:
        axes = [ax]

    for axis in axes:
        axis.set_prop_cycle(DEFAULT_COLOR_CYCLE)
        if axis.get_title():
            axis.set_title(axis.get_title(), **DEFAULT_TITLE_STYLE)
        if BSIC_COLORS:
            for line, color in zip(axis.get_lines(), BSIC_COLORS):
                line.set_color(color)

def find_data_file(filename):
    """Return the first existing location for an uploaded/project data file."""
    candidates = [
        PROJECT_ROOT / "data" / "processed" / filename,
        PROJECT_ROOT / "data" / "raw" / filename,
        Path("dataset") / filename,
        Path(filename),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {filename}. Checked: {candidates}")

def hmm_parameter_count(n_states: int) -> int:
    # Free parameters in a univariate Gaussian HMM:
    # (n_states - 1) for initial state probabilities
    # n_states * (n_states - 1) for the transition matrix
    # n_states means
    # n_states variances
    return n_states ** 2 + 2 * n_states - 1

def sort_result_by_volatility(result):
    order = np.argsort(result["vols"])
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(len(order))

    sorted_result = dict(result)
    sorted_result["state_order_by_volatility"] = order
    sorted_result["means"] = result["means"][order]
    sorted_result["vols"] = result["vols"][order]
    sorted_result["init_probs"] = result["init_probs"][order]
    sorted_result["transmat"] = result["transmat"][np.ix_(order, order)]
    sorted_result["posterior"] = result["posterior"][:, order]
    sorted_result["states"] = inverse_order[result["states"]]
    return sorted_result

def fit_gaussian_hmm(returns, n_states, random_state=123, n_iter=500, tol=1e-8, n_init=20):
    returns = np.asarray(returns, dtype=float).reshape(-1)
    X = returns.reshape(-1, 1)

    best_model = None
    best_log_likelihood = -np.inf

    if GaussianHMM is None:
        raise ImportError("hmmlearn is required to fit GaussianHMM models. Install requirements.txt.")

    for seed in range(random_state, random_state + n_init):
        candidate = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=n_iter,
            tol=tol,
            random_state=seed,
            implementation="scaling",
        )
        candidate.fit(X)
        candidate_log_likelihood = float(candidate.score(X))

        if candidate_log_likelihood > best_log_likelihood:
            best_model = candidate
            best_log_likelihood = candidate_log_likelihood

    means = best_model.means_.reshape(-1)
    covars = np.asarray(best_model.covars_).reshape(n_states, -1)[:, 0]
    posterior = best_model.predict_proba(X)
    states = best_model.predict(X)

    result = {
        "n_states": n_states,
        "log_likelihood": float(best_log_likelihood),
        "init_probs": np.asarray(best_model.startprob_).reshape(-1),
        "transmat": np.asarray(best_model.transmat_),
        "means": means,
        "vols": np.sqrt(np.maximum(covars, 1e-12)),
        "posterior": posterior,
        "states": states,
    }

    result = sort_result_by_volatility(result)
    T = len(returns)
    k = hmm_parameter_count(n_states)
    result["aic"] = 2 * k - 2 * result["log_likelihood"]
    result["bic"] = k * np.log(T) - 2 * result["log_likelihood"]
    result["avg_posterior_confidence"] = float(result["posterior"].max(axis=1).mean())
    result["sample_size"] = T
    return result

def state_summary_table(result):
    transition_diag = np.diag(result["transmat"])
    expected_duration = np.where(np.isclose(1 - transition_diag, 0), np.inf, 1.0 / (1.0 - transition_diag))
    counts = np.bincount(result["states"], minlength=result["n_states"])
    shares = counts / counts.sum()
    summary = pd.DataFrame({
        "state": np.arange(result["n_states"]),
        "mean_daily_return": result["means"],
        "daily_volatility": result["vols"],
        "annualized_volatility": result["vols"] * np.sqrt(TRADING_DAYS_PER_YEAR),
        "count": counts,
        "sample_share": shares,
        "expected_duration_days": expected_duration,
    })
    return summary

def transition_matrix_table(result):
    state_labels = [f"state_{i}" for i in range(result["n_states"])]
    return pd.DataFrame(result["transmat"], index=state_labels, columns=state_labels)

def posterior_preview_table(result, dates, n_rows=5):
    posterior_df = pd.DataFrame(
        result["posterior"],
        columns=[f"posterior_state_{i}" for i in range(result["n_states"])],
    )
    posterior_df.insert(0, "date", pd.Series(dates).reset_index(drop=True))
    posterior_df.insert(1, "most_likely_state", result["states"])
    return posterior_df.head(n_rows)

def plot_hmm_diagnostics(dates, returns, result):
    state_cmap = ListedColormap(BSIC_COLORS[:result["n_states"]])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, returns, linewidth=0.8, alpha=0.6, label="log return")
    scatter = ax.scatter(dates, returns, c=result["states"], s=12, cmap=state_cmap)
    ax.set_title(f"{result['n_states']}-state HMM: returns coloured by inferred state")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log return")
    handles, _ = scatter.legend_elements()
    ax.legend(handles, [f"State {i}" for i in range(result['n_states'])], loc="upper right")
    apply_bsic_colors_only(fig, ax)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 4))
    for i in range(result["n_states"]):
        ax.plot(dates, result["posterior"][:, i], linewidth=1.0, label=f"P(state={i} | data)")
    ax.set_title(f"{result['n_states']}-state HMM: posterior state probabilities")
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.02, 1.02)
    apply_bsic_colors_only(fig, ax)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def volatility_labels(n_states):
    if n_states == 2:
        return ["low-volatility", "high-volatility"]
    if n_states == 3:
        return ["low-volatility", "medium-volatility", "high-volatility"]
    return [f"regime_{i}" for i in range(n_states)]

def discrete_transition_to_generator(transition_matrix, dt_years=1 / TRADING_DAYS_PER_YEAR):
    generator = np.real_if_close(logm(transition_matrix) / dt_years)
    generator = np.asarray(generator, dtype=float)
    return generator

def labeled_matrix(matrix, labels):
    return pd.DataFrame(matrix, index=labels, columns=labels)

def _as_2d_returns(return_series):
    """
    Convert a pandas Series or array of returns into the 2D shape expected by hmmlearn.
    """
    arr = np.asarray(return_series, dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr.reshape(-1, 1)

def _normalize_prob_vector(p, eps=1e-15):
    """
    Normalize a probability vector safely.
    """
    p = np.asarray(p, dtype=float)
    p = np.maximum(p, eps)
    return p / p.sum()

def _gaussian_emission_likelihood(x, means, variances, eps=1e-12):
    """
    Gaussian emission likelihood for each HMM state.
    """
    means = np.asarray(means, dtype=float).reshape(-1)
    variances = np.asarray(variances, dtype=float).reshape(-1)
    std = np.sqrt(np.maximum(variances, eps))
    return norm.pdf(x, loc=means, scale=std)

def filtered_probabilities_no_lookahead(model, returns, initial_prob=None):
    """
    Sequentially compute filtered probabilities for a fitted GaussianHMM.

    Parameters
    ----------
    model : hmmlearn.hmm.GaussianHMM
        Fitted Gaussian HMM.
    returns : array-like
        Return observations in chronological order.
    initial_prob : array-like or None
        Initial probability vector. If None, model.startprob_ is used.

    Returns
    -------
    filtered_probs : np.ndarray
        Array of shape (n_obs, n_states). Row t is P(z_t | r_1, ..., r_t).
    predicted_probs : np.ndarray
        Array of shape (n_obs, n_states). Row t is P(z_t | r_1, ..., r_{t-1}).
        The first row uses the initial probability.
    """
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]

    n_obs = len(x)
    n_states = model.n_components

    if n_obs == 0:
        return np.empty((0, n_states)), np.empty((0, n_states))

    P = np.asarray(model.transmat_, dtype=float)
    means = np.asarray(model.means_, dtype=float).reshape(-1)
    variances = np.asarray(model.covars_, dtype=float).reshape(n_states, -1)[:, 0]

    if initial_prob is None:
        p_prev_filt = _normalize_prob_vector(model.startprob_)
    else:
        p_prev_filt = _normalize_prob_vector(initial_prob)

    filtered_probs = np.zeros((n_obs, n_states))
    predicted_probs = np.zeros((n_obs, n_states))

    for i, obs in enumerate(x):
        if i == 0:
            p_pred = p_prev_filt.copy()
        else:
            p_pred = _normalize_prob_vector(p_prev_filt @ P)

        likelihood = _gaussian_emission_likelihood(obs, means, variances)
        p_filt = _normalize_prob_vector(p_pred * likelihood)

        predicted_probs[i, :] = p_pred
        filtered_probs[i, :] = p_filt

        p_prev_filt = p_filt

    return filtered_probs, predicted_probs

def filtered_probabilities_no_lookahead_params(
    returns,
    startprob,
    transmat,
    means,
    variances,
):
    """
    Sequential filtering using explicitly supplied HMM parameters.

    This is useful after sorting states by volatility without mutating the
    fitted hmmlearn model object.
    """
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]

    startprob = _normalize_prob_vector(startprob)
    transmat = np.asarray(transmat, dtype=float)
    means = np.asarray(means, dtype=float).reshape(-1)
    variances = np.asarray(variances, dtype=float).reshape(-1)

    n_obs = len(x)
    n_states = len(startprob)

    filtered_probs = np.zeros((n_obs, n_states))
    predicted_probs = np.zeros((n_obs, n_states))

    p_prev_filt = startprob.copy()

    for i, obs in enumerate(x):
        if i == 0:
            p_pred = p_prev_filt.copy()
        else:
            p_pred = one_step_ahead_predicted_probability(p_prev_filt, transmat)

        likelihood = _gaussian_emission_likelihood(obs, means, variances)
        p_filt = _normalize_prob_vector(p_pred * likelihood)

        predicted_probs[i, :] = p_pred
        filtered_probs[i, :] = p_filt

        p_prev_filt = p_filt

    return filtered_probs, predicted_probs

def one_step_ahead_predicted_probability(filtered_prob, transition_matrix):
    """
    Compute one-step-ahead predicted probabilities p_tilde = p_filtered @ P.
    """
    filtered_prob = _normalize_prob_vector(filtered_prob)
    P = np.asarray(transition_matrix, dtype=float)
    return _normalize_prob_vector(filtered_prob @ P)

def fit_walkforward_hmm_snapshot(
    train_df,
    return_col="log_return",
    n_components=2,
    trading_days_per_year=TRADING_DAYS_PER_YEAR,
    random_state=42,
    n_iter=1000,
    tol=1e-6,
    min_obs=252,
):
    """
    Fit a Gaussian HMM on a training window and return all quantities
    sorted from low volatility to high volatility.

    Works with both 2-state and 3-state specifications.
    """
    train_returns = pd.to_numeric(train_df[return_col], errors="coerce").dropna()

    if len(train_returns) < min_obs:
        raise ValueError(f"Not enough observations to fit HMM: {len(train_returns)} < {min_obs}")

    X_train = train_returns.values.reshape(-1, 1)

    model = GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
        implementation="scaling",
    )

    model.fit(X_train)

    raw_means = np.asarray(model.means_, dtype=float).reshape(n_components)
    raw_vars = np.asarray(model.covars_, dtype=float).reshape(n_components, -1)[:, 0]
    raw_sigmas = np.sqrt(np.maximum(raw_vars, 1e-12))

    order = np.argsort(raw_sigmas)

    means = raw_means[order]
    daily_vars = raw_vars[order]
    daily_sigmas = raw_sigmas[order]
    annual_sigmas = daily_sigmas * np.sqrt(trading_days_per_year)

    startprob = np.asarray(model.startprob_, dtype=float)[order]
    startprob = _normalize_prob_vector(startprob)

    P = np.asarray(model.transmat_, dtype=float)[np.ix_(order, order)]
    P = np.maximum(P, 1e-12)
    P = P / P.sum(axis=1, keepdims=True)

    Q_raw = logm(P) * trading_days_per_year
    Q_raw = np.real_if_close(Q_raw).real
    Q_raw = np.asarray(Q_raw, dtype=float)

    if Q_raw.shape != (n_components, n_components) or not np.all(np.isfinite(Q_raw)):
        raise ValueError("Continuous-time generator could not be computed.")

    # Clean generator: off-diagonal non-negative, diagonal = minus row sum
    Q = Q_raw.copy()

    for i in range(n_components):
        for j in range(n_components):
            if i != j:
                Q[i, j] = max(float(Q[i, j]), 0.0)

    for i in range(n_components):
        Q[i, i] = -np.sum(Q[i, np.arange(n_components) != i])

    filtered_probs, predicted_probs = filtered_probabilities_no_lookahead_params(
        returns=train_returns.values,
        startprob=startprob,
        transmat=P,
        means=means,
        variances=daily_vars,
    )

    latest_filtered_prob = filtered_probs[-1, :]

    result = {
        "model": model,
        "n_components": n_components,
        "state_order_by_volatility": order,
        "means": means,
        "variances": daily_vars,
        "startprob": startprob,
        "annual_sigmas": annual_sigmas,
        "daily_sigmas": daily_sigmas,
        "P": P,
        "Q": Q,
        "latest_filtered_prob": latest_filtered_prob,
        "latest_predicted_prob": one_step_ahead_predicted_probability(latest_filtered_prob, P),
        "n_train_obs": len(train_returns),
        "train_start": train_df["date"].min(),
        "train_end": train_df["date"].max(),
        "converged": getattr(getattr(model, "monitor_", None), "converged", np.nan),
        "log_likelihood": float(model.score(X_train)),
    }

    # Human-readable aliases
    result["sigma_low"] = float(annual_sigmas[0])
    result["daily_sigma_low"] = float(daily_sigmas[0])

    if n_components == 2:
        result["sigma_high"] = float(annual_sigmas[1])
        result["daily_sigma_high"] = float(daily_sigmas[1])
        result["q_lh"] = float(Q[0, 1])
        result["q_hl"] = float(Q[1, 0])

    elif n_components == 3:
        result["sigma_medium"] = float(annual_sigmas[1])
        result["sigma_high"] = float(annual_sigmas[2])

        result["daily_sigma_medium"] = float(daily_sigmas[1])
        result["daily_sigma_high"] = float(daily_sigmas[2])

        result["q_lm"] = float(Q[0, 1])
        result["q_lh"] = float(Q[0, 2])
        result["q_ml"] = float(Q[1, 0])
        result["q_mh"] = float(Q[1, 2])
        result["q_hl"] = float(Q[2, 0])
        result["q_hm"] = float(Q[2, 1])

    else:
        raise ValueError("This notebook currently supports only n_components=2 or n_components=3.")

    return result

def compute_training_start_date(
    month_start,
    window_type="expanding",
    initial_train_start="2015-01-05",
    rolling_window=5,
    rolling_window_unit="years",
    df=None,
    date_col="date",
):
    """
    Compute the start date of the training window.

    Parameters
    ----------
    month_start : pd.Timestamp
        First day of the pricing month.
    window_type : {"expanding", "rolling"}
        Expanding uses all available data from initial_train_start.
        Rolling uses a finite lookback window.
    initial_train_start : str or pd.Timestamp
        Start date for expanding windows.
    rolling_window : int
        Length of the rolling window.
    rolling_window_unit : {"days", "months", "years", "observations", "trading_days"}
        Unit of the rolling window.
        - "days": calendar days
        - "months": calendar months
        - "years": calendar years
        - "observations" or "trading_days": last N available return observations
    df : pd.DataFrame
        Required only if rolling_window_unit is "observations" or "trading_days".
    date_col : str
        Date column name.

    Returns
    -------
    train_start : pd.Timestamp
    """
    month_start = pd.Timestamp(month_start)
    initial_train_start = pd.Timestamp(initial_train_start)

    if window_type == "expanding":
        return initial_train_start

    if window_type != "rolling":
        raise ValueError("window_type must be either 'expanding' or 'rolling'.")

    rolling_window_unit = str(rolling_window_unit).lower().strip()

    if rolling_window <= 0:
        raise ValueError("rolling_window must be positive.")

    if rolling_window_unit in ["day", "days", "calendar_days"]:
        return month_start - pd.DateOffset(days=rolling_window)

    if rolling_window_unit in ["month", "months", "calendar_months"]:
        return month_start - pd.DateOffset(months=rolling_window)

    if rolling_window_unit in ["year", "years", "calendar_years"]:
        return month_start - pd.DateOffset(years=rolling_window)

    if rolling_window_unit in ["observation", "observations", "trading_day", "trading_days"]:
        if df is None:
            raise ValueError(
                "df must be provided when rolling_window_unit is 'observations' or 'trading_days'."
            )

        train_end = month_start - pd.Timedelta(days=1)

        available_train_dates = (
            df.loc[df[date_col] <= train_end, date_col]
            .dropna()
            .sort_values()
            .drop_duplicates()
            .reset_index(drop=True)
        )

        if len(available_train_dates) == 0:
            return initial_train_start

        if len(available_train_dates) < rolling_window:
            return available_train_dates.iloc[0]

        return available_train_dates.iloc[-rolling_window]

    raise ValueError(
        "rolling_window_unit must be one of: "
        "'days', 'months', 'years', 'observations', 'trading_days'."
    )

def monthly_walkforward_regime_estimates(
    model_df,
    start_date="2020-01-01",
    end_date="2024-12-31",
    initial_train_start="2015-01-05",
    date_col="date",
    return_col="log_return",
    n_components=2,
    min_train_obs=252,
    random_state=42,
    verbose=True,
    window_type="expanding",
    rolling_window=5,
    rolling_window_unit="years",
):
    """
    Monthly walk-forward HMM estimation and daily filtering.

    Works with both 2-state and 3-state HMMs.
    """
    df = model_df.copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[return_col] = pd.to_numeric(df[return_col], errors="coerce")

    df = (
        df.dropna(subset=[date_col, return_col])
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    initial_train_start = pd.Timestamp(initial_train_start)

    if window_type not in ["expanding", "rolling"]:
        raise ValueError("window_type must be either 'expanding' or 'rolling'.")

    if n_components not in [2, 3]:
        raise ValueError("n_components must be either 2 or 3.")

    if df.empty:
        raise ValueError("After cleaning, model_df has no valid date/log_return rows.")

    available_start = max(start_date, df[date_col].min())
    available_end = min(end_date, df[date_col].max())

    if available_start > available_end:
        raise ValueError(
            "No overlap between requested pricing period and model_df dates. "
            f"Requested {start_date.date()} to {end_date.date()}, "
            f"available {df[date_col].min().date()} to {df[date_col].max().date()}."
        )

    if verbose:
        print("Cleaned model_df date range:", df[date_col].min(), "to", df[date_col].max())
        print("Walk-forward pricing range:", available_start, "to", available_end)
        print("Cleaned model_df rows:", len(df))
        print("Training window type:", window_type)
        print("Number of HMM states:", n_components)

        if window_type == "rolling":
            print("Rolling training window:", rolling_window, rolling_window_unit)

    month_starts = pd.date_range(
        available_start.to_period("M").start_time,
        available_end.to_period("M").start_time,
        freq="MS",
    )

    all_rows = []
    snapshots = {}
    failed_months = []

    for month_start in month_starts:
        month_end = month_start + pd.offsets.MonthEnd(0)
        train_end = month_start - pd.Timedelta(days=1)

        train_start = compute_training_start_date(
            month_start=month_start,
            window_type=window_type,
            initial_train_start=initial_train_start,
            rolling_window=rolling_window,
            rolling_window_unit=rolling_window_unit,
            df=df,
            date_col=date_col,
        )

        train_df = df.loc[
            (df[date_col] >= train_start)
            & (df[date_col] <= train_end)
        ].copy()

        month_df = df.loc[
            (df[date_col] >= month_start)
            & (df[date_col] <= month_end)
            & (df[date_col] >= available_start)
            & (df[date_col] <= available_end)
        ].copy()

        if month_df.empty:
            failed_months.append(
                {
                    "month_start": month_start,
                    "reason": "No observations in pricing month",
                    "train_start": train_start,
                    "train_end": train_end,
                    "n_train_obs": len(train_df),
                    "n_month_obs": len(month_df),
                    "window_type": window_type,
                    "rolling_window": rolling_window if window_type == "rolling" else np.nan,
                    "rolling_window_unit": rolling_window_unit if window_type == "rolling" else np.nan,
                    "n_components": n_components,
                }
            )
            continue

        if len(train_df) < min_train_obs:
            failed_months.append(
                {
                    "month_start": month_start,
                    "reason": f"Too few training observations: {len(train_df)} < {min_train_obs}",
                    "train_start": train_start,
                    "train_end": train_end,
                    "n_train_obs": len(train_df),
                    "n_month_obs": len(month_df),
                    "window_type": window_type,
                    "rolling_window": rolling_window if window_type == "rolling" else np.nan,
                    "rolling_window_unit": rolling_window_unit if window_type == "rolling" else np.nan,
                    "n_components": n_components,
                }
            )
            continue

        try:
            snapshot = fit_walkforward_hmm_snapshot(
                train_df=train_df,
                return_col=return_col,
                n_components=n_components,
                random_state=random_state,
                min_obs=min_train_obs,
            )

        except Exception as exc:
            failed_months.append(
                {
                    "month_start": month_start,
                    "reason": f"HMM fit failed: {repr(exc)}",
                    "train_start": train_start,
                    "train_end": train_end,
                    "n_train_obs": len(train_df),
                    "n_month_obs": len(month_df),
                    "window_type": window_type,
                    "rolling_window": rolling_window if window_type == "rolling" else np.nan,
                    "rolling_window_unit": rolling_window_unit if window_type == "rolling" else np.nan,
                    "n_components": n_components,
                }
            )

            if verbose:
                print(f"Skipping {month_start.date()} because HMM fit failed: {repr(exc)}")

            continue

        P = snapshot["P"]
        Q = snapshot["Q"]
        means = snapshot["means"]
        variances = snapshot["variances"]

        p_prev_filt = _normalize_prob_vector(snapshot["latest_filtered_prob"])

        month_rows = []

        for _, row in month_df.iterrows():
            obs_date = row[date_col]
            obs_return = float(row[return_col])

            p_pred = one_step_ahead_predicted_probability(p_prev_filt, P)

            likelihood = _gaussian_emission_likelihood(obs_return, means, variances)
            p_filt = _normalize_prob_vector(p_pred * likelihood)

            row_dict = {
                "date": obs_date,
                "month_start": month_start,
                "train_start": train_start,
                "train_end": train_end,
                "n_train_obs": snapshot["n_train_obs"],
                "hmm_converged": snapshot["converged"],
                "hmm_log_likelihood": snapshot["log_likelihood"],
                "window_type": window_type,
                "rolling_window": rolling_window if window_type == "rolling" else np.nan,
                "rolling_window_unit": rolling_window_unit if window_type == "rolling" else np.nan,
                "n_components": n_components,
            }

            for k in range(n_components):
                row_dict[f"sigma_state_{k}"] = float(snapshot["annual_sigmas"][k])
                row_dict[f"daily_sigma_state_{k}"] = float(snapshot["daily_sigmas"][k])
                row_dict[f"p_state_{k}_pred"] = float(p_pred[k])
                row_dict[f"p_state_{k}_filt"] = float(p_filt[k])

            for i in range(n_components):
                for j in range(n_components):
                    row_dict[f"P_{i}{j}"] = float(P[i, j])
                    row_dict[f"Q_{i}{j}"] = float(Q[i, j])

            row_dict["sigma_low"] = float(snapshot["annual_sigmas"][0])
            row_dict["daily_sigma_low"] = float(snapshot["daily_sigmas"][0])
            row_dict["p_low_pred"] = float(p_pred[0])
            row_dict["p_low_filt"] = float(p_filt[0])

            if n_components == 2:
                row_dict["sigma_high"] = float(snapshot["annual_sigmas"][1])
                row_dict["daily_sigma_high"] = float(snapshot["daily_sigmas"][1])
                row_dict["p_high_pred"] = float(p_pred[1])
                row_dict["p_high_filt"] = float(p_filt[1])

                row_dict["q_lh"] = float(Q[0, 1])
                row_dict["q_hl"] = float(Q[1, 0])

                row_dict["P_LL"] = float(P[0, 0])
                row_dict["P_LH"] = float(P[0, 1])
                row_dict["P_HL"] = float(P[1, 0])
                row_dict["P_HH"] = float(P[1, 1])

            elif n_components == 3:
                row_dict["sigma_medium"] = float(snapshot["annual_sigmas"][1])
                row_dict["sigma_high"] = float(snapshot["annual_sigmas"][2])

                row_dict["daily_sigma_medium"] = float(snapshot["daily_sigmas"][1])
                row_dict["daily_sigma_high"] = float(snapshot["daily_sigmas"][2])

                row_dict["p_medium_pred"] = float(p_pred[1])
                row_dict["p_high_pred"] = float(p_pred[2])

                row_dict["p_medium_filt"] = float(p_filt[1])
                row_dict["p_high_filt"] = float(p_filt[2])

                row_dict["q_lm"] = float(Q[0, 1])
                row_dict["q_lh"] = float(Q[0, 2])
                row_dict["q_ml"] = float(Q[1, 0])
                row_dict["q_mh"] = float(Q[1, 2])
                row_dict["q_hl"] = float(Q[2, 0])
                row_dict["q_hm"] = float(Q[2, 1])

            month_rows.append(row_dict)
            p_prev_filt = p_filt

        month_result = pd.DataFrame(month_rows)
        all_rows.append(month_result)
        snapshots[month_start] = snapshot

    failed_months_df = pd.DataFrame(failed_months)

    if not all_rows:
        display(failed_months_df)
        raise ValueError(
            "No walk-forward regime estimates were produced. "
            "See failed_months_df above for the reason each month was skipped."
        )

    wf_df = pd.concat(all_rows, ignore_index=True)
    wf_df = wf_df.sort_values("date").reset_index(drop=True)

    return wf_df, snapshots, failed_months_df

def get_existing_columns(df, columns):
    return [col for col in columns if col in df.columns]

def get_regime_probability_columns(df, prob_type="pred"):
    ordered_cols = [
        f"p_low_{prob_type}",
        f"p_medium_{prob_type}",
        f"p_high_{prob_type}",
    ]

    existing_ordered = [col for col in ordered_cols if col in df.columns]

    if existing_ordered:
        return existing_ordered

    return sorted(
        [
            col for col in df.columns
            if col.startswith("p_state_") and col.endswith(f"_{prob_type}")
        ]
    )

def get_regime_sigma_columns(df):
    ordered_cols = ["sigma_low", "sigma_medium", "sigma_high"]
    existing_ordered = [col for col in ordered_cols if col in df.columns]

    if existing_ordered:
        return existing_ordered

    return sorted(
        [
            col for col in df.columns
            if col.startswith("sigma_state_")
        ]
    )

def pretty_regime_label(col):
    label_map = {
        "sigma_low": "Low-volatility sigma",
        "sigma_medium": "Medium-volatility sigma",
        "sigma_high": "High-volatility sigma",
        "p_low_pred": "Predicted low-volatility probability",
        "p_medium_pred": "Predicted medium-volatility probability",
        "p_high_pred": "Predicted high-volatility probability",
        "p_low_filt": "Filtered low-volatility probability",
        "p_medium_filt": "Filtered medium-volatility probability",
        "p_high_filt": "Filtered high-volatility probability",
    }

    return label_map.get(col, col)

def _validate_option_type(option_type):
    option_type = str(option_type).lower().strip()
    if option_type in ["c", "call"]:
        return "call"
    if option_type in ["p", "put"]:
        return "put"
    raise ValueError(f"Unsupported option_type: {option_type}")

def _payoff(S_grid, K, option_type):
    option_type = _validate_option_type(option_type)
    if option_type == "call":
        return np.maximum(S_grid - K, 0.0)
    return np.maximum(K - S_grid, 0.0)

def _boundary_values(tau, Smax, K, r, option_type):
    """
    Boundary values at time-to-maturity tau.
    The same boundary is used for both regimes.
    """
    option_type = _validate_option_type(option_type)

    if option_type == "call":
        lower = 0.0
        upper = Smax - K * np.exp(-r * tau)
    else:
        lower = K * np.exp(-r * tau)
        upper = 0.0

    return float(max(lower, 0.0)), float(max(upper, 0.0))

def solve_rsbs_pde_2state(
    S0,
    K,
    T,
    r,
    sigma_low,
    sigma_high,
    q_lh,
    q_hl,
    option_type="call",
    Smax=None,
    n_S=300,
    n_t=300,
    return_grid=False,
):
    """
    Solve the two-state regime-switching Black-Scholes PDE using a fully
    implicit finite-difference scheme.

    Parameters
    ----------
    S0, K : float
        Spot and strike.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    sigma_low, sigma_high : float
        Annualized volatilities for the low- and high-volatility regimes.
    q_lh, q_hl : float
        Annualized transition intensities from low to high and high to low.
    option_type : {"call", "put"}
        European option type.
    Smax : float or None
        Upper stock-price grid boundary. If None, a conservative default is used.
    n_S, n_t : int
        Number of stock and time grid steps.
    return_grid : bool
        If True, return the full grid output.

    Returns
    -------
    V_low_0, V_high_0 : float
        Regime-conditional prices at S0.
    Optional third output:
        dictionary with S_grid, V_low, V_high.
    """
    option_type = _validate_option_type(option_type)

    S0 = float(S0)
    K = float(K)
    T = float(T)
    r = float(r)
    sigma_low = float(sigma_low)
    sigma_high = float(sigma_high)
    q_lh = max(float(q_lh), 0.0)
    q_hl = max(float(q_hl), 0.0)

    if not np.isfinite(S0) or not np.isfinite(K) or S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive finite numbers.")

    if not np.isfinite(T) or T <= 0:
        intrinsic = float(_payoff(np.array([S0]), K, option_type)[0])
        if return_grid:
            return intrinsic, intrinsic, {"S_grid": np.array([S0]), "V_low": np.array([intrinsic]), "V_high": np.array([intrinsic])}
        return intrinsic, intrinsic

    if sigma_low <= 0 or sigma_high <= 0:
        raise ValueError("Regime volatilities must be positive.")

    if Smax is None:
        Smax = max(3.0 * S0, 2.0 * K)

    Smax = float(Smax)
    n_S = int(n_S)
    n_t = int(n_t)

    if n_S < 50:
        raise ValueError("n_S should be at least 50 for a stable grid.")
    if n_t < 50:
        raise ValueError("n_t should be at least 50 for a stable grid.")

    dS = Smax / n_S
    dt = T / n_t

    S_grid = np.linspace(0.0, Smax, n_S + 1)
    interior_S = S_grid[1:-1]
    m = len(interior_S)

    V_low = _payoff(S_grid, K, option_type)
    V_high = V_low.copy()

    sigmas = [sigma_low, sigma_high]

    # Build finite-difference coefficients for each regime on the interior grid.
    # PDE operator:
    # L V = 0.5 sigma^2 S^2 V_SS + r S V_S
    # Backward implicit step in tau:
    # (I - dt * A) V^{new} - dt * q_ij V_j^{new} = V^{old}
    # where A includes diffusion, drift, discount, and own-state switching terms.
    A_blocks = []

    for sigma, q_out in [(sigma_low, q_lh), (sigma_high, q_hl)]:
        a = 0.5 * sigma**2 * interior_S**2 / dS**2 - 0.5 * r * interior_S / dS
        b = -sigma**2 * interior_S**2 / dS**2 - r - q_out
        c = 0.5 * sigma**2 * interior_S**2 / dS**2 + 0.5 * r * interior_S / dS

        lower = -dt * a[1:]
        diag = 1.0 - dt * b
        upper = -dt * c[:-1]

        A = np.zeros((m, m))
        np.fill_diagonal(A, diag)
        np.fill_diagonal(A[1:], lower)
        np.fill_diagonal(A[:, 1:], upper)

        A_blocks.append((A, a, c))

    A_low, a_low, c_low = A_blocks[0]
    A_high, a_high, c_high = A_blocks[1]

    coupling_low_to_high = -dt * q_lh * np.eye(m)
    coupling_high_to_low = -dt * q_hl * np.eye(m)

    system_matrix = np.block(
        [
            [A_low, coupling_low_to_high],
            [coupling_high_to_low, A_high],
        ]
    )

    # The system matrix is constant in time, so factor it once.
    system_lu = lu_factor(system_matrix)

    # Step forward in time-to-maturity from payoff at tau=0 to price at tau=T.
    for step in range(n_t):
        tau_new = (step + 1) * dt

        low_lower_bc, low_upper_bc = _boundary_values(tau_new, Smax, K, r, option_type)
        high_lower_bc, high_upper_bc = low_lower_bc, low_upper_bc

        rhs_low = V_low[1:-1].copy()
        rhs_high = V_high[1:-1].copy()

        # Boundary contributions from the implicit left-hand side.
        rhs_low[0] += dt * a_low[0] * low_lower_bc
        rhs_low[-1] += dt * c_low[-1] * low_upper_bc

        rhs_high[0] += dt * a_high[0] * high_lower_bc
        rhs_high[-1] += dt * c_high[-1] * high_upper_bc

        rhs = np.concatenate([rhs_low, rhs_high])

        solution = lu_solve(system_lu, rhs)

        V_low[0] = low_lower_bc
        V_low[-1] = low_upper_bc
        V_low[1:-1] = solution[:m]

        V_high[0] = high_lower_bc
        V_high[-1] = high_upper_bc
        V_high[1:-1] = solution[m:]

    V_low_0 = float(np.interp(S0, S_grid, V_low))
    V_high_0 = float(np.interp(S0, S_grid, V_high))

    if return_grid:
        return V_low_0, V_high_0, {
            "S_grid": S_grid,
            "V_low": V_low,
            "V_high": V_high,
            "Smax": Smax,
            "n_S": n_S,
            "n_t": n_t,
        }

    return V_low_0, V_high_0

def price_rsbs_pde(
    S0,
    K,
    T,
    r,
    sigma_low,
    sigma_high,
    q_lh,
    q_hl,
    p_low,
    p_high,
    option_type="call",
    Smax=None,
    n_S=300,
    n_t=300,
):
    """
    Price a European option using the 2-state regime-switching BS PDE.

    Returns
    -------
    dict
        V_low, V_high, rs_price, and normalized probabilities.
    """
    p = _normalize_prob_vector([p_low, p_high])
    p_low = float(p[0])
    p_high = float(p[1])

    V_low, V_high = solve_rsbs_pde_2state(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma_low=sigma_low,
        sigma_high=sigma_high,
        q_lh=q_lh,
        q_hl=q_hl,
        option_type=option_type,
        Smax=Smax,
        n_S=n_S,
        n_t=n_t,
        return_grid=False,
    )

    rs_price = p_low * V_low + p_high * V_high

    return {
        "V_low": V_low,
        "V_high": V_high,
        "rs_price": float(rs_price),
        "p_low": p_low,
        "p_high": p_high,
    }

def clean_generator_matrix(Q):
    """
    Ensure a valid continuous-time generator:
    off-diagonal entries non-negative and rows summing to zero.
    """
    Q = np.asarray(Q, dtype=float).copy()
    n_states = Q.shape[0]

    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                Q[i, j] = max(float(Q[i, j]), 0.0)

    for i in range(n_states):
        Q[i, i] = -np.sum(Q[i, np.arange(n_states) != i])

    return Q

def solve_rsbs_pde_nstate(
    S0,
    K,
    T,
    r,
    sigmas,
    Q,
    option_type="call",
    Smax=None,
    n_S=300,
    n_t=300,
    return_grid=False,
):
    """
    Solve the N-state regime-switching Black-Scholes PDE using a fully
    implicit finite-difference scheme.
    """
    option_type = _validate_option_type(option_type)

    S0 = float(S0)
    K = float(K)
    T = float(T)
    r = float(r)

    sigmas = np.asarray(sigmas, dtype=float).reshape(-1)
    n_states = len(sigmas)

    Q = clean_generator_matrix(Q)

    if Q.shape != (n_states, n_states):
        raise ValueError("Q must have shape (n_states, n_states).")

    if not np.isfinite(S0) or not np.isfinite(K) or S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive finite numbers.")

    if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0):
        raise ValueError("All regime volatilities must be positive finite numbers.")

    if not np.isfinite(T) or T <= 0:
        intrinsic = float(_payoff(np.array([S0]), K, option_type)[0])
        values = np.repeat(intrinsic, n_states)

        if return_grid:
            return values, {
                "S_grid": np.array([S0]),
                "V": values.reshape(n_states, 1),
            }

        return values

    if Smax is None:
        Smax = max(3.0 * S0, 2.0 * K)

    Smax = float(Smax)
    n_S = int(n_S)
    n_t = int(n_t)

    if n_S < 50:
        raise ValueError("n_S should be at least 50 for a stable grid.")
    if n_t < 50:
        raise ValueError("n_t should be at least 50 for a stable grid.")

    dS = Smax / n_S
    dt = T / n_t

    S_grid = np.linspace(0.0, Smax, n_S + 1)
    interior_S = S_grid[1:-1]
    m = len(interior_S)

    payoff = _payoff(S_grid, K, option_type)
    V = np.vstack([payoff.copy() for _ in range(n_states)])

    system_matrix = np.zeros((n_states * m, n_states * m))
    boundary_a = []
    boundary_c = []

    for state in range(n_states):
        sigma = sigmas[state]
        q_out = np.sum(Q[state, np.arange(n_states) != state])

        a = 0.5 * sigma**2 * interior_S**2 / dS**2 - 0.5 * r * interior_S / dS
        b = -sigma**2 * interior_S**2 / dS**2 - r - q_out
        c = 0.5 * sigma**2 * interior_S**2 / dS**2 + 0.5 * r * interior_S / dS

        lower = -dt * a[1:]
        diag = 1.0 - dt * b
        upper = -dt * c[:-1]

        A = np.zeros((m, m))
        np.fill_diagonal(A, diag)
        np.fill_diagonal(A[1:], lower)
        np.fill_diagonal(A[:, 1:], upper)

        row_start = state * m
        row_end = (state + 1) * m

        system_matrix[row_start:row_end, row_start:row_end] = A

        for other_state in range(n_states):
            if other_state == state:
                continue

            col_start = other_state * m
            col_end = (other_state + 1) * m

            system_matrix[row_start:row_end, col_start:col_end] = (
                -dt * Q[state, other_state] * np.eye(m)
            )

        boundary_a.append(a)
        boundary_c.append(c)

    system_lu = lu_factor(system_matrix)

    for step in range(n_t):
        tau_new = (step + 1) * dt
        lower_bc, upper_bc = _boundary_values(tau_new, Smax, K, r, option_type)

        rhs_blocks = []

        for state in range(n_states):
            rhs_state = V[state, 1:-1].copy()

            rhs_state[0] += dt * boundary_a[state][0] * lower_bc
            rhs_state[-1] += dt * boundary_c[state][-1] * upper_bc

            rhs_blocks.append(rhs_state)

        rhs = np.concatenate(rhs_blocks)
        solution = lu_solve(system_lu, rhs)

        for state in range(n_states):
            start = state * m
            end = (state + 1) * m

            V[state, 0] = lower_bc
            V[state, -1] = upper_bc
            V[state, 1:-1] = solution[start:end]

    values_at_S0 = np.array(
        [
            float(np.interp(S0, S_grid, V[state]))
            for state in range(n_states)
        ]
    )

    if return_grid:
        return values_at_S0, {
            "S_grid": S_grid,
            "V": V,
            "Smax": Smax,
            "n_S": n_S,
            "n_t": n_t,
        }

    return values_at_S0

def price_rsbs_pde_nstate(
    S0,
    K,
    T,
    r,
    sigmas,
    Q,
    probabilities,
    option_type="call",
    Smax=None,
    n_S=300,
    n_t=300,
):
    """
    Price a European option using an N-state regime-switching BS PDE.
    """
    sigmas = np.asarray(sigmas, dtype=float).reshape(-1)
    probabilities = _normalize_prob_vector(probabilities)
    Q = np.asarray(Q, dtype=float)

    if len(sigmas) != len(probabilities):
        raise ValueError("sigmas and probabilities must have the same length.")

    values = solve_rsbs_pde_nstate(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigmas=sigmas,
        Q=Q,
        option_type=option_type,
        Smax=Smax,
        n_S=n_S,
        n_t=n_t,
        return_grid=False,
    )

    rs_price = float(np.dot(probabilities, values))

    out = {
        "rs_price": rs_price,
        "probabilities": probabilities,
        "regime_values": values,
    }

    for i, value in enumerate(values):
        out[f"V_state_{i}"] = float(value)
        out[f"p_state_{i}"] = float(probabilities[i])

    return out

def prepare_atm_option_pricing_df(
    final_df,
    option_price_col="option_price",
    option_type=OPTION_TYPE,
):
    """
    Build the ATM pricing dataframe using the observed strike and expiry date.
    """
    pricing_df = final_df[
        ["date", "spx", "r_3m", "strike", "expiry_date", option_price_col]
    ].copy()

    pricing_df["date"] = pd.to_datetime(pricing_df["date"], errors="coerce")
    pricing_df["expiry_date"] = pd.to_datetime(pricing_df["expiry_date"], errors="coerce")

    for col in ["spx", "r_3m", "strike", option_price_col]:
        pricing_df[col] = pd.to_numeric(pricing_df[col], errors="coerce")

    pricing_df = pricing_df.rename(columns={option_price_col: "market_price"})
    pricing_df["T"] = (pricing_df["expiry_date"] - pricing_df["date"]).dt.days / 365.0
    pricing_df["maturity_days"] = (pricing_df["expiry_date"] - pricing_df["date"]).dt.days
    pricing_df["option_type"] = option_type
    pricing_df["moneyness"] = pricing_df["spx"] / pricing_df["strike"]

    pricing_df = pricing_df.replace([np.inf, -np.inf], np.nan)
    pricing_df = pricing_df.dropna(subset=["date", "spx", "r_3m", "strike", "market_price", "T"])
    pricing_df = pricing_df[
        (pricing_df["spx"] > 0)
        & (pricing_df["strike"] > 0)
        & (pricing_df["market_price"] > 0)
        & (pricing_df["T"] > 0)
    ].copy()

    pricing_df = pricing_df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    print("ATM option pricing dataframe")
    print("-" * 80)
    print(f"Rows: {len(pricing_df)}")
    print(f"Date range: {pricing_df['date'].min().date()} to {pricing_df['date'].max().date()}")
    print(f"Mean maturity: {pricing_df['maturity_days'].mean():.2f} calendar days")
    print(f"Mean moneyness S/K: {pricing_df['moneyness'].mean():.4f}")

    return pricing_df

def merge_atm_options_with_walkforward(atm_option_df, walkforward_regime_df):
    """
    Merge ATM option market prices with walk-forward HMM regime estimates.

    Works for both 2-state and 3-state HMM outputs.
    """
    atm = atm_option_df.copy()
    atm["date"] = pd.to_datetime(atm["date"], errors="coerce")

    regimes = walkforward_regime_df.copy()
    regimes["date"] = pd.to_datetime(regimes["date"], errors="coerce")

    if "n_components" not in regimes.columns:
        raise ValueError("walkforward_regime_df must contain 'n_components'.")

    sigma_cols = sorted([col for col in regimes.columns if col.startswith("sigma_state_")])
    prob_cols = sorted([col for col in regimes.columns if col.startswith("p_state_") and col.endswith("_pred")])
    Q_cols = sorted([col for col in regimes.columns if col.startswith("Q_")])

    alias_cols = [
        "sigma_low",
        "sigma_medium",
        "sigma_high",
        "p_low_pred",
        "p_medium_pred",
        "p_high_pred",
    ]

    keep_cols = ["date", "n_components"] + sigma_cols + prob_cols + Q_cols
    keep_cols += [col for col in alias_cols if col in regimes.columns]
    keep_cols = list(dict.fromkeys(keep_cols))

    pricing_df = atm.merge(
        regimes[keep_cols],
        on="date",
        how="inner",
    )

    pricing_df = pricing_df.sort_values("date").reset_index(drop=True)

    return pricing_df

def extract_sigmas_probabilities_Q_from_row(row):
    """
    Extract sigmas, predicted probabilities, and Q matrix from a pricing row.
    Works for both 2-state and 3-state rows.
    """
    n_components = int(row["n_components"])

    sigma_cols = [f"sigma_state_{i}" for i in range(n_components)]
    prob_cols = [f"p_state_{i}_pred" for i in range(n_components)]

    sigmas = np.array([float(row[col]) for col in sigma_cols], dtype=float)
    probabilities = np.array([float(row[col]) for col in prob_cols], dtype=float)

    Q = np.zeros((n_components, n_components), dtype=float)

    for i in range(n_components):
        for j in range(n_components):
            col = f"Q_{i}{j}"
            if col not in row.index:
                raise ValueError(f"Missing Q column: {col}")
            Q[i, j] = float(row[col])

    return sigmas, probabilities, Q

def price_atm_options_walkforward(pricing_df, n_S=150, n_t=150, progress_every=50):
    """
    Price the ATM 30D option series using the walk-forward RSBS PDE.

    Works with both 2-state and 3-state HMM outputs.
    """
    rows = []

    for idx, row in pricing_df.iterrows():
        try:
            sigmas, probabilities, Q = extract_sigmas_probabilities_Q_from_row(row)

            out = price_rsbs_pde_nstate(
                S0=float(row["spx"]),
                K=float(row["strike"]),
                T=float(row["T"]),
                r=float(row["r_3m"]),
                sigmas=sigmas,
                Q=Q,
                probabilities=probabilities,
                option_type=row["option_type"],
                n_S=n_S,
                n_t=n_t,
            )

            new_row = row.to_dict()
            new_row["rs_price"] = out["rs_price"]
            new_row["success"] = True
            new_row["error_message"] = ""

            for i in range(len(sigmas)):
                new_row[f"V_state_{i}"] = out[f"V_state_{i}"]
                new_row[f"p_state_{i}_used"] = out[f"p_state_{i}"]

            new_row["V_low"] = out["V_state_0"]

            if len(sigmas) == 2:
                new_row["V_high"] = out["V_state_1"]

            if len(sigmas) == 3:
                new_row["V_medium"] = out["V_state_1"]
                new_row["V_high"] = out["V_state_2"]

        except Exception as exc:
            new_row = row.to_dict()
            new_row["rs_price"] = np.nan
            new_row["success"] = False
            new_row["error_message"] = repr(exc)

        rows.append(new_row)

        if progress_every is not None and progress_every > 0:
            if (idx + 1) % progress_every == 0:
                print(f"Priced {idx + 1} / {len(pricing_df)} observations")

    result = pd.DataFrame(rows)

    result["pricing_error"] = result["rs_price"] - result["market_price"]
    result["abs_error"] = result["pricing_error"].abs()
    result["relative_error"] = result["abs_error"] / result["market_price"]

    return result

def black_scholes_price(S0, K, T, r, sigma, option_type="call"):
    option_type = _validate_option_type(option_type)

    S0, K, T, r, sigma = map(float, [S0, K, T, r, sigma])

    if S0 <= 0 or K <= 0 or not np.isfinite([S0, K, T, r, sigma]).all():
        return np.nan

    if T <= 0:
        return float(_payoff(np.array([S0]), K, option_type)[0])

    if sigma <= 0:
        return np.nan

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1))

def summarize_model_errors(df, model_name, price_col, error_col, abs_error_col, rel_error_col):
    return {
        "model": model_name,
        "n_obs": len(df),
        "MAE": df[abs_error_col].mean(),
        "RMSE": np.sqrt((df[error_col] ** 2).mean()),
        "mean_error": df[error_col].mean(),
        "median_error": df[error_col].median(),
        "mean_relative_error": df[rel_error_col].mean(),
        "median_relative_error": df[rel_error_col].median(),
        "mean_model_price": df[price_col].mean(),
        "mean_market_price": df["market_price"].mean(),
    }

def black_scholes_implied_vol(
    market_price,
    S0,
    K,
    T,
    r,
    option_type="call",
    sigma_lower=1e-6,
    sigma_upper=5.0,
):
    """
    Black-Scholes implied volatility using the same option inputs
    used in the pricing exercise: observed strike and observed maturity.
    """
    if (
        not np.isfinite(market_price)
        or not np.isfinite(S0)
        or not np.isfinite(K)
        or not np.isfinite(T)
        or not np.isfinite(r)
    ):
        return np.nan

    if market_price <= 0 or S0 <= 0 or K <= 0 or T <= 0:
        return np.nan

    intrinsic = float(_payoff(np.array([S0]), K, option_type)[0])
    discounted_upper_bound = float(S0) if _validate_option_type(option_type) == "call" else float(K) * np.exp(-float(r) * float(T))

    if market_price < intrinsic - 1e-8:
        return np.nan

    if market_price > discounted_upper_bound + 1e-8:
        return np.nan

    def objective(sigma):
        return black_scholes_price(
            S0=S0,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            option_type=option_type,
        ) - market_price

    try:
        low_value = objective(sigma_lower)
        high_value = objective(sigma_upper)

        if (
            not np.isfinite(low_value)
            or not np.isfinite(high_value)
            or low_value * high_value > 0
        ):
            return np.nan

        return brentq(objective, sigma_lower, sigma_upper, maxiter=200)

    except Exception:
        return np.nan

def newey_west_variance_of_mean(x, max_lag=None):
    """
    Newey-West/HAC variance estimate for the sample mean of x.

    Returns Var(mean(x)). Bartlett weights are used. If max_lag is None,
    a common automatic bandwidth floor(4 * (n / 100) ** (2/9)) is used.
    """
    x = pd.Series(x).dropna().astype(float).to_numpy()
    n = len(x)

    if n < 2:
        return np.nan, 0

    demeaned = x - x.mean()

    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))

    max_lag = max(0, min(int(max_lag), n - 1))

    gamma_0 = np.mean(demeaned * demeaned)
    long_run_variance = gamma_0

    for lag in range(1, max_lag + 1):
        gamma_lag = np.mean(demeaned[lag:] * demeaned[:-lag])
        weight = 1.0 - lag / (max_lag + 1.0)
        long_run_variance += 2.0 * weight * gamma_lag

    return long_run_variance / n, max_lag

def diebold_mariano_hac_test(loss_model_1, loss_model_2, alternative="two-sided", max_lag=None):
    """
    Diebold-Mariano-style test for equal predictive accuracy.

    The loss differential is d_t = loss_model_1 - loss_model_2.
    Here model_1 is RSBS and model_2 is rolling-volatility BS.
    Therefore, a negative mean differential favours RSBS.
    """
    d = pd.Series(loss_model_1).astype(float) - pd.Series(loss_model_2).astype(float)
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(d)

    if n < 2:
        return {
            "n_obs": n,
            "mean_loss_diff": np.nan,
            "test_stat": np.nan,
            "p_value": np.nan,
            "hac_lag": np.nan,
        }

    var_mean, used_lag = newey_west_variance_of_mean(d, max_lag=max_lag)

    if not np.isfinite(var_mean) or var_mean <= 0:
        test_stat = np.nan
        p_value = np.nan
    else:
        test_stat = d.mean() / np.sqrt(var_mean)

        if alternative == "less":
            p_value = student_t.cdf(test_stat, df=n - 1)
        elif alternative == "greater":
            p_value = 1.0 - student_t.cdf(test_stat, df=n - 1)
        else:
            p_value = 2.0 * (1.0 - student_t.cdf(abs(test_stat), df=n - 1))

    return {
        "n_obs": n,
        "mean_loss_diff": d.mean(),
        "test_stat": test_stat,
        "p_value": p_value,
        "hac_lag": used_lag,
    }

def prepare_option_pricing_df(
    df,
    option_price_col="option_price",
    option_type=OPTION_TYPE,
    use_actual_expiry=False,
    maturity_days=ATM_MATURITY_DAYS,
):
    pricing_df = df[
        ["date", "spx", "r_3m", "strike", "expiry_date", option_price_col]
    ].copy()

    pricing_df["date"] = pd.to_datetime(pricing_df["date"], errors="coerce")
    pricing_df["expiry_date"] = pd.to_datetime(pricing_df["expiry_date"], errors="coerce")

    for col in ["spx", "r_3m", "strike", option_price_col]:
        pricing_df[col] = pd.to_numeric(pricing_df[col], errors="coerce")

    pricing_df["market_price"] = pricing_df[option_price_col]

    if use_actual_expiry:
        pricing_df["days_to_expiry"] = (
            pricing_df["expiry_date"] - pricing_df["date"]
        ).dt.days
        pricing_df["T"] = pricing_df["days_to_expiry"] / 365
    else:
        pricing_df["days_to_expiry"] = maturity_days
        pricing_df["T"] = maturity_days / 365

    pricing_df["option_type"] = option_type
    pricing_df["moneyness"] = pricing_df["spx"] / pricing_df["strike"]

    pricing_df = pricing_df[
        pricing_df["date"].notna()
        & (pricing_df["spx"] > 0)
        & (pricing_df["strike"] > 0)
        & (pricing_df["T"] > 0)
        & pricing_df["r_3m"].notna()
        & (pricing_df["market_price"] > 0)
    ].copy()

    return pricing_df.sort_values("date").reset_index(drop=True)

def price_otm_options_alpha_adjusted(pricing_df, n_S=150, n_t=150, progress_every=50):
    rows = []

    for idx, row in pricing_df.iterrows():
        try:
            sigmas, probabilities, Q = extract_sigmas_probabilities_Q_from_row(row)

            alpha = float(row["alpha_otm"])
            adjusted_sigmas = alpha * sigmas

            out = price_rsbs_pde_nstate(
                S0=float(row["spx"]),
                K=float(row["strike"]),
                T=float(row["T"]),
                r=float(row["r_3m"]),
                sigmas=adjusted_sigmas,
                Q=Q,
                probabilities=probabilities,
                option_type=row["option_type"],
                n_S=n_S,
                n_t=n_t,
            )

            new_row = row.to_dict()
            new_row["rs_price_alpha"] = out["rs_price"]
            new_row["alpha_otm_used"] = alpha
            new_row["success_alpha"] = True
            new_row["error_message_alpha"] = ""

            for i in range(len(adjusted_sigmas)):
                new_row[f"sigma_state_{i}_alpha"] = adjusted_sigmas[i]
                new_row[f"V_state_{i}_alpha"] = out[f"V_state_{i}"]

        except Exception as exc:
            new_row = row.to_dict()
            new_row["rs_price_alpha"] = np.nan
            new_row["alpha_otm_used"] = np.nan
            new_row["success_alpha"] = False
            new_row["error_message_alpha"] = repr(exc)

        rows.append(new_row)

        if progress_every is not None and progress_every > 0:
            if (idx + 1) % progress_every == 0:
                print(f"Alpha-adjusted priced {idx + 1} / {len(pricing_df)} observations")

    result = pd.DataFrame(rows)

    result["rs_error_alpha"] = result["rs_price_alpha"] - result["market_price"]
    result["rs_abs_error_alpha"] = result["rs_error_alpha"].abs()
    result["rs_relative_error_alpha"] = result["rs_abs_error_alpha"] / result["market_price"]

    return result

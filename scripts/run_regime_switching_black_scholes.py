#!/usr/bin/env python
# coding: utf-8

# # Regime-Switching Black-Scholes Option Pricing
# 
# The notebook estimates Gaussian Hidden Markov Models on S&P 500 log returns, selects the two-state specification as the operational model, and applies a walk-forward regime-switching Black-Scholes PDE to ATM and OTM option pricing.
# 

# # 1. Load Dataset & Log-return (PDF Chap.2)

# The input file `option_price_atm.csv` contains the cleaned and aligned ATM option dataset used in the thesis. It includes:
# 
# - `date`: observation date;
# - `spx`: S&P 500 index level;
# - `r_3m`: 3-month U.S. Treasury rate;
# - `strike`: option strike;
# - `expiry_date`: option expiration date;
# - `option_price`: SPX call option market price.
# 
# The code looks for the file in `dataset/`, in the notebook folder.
# 

# ## Imports and setup

# In[1]:


from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import norm
from scipy.linalg import logm
from hmmlearn.hmm import GaussianHMM
from IPython.display import display

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rsbs.core import *  # noqa: F403 - research script namespace

try:
    from mpl_bsic.apply_bsic_style import BSIC_COLORS, DEFAULT_COLOR_CYCLE, DEFAULT_TITLE_STYLE
except ImportError:
    # Fallback keeps the notebook executable even when mpl-bsic is not installed.
    BSIC_COLORS = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    DEFAULT_COLOR_CYCLE = plt.rcParams["axes.prop_cycle"]
    DEFAULT_TITLE_STYLE = {}






warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda x: f"{x:,.6f}")

TRADING_DAYS_PER_YEAR = 252
DATA_FILE = find_data_file("option_price_atm.csv")
PREVIEW_ROWS = 5

print(f"ATM data file: {DATA_FILE}")


# In[2]:


df_raw = pd.read_csv(
    DATA_FILE,
    sep=";",
    decimal=","
).copy()

print("Raw dataset shape:", df_raw.shape)
print("Raw columns:", list(df_raw.columns))
display(df_raw.head(PREVIEW_ROWS))


# ## Validation and minimal cleaning
# 
# The cleaning procedure performs the following steps:
# 
# - parse the date column,
# 
# - coerce `spx`, `r_3m`, and `option_price` to numeric format,
# 
# - drop rows with missing essential values,
# 
# - sort observations by date,
# 
# - remove duplicate dates if any.
# 
# The `option_price` column is kept in the working dataset because it will later be used as the market benchmark against which the regime-switching Black-Scholes PDE price is compared.

# In[3]:


required_columns = ["date", "spx", "r_3m", "strike", "expiry_date", "option_price"]
missing_columns = [col for col in required_columns if col not in df_raw.columns]
if missing_columns:
    raise ValueError(f"The dataset is missing the following required columns: {missing_columns}")

df = df_raw.copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")

df["spx"] = pd.to_numeric(df["spx"], errors="coerce")
df["r_3m"] = pd.to_numeric(df["r_3m"], errors="coerce")
df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
df["option_price"] = pd.to_numeric(df["option_price"], errors="coerce")

before_rows = len(df)

df = df[["date", "spx", "r_3m", "strike", "expiry_date", "option_price"]].dropna(
    subset=["date", "spx", "r_3m", "strike", "expiry_date", "option_price"]
).copy()

duplicated_dates = int(df["date"].duplicated().sum())
if duplicated_dates > 0:
    df = df.groupby("date", as_index=False)[["spx", "r_3m", "strike", "expiry_date", "option_price"]].last()

df = df.sort_values("date").reset_index(drop=True)

after_rows = len(df)

print(f"Rows before cleaning: {before_rows}")
print(f"Rows after cleaning: {after_rows}")
print(f"Dropped rows: {before_rows - after_rows}")
print(f"Duplicate dates resolved: {duplicated_dates}")

print("\nClean dataset preview:")
display(df.head())

print("\nClean dataset tail:")
display(df.tail())


# ## Final working dataframe and feature construction
# 
# Since the dataset is already aligned, the only remaining construction step is the computation of daily log returns:
# 
# $$
# R_t = \log\left(\frac{S_t}{S_{t-1}}\right).
# $$
# 
# The dataframe `final_df` contains the cleaned market data, while `model_df` removes the first missing return and is used for HMM estimation.
# 

# In[4]:


df["log_return"] = np.log(df["spx"] / df["spx"].shift(1))

final_df = df.copy()
model_df = final_df.dropna(subset=["log_return"]).reset_index(drop=True)

assert final_df["date"].is_monotonic_increasing
assert final_df["date"].is_unique
assert final_df["spx"].notna().all()
assert final_df["r_3m"].notna().all()

print("Final dataframe summary")
print("-" * 80)
print(f"Rows in final_df: {len(final_df)}")
print(f"Rows in model_df (after dropping the first NaN return): {len(model_df)}")
print(f"Date range: {final_df['date'].min().date()} to {final_df['date'].max().date()}")

print("Head of model_df:")
display(model_df.head())

print("Tail of model_df:")
display(model_df.tail())


# The final working dataframe contains the variables required for the thesis discussion:
# 
# - `date`
# - `spx`
# - `r_3m`
# - `log_return`
# - `option_price`

# 
# ## Exploratory analysis
# 
# The figures below provide a first visual check of the data. The rolling volatility uses a 21-trading-day window and is annualized by multiplying the rolling standard deviation of daily log returns by `sqrt(252)`.
# 

# In[5]:


ROLLING_WINDOW = 21

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(final_df["date"], final_df["spx"])
ax.set_title("S&P 500 level")
ax.set_xlabel("Date")
ax.set_ylabel("Index level")
apply_bsic_colors_only(fig, ax)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(model_df["date"], model_df["log_return"])
ax.set_title("Daily log returns")
ax.set_xlabel("Date")
ax.set_ylabel("Log return")
apply_bsic_colors_only(fig, ax)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(model_df["log_return"], bins=40, color=BSIC_COLORS[0], edgecolor="white")
ax.set_title("Histogram of daily log returns")
ax.set_xlabel("Log return")
ax.set_ylabel("Frequency")
apply_bsic_colors_only(fig, ax)
plt.tight_layout()
plt.show()

rolling_vol_annualized = model_df["log_return"].rolling(ROLLING_WINDOW).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(model_df["date"], rolling_vol_annualized)
ax.set_title(f"{ROLLING_WINDOW}-day rolling volatility (annualized)")
ax.set_xlabel("Date")
ax.set_ylabel("Volatility")
apply_bsic_colors_only(fig, ax)
plt.tight_layout()
plt.show()



# # 2. Hidden Markov Models with different numbers of states (PDF Chap.3)

# ## Parameter estimation
# 
# The empirical regime classification is based on daily log returns. Conditional on a latent regime $X_t = i$, returns are assumed to follow
# 
# $$
# R_t \mid X_t = i \sim \mathcal{N}(\mu_i, \sigma_i^2),
# $$
# 
# while the latent state follows a discrete Markov chain with transition matrix $P=(p_{ij})$.
# 
# Both the two-state and three-state Gaussian HMMs are estimated and compared before selecting the operational specification used in pricing.
# 

# In[6]:

















# In[7]:


HMM_RANDOM_STATE = 123
HMM_MAX_ITER = 500
HMM_TOL = 1e-8
HMM_N_INIT = 20

returns = model_df["log_return"].to_numpy()

hmm_results = {
    n_states: fit_gaussian_hmm(
        returns,
        n_states=n_states,
        random_state=HMM_RANDOM_STATE,
        n_iter=HMM_MAX_ITER,
        tol=HMM_TOL,
        n_init=HMM_N_INIT,
    )
    for n_states in [2, 3]
}

comparison_table = pd.DataFrame([
    {
        "n_states": result["n_states"],  # number of latent regimes in the HMM
        "log_likelihood": result["log_likelihood"],  # higher values indicate better in-sample fit
        "AIC": result["aic"],  # lower values indicate a better fit-complexity tradeoff
        "BIC": result["bic"],  # lower values indicate a better fit-complexity tradeoff, with a stronger penalty than AIC
        "avg_posterior_confidence": result["avg_posterior_confidence"],  # values closer to 1 mean clearer regime classification
        "smallest_regime_share": state_summary_table(result)["sample_share"].min(),  # very small values may indicate a negligible or unstable regime
    }
    for result in hmm_results.values()
]).sort_values("n_states").reset_index(drop=True)

print("Model comparison table")
display(comparison_table)

for n_states, result in hmm_results.items():
    print("\n" + "=" * 80)
    print(f"{n_states}-state Gaussian HMM estimated with hmmlearn")
    print(f"Log-likelihood: {result['log_likelihood']:.6f}")
    print(f"AIC: {result['aic']:.6f}")
    print(f"BIC: {result['bic']:.6f}")
    print("State summary (states are sorted from low to high volatility):")
    display(state_summary_table(result))
    print("Transition matrix:")
    display(transition_matrix_table(result))
    print("Posterior probability preview:")
    display(posterior_preview_table(result, model_df["date"]))


# In[8]:


for n_states, result in hmm_results.items():
    plot_hmm_diagnostics(model_df["date"], model_df["log_return"], result)


# ## Regime interpretation
# 
# The hidden states are relabelled by increasing volatility. This gives economically meaningful labels:
# 
# - **2 states**: low-volatility, high-volatility;
# - **3 states**: low-volatility, medium-volatility, high-volatility.
# 
# The labels are based primarily on volatility rather than on the regime-dependent mean returns, because daily mean estimates are typically much noisier than daily volatility estimates.
# 
# Although a 3-state model may be statistically preferred on the basis of information criteria, a 2-state model may still remain valuable as a more tractable benchmark for regime-switching option pricing.

# In[9]:


preferred_states = 2
# Choose the working HMM specification directly: set to 2 or 3.

preferred_result = hmm_results[preferred_states]




preferred_summary = state_summary_table(preferred_result).copy()
preferred_summary["regime_label"] = volatility_labels(preferred_states)
preferred_summary = preferred_summary[
    [
        "state",
        "regime_label",
        "mean_daily_return",
        "daily_volatility",
        "annualized_volatility",
        "count",
        "sample_share",
        "expected_duration_days",
    ]
]

display(preferred_summary)

regime_labels = dict(zip(preferred_summary["state"], preferred_summary["regime_label"]))

model_df["preferred_state"] = preferred_result["states"]
model_df["preferred_regime_label"] = model_df["preferred_state"].map(regime_labels)

for state, label in regime_labels.items():
    safe_label = label.replace("-", "_").replace(" ", "_")
    model_df[f"posterior_{safe_label}"] = preferred_result["posterior"][:, state]

print("Preview of model_df with regime labels:")
display(model_df.head())

fig, ax = plt.subplots(figsize=(12, 4))
preferred_cmap = ListedColormap(BSIC_COLORS[:preferred_states])
ax.scatter(model_df["date"], model_df["spx"], c=model_df["preferred_state"], s=12, cmap=preferred_cmap)
ax.plot(model_df["date"], model_df["spx"], linewidth=0.8, alpha=0.5)
ax.set_title(f"S&P 500 level with inferred regimes ({preferred_states}-state working model)")
ax.set_xlabel("Date")
ax.set_ylabel("Index level")
apply_bsic_colors_only(fig, ax)
plt.tight_layout()
plt.show()



# 
# ## Transition intensities
# 
# If $P$ denotes the estimated one-step transition matrix from daily data and $\Delta t = 1/252$ years, a continuous-time approximation of the generator matrix is
# 
# $$
# Q \approx \frac{1}{\Delta t} \log(P).
# $$
# 
# For a 2-state model,
# 
# $$
# Q = \begin{pmatrix}
# -\lambda_{12} & \lambda_{12} \\
# \lambda_{21} & -\lambda_{21}
# \end{pmatrix},
# $$
# 
# so the off-diagonal elements are the annualized switching intensities.
# 
# For a 3-state model the same matrix-logarithm idea generalizes directly, but numerical estimates may sometimes require a constrained projection if tiny negative off-diagonal terms appear because of sampling noise or numerical approximation.
# 

# In[10]:





preferred_labels = preferred_summary["regime_label"].tolist()
preferred_Q = discrete_transition_to_generator(preferred_result["transmat"])

print(f"Continuous-time approximation for the {preferred_states}-state working model")
display(labeled_matrix(preferred_Q, preferred_labels))
print("Row sums (should be numerically close to zero):")
print(labeled_matrix(preferred_Q, preferred_labels).sum(axis=1))

if preferred_states == 2:
    print("\nApproximate annualized switching intensities:")
    print(f"lambda_12 = {preferred_Q[0, 1]:.6f}")
    print(f"lambda_21 = {preferred_Q[1, 0]:.6f}")
else:
    off_diagonal_mask = ~np.eye(preferred_Q.shape[0], dtype=bool)
    problematic_entries = preferred_Q[off_diagonal_mask] < -1e-8
    if problematic_entries.any():
        print(
            "Some off-diagonal entries are slightly negative. "
            "This can happen when a discrete-time estimate is converted to a continuous-time generator by matrix logarithm. "
            "A constrained projection or regularisation step can be added before solving the PDE system."
        )
    print("\nBecause a 2-state specification is often used as the pricing benchmark, its generator is also shown below:")
    benchmark_2state_result = hmm_results[2]
    benchmark_2state_labels = volatility_labels(2)
    benchmark_2state_Q = discrete_transition_to_generator(benchmark_2state_result["transmat"])
    display(labeled_matrix(benchmark_2state_Q, benchmark_2state_labels))


# # 3. Walk-forward regime-switching option pricing (PDF Chap.4)
# 
# The full-sample HMM estimates above are useful for describing the historical behavior of SPX returns. However, full-sample regime probabilities, especially posterior probabilities obtained from the entire sample, are not appropriate for option pricing because they may use information from the future. In particular, smoothed or full-sample `predict_proba` probabilities can introduce look-ahead bias.
# 
# For pricing, this section uses a walk-forward design. The HMM is retrained monthly using only data available up to the previous month-end. During each pricing month, the HMM parameters are kept fixed and regime probabilities are updated sequentially. For strict no-look-ahead option pricing on date $t$, the model uses the one-step-ahead predicted probability
# 
# $$
# \tilde p_t = p_{t-1} P,
# $$
# 
# where $p_{t-1}$ is the filtered probability after observing returns up to date $t-1$, and $P$ is the estimated discrete-time transition matrix.
# 
# This ensures that the same-day return is not used to price the same-day option quote.

# In[11]:


# ---------------------------------------------------------------------
# Helper functions for HMM filtering without look-ahead
# ---------------------------------------------------------------------











# ### Monthly walk-forward HMM estimation
# 
# At the beginning of each calendar month in the pricing period, the two-state Gaussian HMM is re-estimated using only information available up to the previous month-end. The baseline configuration uses an expanding window starting in 2015. The fitted states are sorted by increasing volatility, so state 0 is always interpreted as the low-volatility regime and state 1 as the high-volatility regime.
# 
# This timing convention avoids look-ahead bias: option prices in month \(m\) use HMM parameters estimated before month \(m\) begins, and same-day returns are not used to form the pricing-date regime probabilities.

# In[12]:


# ================================================================
# Global walk-forward HMM / RSBS configuration
# ================================================================

N_COMPONENTS = 2  # choose 2 or 3

WF_START_DATE = "2020-01-01"
WF_END_DATE = "2024-12-31"
INITIAL_TRAIN_START = "2015-01-05"

WINDOW_TYPE = "expanding"  # choose "rolling" or "expanding"
ROLLING_WINDOW = 5
ROLLING_WINDOW_UNIT = "years"
MIN_TRAIN_OBS = 30

HMM_RANDOM_STATE_WF = 42


# In[13]:


# ---------------------------------------------------------------------
# Monthly walk-forward HMM estimation
# ---------------------------------------------------------------------





# In[14]:


# ================================================================
# Run monthly walk-forward HMM estimation
# ================================================================

walkforward_regime_df, walkforward_hmm_snapshots, failed_months_df = monthly_walkforward_regime_estimates(
    model_df=model_df,
    start_date=WF_START_DATE,
    end_date=WF_END_DATE,
    initial_train_start=INITIAL_TRAIN_START,
    random_state=HMM_RANDOM_STATE_WF,
    verbose=True,
    window_type=WINDOW_TYPE,
    rolling_window=ROLLING_WINDOW,
    rolling_window_unit=ROLLING_WINDOW_UNIT,
    min_train_obs=MIN_TRAIN_OBS,
    n_components=N_COMPONENTS,
)

display(walkforward_regime_df.head())
display(walkforward_regime_df.tail())

print("Failed or skipped months:")
display(failed_months_df.head())




summary_base_cols = [
    "date",
    "n_components",
    "sigma_low",
    "sigma_medium",
    "sigma_high",
    "p_low_pred",
    "p_medium_pred",
    "p_high_pred",
    "p_low_filt",
    "p_medium_filt",
    "p_high_filt",
]

q_cols = sorted([col for col in walkforward_regime_df.columns if col.startswith("q_")])
P_cols = sorted([col for col in walkforward_regime_df.columns if col.startswith("P_")])
Q_cols = sorted([col for col in walkforward_regime_df.columns if col.startswith("Q_")])

summary_cols = get_existing_columns(
    walkforward_regime_df,
    summary_base_cols + q_cols + P_cols + Q_cols,
)

display(walkforward_regime_df[summary_cols].describe())


# ## Walk-forward regime probability diagnostics
# 
# The following plots show the predicted high-volatility probability used for pricing. This probability is one-step-ahead: it is formed before observing the return on the pricing date. This is the probability used later to combine the regime-conditional PDE prices.

# In[15]:


# ================================================================
# Flexible plotting helpers
# ================================================================







# In[16]:


# ---------------------------------------------------------------------
# Plot walk-forward probabilities
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 5))

pred_prob_cols = get_regime_probability_columns(
    walkforward_regime_df,
    prob_type="pred",
)

for col in pred_prob_cols:
    ax.plot(
        walkforward_regime_df["date"],
        walkforward_regime_df[col],
        label=pretty_regime_label(col),
    )

ax.set_title(f"Walk-forward predicted regime probabilities ({N_COMPONENTS}-state HMM)")
ax.set_xlabel("Date")
ax.set_ylabel("Probability")
ax.set_ylim(-0.02, 1.02)
ax.legend()
apply_bsic_colors_only(fig, ax)
plt.tight_layout()
plt.show()


# ## Regime-switching Black-Scholes PDE
# 
# The regime-switching Black-Scholes model prices the option jointly across regimes. For the two-state case, the low-volatility and high-volatility option values satisfy the coupled PDE system
# 
# $$
# \frac{\partial V_L}{\partial t}
# + \frac{1}{2}\sigma_L^2 S^2 \frac{\partial^2 V_L}{\partial S^2}
# + rS\frac{\partial V_L}{\partial S}
# - rV_L
# + q_{LH}(V_H - V_L)
# =0,
# $$
# 
# $$
# \frac{\partial V_H}{\partial t}
# + \frac{1}{2}\sigma_H^2 S^2 \frac{\partial^2 V_H}{\partial S^2}
# + rS\frac{\partial V_H}{\partial S}
# - rV_H
# + q_{HL}(V_L - V_H)
# =0.
# $$
# 
# At maturity, both regimes share the same payoff. For a call,
# 
# $$
# V_L(S,T)=V_H(S,T)=\max(S-K,0),
# $$
# 
# and for a put,
# 
# $$
# V_L(S,T)=V_H(S,T)=\max(K-S,0).
# $$
# 
# This PDE approach differs from a simple weighted Black-Scholes approximation. A weighted Black-Scholes approximation only averages fixed-volatility Black-Scholes prices at the valuation date. The coupled PDE instead prices the possibility that the volatility regime may switch during the option's remaining life.
# 
# Given the two regime-conditional option values, the final regime-switching model price is obtained by averaging them with the current regime probabilities:
# 
# $$
# C^{RS}(S,t)
# =
# p_L(t)V_L(S,t)
# +
# p_H(t)V_H(S,t),
# $$
# 
# where $p_L(t)$ and $p_H(t)$ are the probabilities that the market is currently in the low-volatility and high-volatility regimes, respectively. In the empirical implementation, these probabilities are the one-step-ahead predicted HMM probabilities,
# 
# $$
# p_L(t)=p_{L,t}^{pred},
# \qquad
# p_H(t)=p_{H,t}^{pred},
# $$
# 
# with
# 
# $$
# p_L(t)+p_H(t)=1.
# $$
# 
# This aggregation works because $V_L(S,t)$ and $V_H(S,t)$ are conditional prices. The value $V_L$ is the option price conditional on the current regime being low volatility, while $V_H$ is the option price conditional on the current regime being high volatility. Since the current regime is latent and therefore not directly observed, the unconditional model price is the expectation of the regime-conditional prices under the current regime probability distribution:
# 
# $$
# C^{RS}(S,t)
# =
# \mathbb{E}\left[V_{X_t}(S,t)\mid \mathcal{F}_{t-1}\right].
# $$
# 
# For the two-state case, this expectation is exactly
# 
# $$
# C^{RS}(S,t)
# =
# \mathbb{P}(X_t=L\mid \mathcal{F}_{t-1})V_L(S,t)
# +
# \mathbb{P}(X_t=H\mid \mathcal{F}_{t-1})V_H(S,t).
# $$
# 
# The implementation below uses a fully implicit finite-difference scheme. The two regime values are solved jointly at every time step, which incorporates the transition intensities $q_{LH}$ and $q_{HL}$ directly into the pricing equation. After the coupled PDE returns $V_L$ and $V_H$, the final option price is computed using the one-step-ahead predicted probabilities. This timing avoids look-ahead bias, because the date-$t$ option price is computed using only information available before observing the date-$t$ return.

# In[17]:


# ---------------------------------------------------------------------
# Finite-difference solver for the coupled 2-state RSBS PDE
# ---------------------------------------------------------------------










# ================================================================
# General N-state RSBS PDE solver
# ================================================================







# # 4. Walk-forward pricing of the ATM SPX option
# 
# The ATM dataset contains one observed option price per trading day. The pricing exercise uses the actual observed strike and expiry date when available, while retaining the same walk-forward HMM inputs described above.
# 
# For each date $t$, the pricing inputs are $S_t$, $K_t$, $T_t$, $r_t$, the observed market price $C_t^{mkt}$, the regime volatilities, the switching intensities, and the one-step-ahead predicted regime probabilities. The model price is compared directly with the market option price.
# 

# In[18]:


# ================================================================
# Walk-forward pricing of the already available ATM 30D option price
# ================================================================

ATM_MATURITY_DAYS = 30
ATM_MATURITY_YEARS = ATM_MATURITY_DAYS / 365
OPTION_TYPE = "call"  # change to "put" if your option_price column refers to puts


# ## Prepare the ATM option pricing dataframe
# 
# The original dataset contains the market option price in `option_price`, together with `strike` and `expiry_date`.
# 
# The current implementation uses the actual strike and computes time to maturity from the observed expiry date. If an expiry date is missing or invalid, the row is excluded from the pricing sample. This keeps the notebook consistent with the current thesis implementation and avoids the older approximation $K_t=S_t$, $T=30/365$.
# 

# In[19]:




# ## Merge ATM option prices with walk-forward regime estimates
# 
# The walk-forward HMM output contains the model inputs needed by the PDE solver:
# 
# $$
# \sigma_L,\quad \sigma_H,
# $$
# 
# $$
# q_{LH},\quad q_{HL},
# $$
# 
# $$
# p_{L,t}^{pred},\quad p_{H,t}^{pred}.
# $$
# 
# The next cell merges these quantities with the ATM option price series by date.

# In[20]:




# ## Price the ATM option with the regime-switching PDE
# 
# For each date, the PDE solver computes:
# 
# $$
# V_L(S_t,t)
# $$
# 
# and
# 
# $$
# V_H(S_t,t).
# $$
# 
# Then the model price is:
# 
# $$
# C_t^{RS}
# =
# p_{L,t}^{pred}V_L(S_t,t)
# +
# p_{H,t}^{pred}V_H(S_t,t).
# $$
# 
# This is compared with the observed market option price.

# In[21]:






# ## Run the walk-forward ATM option pricing exercise
# 
# This cell creates the ATM option pricing dataframe, merges it with the walk-forward HMM estimates, prices each option using the regime-switching PDE, and stores the pricing errors.
# 
# For a first run, the PDE grid uses `n_S=150` and `n_t=150`. For final thesis results, you can increase these values after checking runtime.

# In[22]:


# ================================================================
# Run walk-forward ATM option pricing
# ================================================================

atm_option_df = prepare_atm_option_pricing_df(
    final_df=final_df,
    option_price_col="option_price",
    option_type=OPTION_TYPE,
)

print("ATM option base dataframe:")
print(atm_option_df.shape)
display(atm_option_df.head())

atm_pricing_df = merge_atm_options_with_walkforward(
    atm_option_df=atm_option_df,
    walkforward_regime_df=walkforward_regime_df,
)

print("ATM pricing dataframe after merging with walk-forward regimes:")
print(atm_pricing_df.shape)

pricing_preview_cols = [
    "date",
    "spx",
    "strike",
    "T",
    "r_3m",
    "market_price",
    "n_components",
    "sigma_low",
    "sigma_medium",
    "sigma_high",
    "p_low_pred",
    "p_medium_pred",
    "p_high_pred",
]

pricing_preview_cols = get_existing_columns(atm_pricing_df, pricing_preview_cols)

display(atm_pricing_df[pricing_preview_cols].head())

example_row = atm_pricing_df.dropna(
    subset=["spx", "strike", "T", "r_3m", "market_price"]
).iloc[-1]

example_sigmas, example_probabilities, example_Q = extract_sigmas_probabilities_Q_from_row(example_row)

example_price = price_rsbs_pde_nstate(
    S0=float(example_row["spx"]),
    K=float(example_row["strike"]),
    T=float(example_row["T"]),
    r=float(example_row["r_3m"]),
    sigmas=example_sigmas,
    Q=example_Q,
    probabilities=example_probabilities,
    option_type=example_row["option_type"],
    n_S=250,
    n_t=250,
)

display(pd.DataFrame([example_price]))

priced_atm_options = price_atm_options_walkforward(
    atm_pricing_df,
    n_S=150,
    n_t=150,
    progress_every=50,
)

print("Priced ATM option dataframe:")
print(priced_atm_options.shape)

priced_preview_cols = [
    "date",
    "spx",
    "strike",
    "T",
    "market_price",
    "V_low",
    "V_medium",
    "V_high",
    "rs_price",
    "pricing_error",
    "abs_error",
    "relative_error",
    "p_low_pred",
    "p_medium_pred",
    "p_high_pred",
    "success",
]

priced_preview_cols = get_existing_columns(priced_atm_options, priced_preview_cols)

display(priced_atm_options[priced_preview_cols].head())


# ## Pricing error summary
# 
# The pricing accuracy is summarized using:
# 
# $$
# MAE = \frac{1}{N}\sum_{t=1}^{N}|C_t^{RS}-C_t^{mkt}|
# $$
# 
# $$
# RMSE =
# \sqrt{
# \frac{1}{N}\sum_{t=1}^{N}(C_t^{RS}-C_t^{mkt})^2
# }
# $$
# 
# $$
# MeanRelativeError =
# \frac{1}{N}\sum_{t=1}^{N}
# \frac{|C_t^{RS}-C_t^{mkt}|}{C_t^{mkt}}
# $$

# In[23]:


valid_atm = priced_atm_options.dropna(subset=["rs_price", "market_price"]).copy()
valid_atm = valid_atm[valid_atm["success"] & (valid_atm["market_price"] > 0)].copy()

atm_summary = pd.DataFrame([
    {
        "n_obs": len(valid_atm),
        "MAE": valid_atm["abs_error"].mean(),
        "RMSE": np.sqrt((valid_atm["pricing_error"] ** 2).mean()),
        "mean_relative_error": valid_atm["relative_error"].mean(),
        "median_relative_error": valid_atm["relative_error"].median(),
        "mean_market_price": valid_atm["market_price"].mean(),
        "mean_model_price": valid_atm["rs_price"].mean(),
        "success_rate": priced_atm_options["success"].mean(),
    }
])

display(atm_summary)


# ## Market option price versus model price
# 
# The next figure compares the observed ATM option price with the regime-switching PDE price over time.

# In[24]:


fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(valid_atm["date"], valid_atm["market_price"], label="Market ATM option price")
ax.plot(valid_atm["date"], valid_atm["rs_price"], label=f"{N_COMPONENTS}-RSBS PDE price")
ax.set_title("ATM 30D SPX option: market price vs walk-forward RSBS PDE price")
ax.set_xlabel("Date")
ax.set_ylabel("Option price")
ax.legend()
apply_bsic_colors_only(fig, ax)
plt.tight_layout()
plt.show()


# ## Pricing error over time
# 
# This figure plots:
# 
# $$
# C_t^{RS}-C_t^{mkt}.
# $$
# 
# Values above zero indicate model overpricing. Values below zero indicate model underpricing.

# In[25]:


fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(valid_atm["date"], valid_atm["pricing_error"], label=f"{N_COMPONENTS}-RSBS PDE price - market price")
ax.axhline(0, linestyle="--", linewidth=1)
ax.set_title("Walk-forward RSBS pricing error for ATM 30D option")
ax.set_xlabel("Date")
ax.set_ylabel("Pricing error")
ax.legend()
apply_bsic_colors_only(fig, ax)
plt.tight_layout()
plt.show()


# # 5. Benchmark comparison: RSBS versus standard Black-Scholes (PDF Chap.4)
# 
# To assess whether the regime-switching structure improves pricing performance, the RSBS model is compared with a standard Black-Scholes benchmark. The benchmark uses the same option inputs as the RSBS pricing exercise, namely the observed index level $S_t$, the observed strike $K_t$, the observed time to maturity $T_t$, and the short-term rate $r_t$. The only difference is that the benchmark replaces the regime-switching volatility structure with a single rolling historical volatility estimate.
# 
# To avoid look-ahead bias, the rolling volatility used on date $t$ is computed using returns available up to date $t-1$:
# 
# $$
# \sigma_t^{roll}
# =
# \operatorname{Std}(R_{t-w},\dots,R_{t-1})\sqrt{252}.
# $$
# 
# The benchmark Black-Scholes price is denoted by $C_t^{BS}$ and is compared with the regime-switching price $C_t^{RS}$ and the observed market price $C_t^{mkt}$.
# 

# In[26]:


# ---------------------------------------------------------------------
# Benchmark comparison: RSBS versus rolling-volatility Black-Scholes
# ---------------------------------------------------------------------

ROLLING_BS_WINDOW = 21



# Rolling historical volatility with one-day lag to avoid look-ahead bias
bs_vol_df = model_df[["date", "log_return"]].copy()
bs_vol_df["date"] = pd.to_datetime(bs_vol_df["date"])
bs_vol_df["bs_rolling_vol"] = (
    bs_vol_df["log_return"]
    .rolling(ROLLING_BS_WINDOW)
    .std()
    .shift(1)
    * np.sqrt(TRADING_DAYS_PER_YEAR)
)

comparison_df = priced_atm_options.copy()
comparison_df["date"] = pd.to_datetime(comparison_df["date"])

comparison_df = comparison_df.merge(
    bs_vol_df[["date", "bs_rolling_vol"]],
    on="date",
    how="left",
)

comparison_df["bs_price"] = comparison_df.apply(
    lambda row: black_scholes_price(
        S0=row["spx"],
        K=row["strike"],
        T=row["T"],
        r=row["r_3m"],
        sigma=row["bs_rolling_vol"],
        option_type=row["option_type"],
    ),
    axis=1,
)

comparison_df["rs_error"] = comparison_df["rs_price"] - comparison_df["market_price"]
comparison_df["bs_error"] = comparison_df["bs_price"] - comparison_df["market_price"]

comparison_df["rs_abs_error"] = comparison_df["rs_error"].abs()
comparison_df["bs_abs_error"] = comparison_df["bs_error"].abs()

comparison_df["rs_relative_error"] = comparison_df["rs_abs_error"] / comparison_df["market_price"]
comparison_df["bs_relative_error"] = comparison_df["bs_abs_error"] / comparison_df["market_price"]

valid_comparison = comparison_df.dropna(
    subset=["market_price", "rs_price", "bs_price", "rs_abs_error", "bs_abs_error"]
).copy()

valid_comparison = valid_comparison[
    (valid_comparison["market_price"] > 0) & (valid_comparison["success"])
].copy()

print("Valid comparison observations:", len(valid_comparison))

comparison_preview_cols = [
    "date",
    "market_price",
    "rs_price",
    "bs_price",
    "rs_error",
    "bs_error",
    "rs_abs_error",
    "bs_abs_error",
    "rs_relative_error",
    "bs_relative_error",
    "bs_rolling_vol",
    "p_low_pred",
    "p_medium_pred",
    "p_high_pred",
]

comparison_preview_cols = get_existing_columns(
    valid_comparison,
    comparison_preview_cols,
)

display(valid_comparison[comparison_preview_cols].head())


# ## Overall pricing performance
# 
# The two models are compared using MAE, RMSE, mean error, and relative pricing errors. Lower MAE and RMSE indicate better pricing accuracy. The mean error indicates whether a model tends to overprice or underprice on average.

# In[27]:




model_comparison_summary = pd.DataFrame(
    [
        summarize_model_errors(
            valid_comparison,
            "Regime-Switching BS",
            "rs_price",
            "rs_error",
            "rs_abs_error",
            "rs_relative_error",
        ),
        summarize_model_errors(
            valid_comparison,
            "Rolling-volatility BS",
            "bs_price",
            "bs_error",
            "bs_abs_error",
            "bs_relative_error",
        ),
    ]
)

display(model_comparison_summary)


# In[28]:


fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(valid_comparison["date"], valid_comparison["market_price"], label="Market ATM option price")
ax.plot(valid_comparison["date"], valid_comparison["rs_price"], label=f"{N_COMPONENTS}-state RSBS PDE price")
ax.plot(
    valid_comparison["date"],
    valid_comparison["bs_price"],
    label=f"BS price ({ROLLING_BS_WINDOW}D rolling volatility)",
    alpha=0.85,
)

ax.set_title("ATM 30D SPX option: market price vs RSBS and Black-Scholes benchmark")
ax.set_xlabel("Date")
ax.set_ylabel("Option price")
ax.legend()
apply_bsic_colors_only(fig, ax)
plt.tight_layout()
plt.show()


# ## Pricing errors over time
# 
# The next figure compares the pricing errors of the two models:
# 
# $$
# Error_t^{RS}=C_t^{RS}-C_t^{mkt},
# \qquad
# Error_t^{BS}=C_t^{BS}-C_t^{mkt}.
# $$
# 
# Values above zero indicate overpricing, while values below zero indicate underpricing.

# In[29]:


fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(valid_comparison["date"], valid_comparison["rs_error"], label="RSBS error")
ax.plot(valid_comparison["date"], valid_comparison["bs_error"], label="BS benchmark error", alpha=0.85)
ax.axhline(0, linestyle="--", linewidth=1)

ax.set_title("Pricing errors: RSBS versus rolling-volatility Black-Scholes")
ax.set_xlabel("Date")
ax.set_ylabel("Model price - market price")
ax.legend()
apply_bsic_colors_only(fig, ax)
plt.tight_layout()
plt.show()


# ## Rolling average absolute pricing errors
# 
# Because daily pricing errors are noisy, the comparison is also shown using a rolling average of absolute errors:
# 
# $$
# RollingMAE_t
# =
# \frac{1}{w}
# \sum_{j=0}^{w-1}
# |C_{t-j}^{model}-C_{t-j}^{mkt}|.
# $$

# In[30]:


ERROR_ROLLING_WINDOW = 21

valid_comparison["rs_abs_error_roll"] = (
    valid_comparison["rs_abs_error"].rolling(ERROR_ROLLING_WINDOW).mean()
)

valid_comparison["bs_abs_error_roll"] = (
    valid_comparison["bs_abs_error"].rolling(ERROR_ROLLING_WINDOW).mean()
)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    valid_comparison["date"],
    valid_comparison["rs_abs_error_roll"],
    label=f"RSBS {ERROR_ROLLING_WINDOW}D average absolute error",
)

ax.plot(
    valid_comparison["date"],
    valid_comparison["bs_abs_error_roll"],
    label=f"BS {ERROR_ROLLING_WINDOW}D average absolute error",
)

ax.set_title("Rolling average absolute pricing error")
ax.set_xlabel("Date")
ax.set_ylabel("Average absolute error")
ax.legend()
apply_bsic_colors_only(fig, ax)
plt.tight_layout()
plt.show()


# ## Implied-volatility comparison: market price versus RSBS price
# 
# The previous diagnostics compare models in price space. As an additional check, this section converts both the observed market option price and the RSBS model price into Black-Scholes implied volatilities.
# 
# For each ATM date $t$, the inversion uses the same option inputs as the pricing exercise:
# 
# $$
# S_t,\quad K_t,\quad T_t,\quad r_t.
# $$
# 
# Here $K_t$ is the observed strike and $T_t$ is computed from the observed expiry date, so the comparison is consistent with the ATM pricing dataframe and does not use the older approximation $K_t=S_t$ or a fixed maturity $T=30/365$.
# 
# The two implied-volatility series are
# 
# $$
# \sigma^{mkt,IV}_t
# =
# BS^{-1}
# \left(
# C_t^{mkt}; S_t,K_t,T_t,r_t
# \right),
# $$
# 
# and
# 
# $$
# \sigma^{RSBS,IV}_t
# =
# BS^{-1}
# \left(
# C_t^{RSBS}; S_t,K_t,T_t,r_t
# \right).
# $$
# 
# The implied-volatility error is defined as
# 
# $$
# IVError_t
# =
# \sigma^{RSBS,IV}_t
# -
# \sigma^{mkt,IV}_t.
# $$
# 
# A positive value means that the RSBS price embeds a higher Black-Scholes-equivalent volatility than the market price, while a negative value means that the RSBS price embeds a lower Black-Scholes-equivalent volatility.

# In[31]:


from scipy.optimize import brentq




# In[32]:


# ================================================================
# Implied-volatility comparison: market price vs RSBS price
# ================================================================

iv_input_check_cols = [
    "date",
    "spx",
    "strike",
    "expiry_date",
    "T",
    "r_3m",
    "market_price",
    "rs_price",
    "option_type",
]

missing_iv_cols = [
    col for col in iv_input_check_cols
    if col not in valid_comparison.columns
]

if missing_iv_cols:
    raise ValueError(
        "The following columns are required for the implied-volatility comparison "
        f"but are missing from valid_comparison: {missing_iv_cols}"
    )

valid_comparison["maturity_days_check"] = (
    pd.to_datetime(valid_comparison["expiry_date"])
    - pd.to_datetime(valid_comparison["date"])
).dt.days

valid_comparison["T_check"] = valid_comparison["maturity_days_check"] / 365.0

max_T_difference = (
    valid_comparison["T"] - valid_comparison["T_check"]
).abs().max()

print("ATM implied-volatility input consistency check")
print("-" * 80)
print("The IV inversion uses the same observed strike and maturity used for ATM pricing.")
print(f"Maximum |T - T_check|: {max_T_difference:.12f}")
print(f"Mean maturity in days: {valid_comparison['maturity_days_check'].mean():.2f}")
print(f"Median maturity in days: {valid_comparison['maturity_days_check'].median():.2f}")

if max_T_difference > 1e-10:
    raise ValueError(
        "Mismatch between stored T and expiry-date-implied T. "
        "Check the ATM pricing dataframe construction."
    )

valid_comparison["market_iv"] = valid_comparison.apply(
    lambda row: black_scholes_implied_vol(
        market_price=row["market_price"],
        S0=row["spx"],
        K=row["strike"],
        T=row["T"],
        r=row["r_3m"],
        option_type=row["option_type"],
    ),
    axis=1,
)

valid_comparison["rsbs_iv"] = valid_comparison.apply(
    lambda row: black_scholes_implied_vol(
        market_price=row["rs_price"],
        S0=row["spx"],
        K=row["strike"],
        T=row["T"],
        r=row["r_3m"],
        option_type=row["option_type"],
    ),
    axis=1,
)

iv_comparison_df = valid_comparison.dropna(
    subset=["market_iv", "rsbs_iv"]
).copy()

iv_comparison_df["iv_error"] = (
    iv_comparison_df["rsbs_iv"] - iv_comparison_df["market_iv"]
)

iv_comparison_df["abs_iv_error"] = iv_comparison_df["iv_error"].abs()

print("\nValid implied-volatility comparison observations:", len(iv_comparison_df))

display(
    iv_comparison_df[
        get_existing_columns(
            iv_comparison_df,
            [
                "date",
                "spx",
                "strike",
                "maturity_days_check",
                "T",
                "r_3m",
                "market_price",
                "rs_price",
                "market_iv",
                "rsbs_iv",
                "iv_error",
                "abs_iv_error",
                "sigma_low",
                "sigma_medium",
                "sigma_high",
                "p_low_pred",
                "p_medium_pred",
                "p_high_pred",
            ],
        )
    ].head()
)


# In[33]:


iv_comparison_summary = pd.DataFrame(
    [
        {
            "series": "Market implied volatility",
            "mean": iv_comparison_df["market_iv"].mean(),
            "median": iv_comparison_df["market_iv"].median(),
            "std": iv_comparison_df["market_iv"].std(),
            "min": iv_comparison_df["market_iv"].min(),
            "max": iv_comparison_df["market_iv"].max(),
        },
        {
            "series": "RSBS price-implied volatility",
            "mean": iv_comparison_df["rsbs_iv"].mean(),
            "median": iv_comparison_df["rsbs_iv"].median(),
            "std": iv_comparison_df["rsbs_iv"].std(),
            "min": iv_comparison_df["rsbs_iv"].min(),
            "max": iv_comparison_df["rsbs_iv"].max(),
        },
        {
            "series": "RSBS IV - Market IV",
            "mean": iv_comparison_df["iv_error"].mean(),
            "median": iv_comparison_df["iv_error"].median(),
            "std": iv_comparison_df["iv_error"].std(),
            "min": iv_comparison_df["iv_error"].min(),
            "max": iv_comparison_df["iv_error"].max(),
        },
        {
            "series": "|RSBS IV - Market IV|",
            "mean": iv_comparison_df["abs_iv_error"].mean(),
            "median": iv_comparison_df["abs_iv_error"].median(),
            "std": iv_comparison_df["abs_iv_error"].std(),
            "min": iv_comparison_df["abs_iv_error"].min(),
            "max": iv_comparison_df["abs_iv_error"].max(),
        },
    ]
)

display(iv_comparison_summary)


# In[34]:


fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    iv_comparison_df["date"],
    iv_comparison_df["market_iv"],
    label="Market implied volatility",
)

ax.plot(
    iv_comparison_df["date"],
    iv_comparison_df["rsbs_iv"],
    label="RSBS price-implied volatility",
)

ax.set_title("ATM option: market implied volatility vs RSBS price-implied volatility")
ax.set_xlabel("Date")
ax.set_ylabel("Annualized implied volatility")
ax.legend()

apply_bsic_colors_only(fig, ax)
plt.tight_layout()
plt.show()


# ## HMM training-window robustness check
# 
# This section performs a robustness check on the HMM estimation window used in the walk-forward RSBS pricing exercise. Different rolling windows are tested, ranging from short windows of a few months to longer windows of several years.
# 
# For each window, the HMM is re-estimated using only information available before each pricing month. The resulting regime volatilities, transition intensities, and predicted probabilities are then used to price the ATM 30-day option through the RSBS PDE. The models are compared using MAE, RMSE, signed pricing errors, and relative errors.
# 
# This allows us to assess whether the pricing results are driven by a specific arbitrary training-window choice or whether a longer and more stable HMM estimation window improves out-of-sample performance.

# In[35]:


hmm_window_grid = [
    {"rolling_window": 2, "rolling_window_unit": "years", "min_train_obs": 252},
    {"rolling_window": 3, "rolling_window_unit": "years", "min_train_obs": 252},
    {"rolling_window": 5, "rolling_window_unit": "years", "min_train_obs": 252},
]

hmm_window_results = []

for spec in hmm_window_grid:
    print(f"Testing HMM window: {spec['rolling_window']} {spec['rolling_window_unit']}")

    try:
        wf_test, snapshots_test, failed_test = monthly_walkforward_regime_estimates(
            model_df=model_df,
            start_date="2018-01-01",
            end_date="2024-12-31",
            initial_train_start="2015-01-05",
            random_state=42,
            verbose=False,
            window_type="rolling",
            rolling_window=spec["rolling_window"],
            rolling_window_unit=spec["rolling_window_unit"],
            min_train_obs=spec["min_train_obs"],
            n_components=N_COMPONENTS,
        )

        atm_option_df_test = prepare_atm_option_pricing_df(
            final_df=final_df,
            option_price_col="option_price",
            option_type=OPTION_TYPE,
        )

        atm_pricing_df_test = merge_atm_options_with_walkforward(
            atm_option_df=atm_option_df_test,
            walkforward_regime_df=wf_test,
        )

        priced_test = price_atm_options_walkforward(
            atm_pricing_df_test,
            n_S=150,
            n_t=150,
            progress_every=None,
        )

        valid_test = priced_test.dropna(
            subset=["rs_price", "market_price"]
        ).copy()

        valid_test = valid_test[
            valid_test["success"]
            & (valid_test["market_price"] > 0)
            & (valid_test["T"] > 0)
        ].copy()

        avg_extra_metrics = {}

        for col in get_regime_sigma_columns(valid_test):
            avg_extra_metrics[f"avg_{col}"] = valid_test[col].mean()

        for col in get_regime_probability_columns(valid_test, prob_type="pred"):
            avg_extra_metrics[f"avg_{col}"] = valid_test[col].mean()

        hmm_window_results.append(
            {
                "rolling_window": spec["rolling_window"],
                "rolling_window_unit": spec["rolling_window_unit"],
                "min_train_obs": spec["min_train_obs"],
                "n_obs": len(valid_test),
                "failed_months": len(failed_test),
                "MAE": valid_test["abs_error"].mean(),
                "RMSE": np.sqrt((valid_test["pricing_error"] ** 2).mean()),
                "mean_error": valid_test["pricing_error"].mean(),
                "median_error": valid_test["pricing_error"].median(),
                "mean_relative_error": valid_test["relative_error"].mean(),
                "median_relative_error": valid_test["relative_error"].median(),
                "mean_model_price": valid_test["rs_price"].mean(),
                "mean_market_price": valid_test["market_price"].mean(),
                "mean_T_days": (valid_test["T"] * 365).mean(),
                "median_T_days": (valid_test["T"] * 365).median(),
                "mean_strike": valid_test["strike"].mean(),
                "n_components": N_COMPONENTS,
                **avg_extra_metrics,
            }
        )

    except Exception as exc:
        hmm_window_results.append(
            {
                "rolling_window": spec["rolling_window"],
                "rolling_window_unit": spec["rolling_window_unit"],
                "min_train_obs": spec["min_train_obs"],
                "n_obs": 0,
                "failed_months": np.nan,
                "MAE": np.nan,
                "RMSE": np.nan,
                "mean_error": np.nan,
                "median_error": np.nan,
                "mean_relative_error": np.nan,
                "median_relative_error": np.nan,
                "mean_model_price": np.nan,
                "mean_market_price": np.nan,
                "mean_T_days": np.nan,
                "median_T_days": np.nan,
                "mean_strike": np.nan,
                "error": repr(exc),
                "n_components": N_COMPONENTS,
            }
        )

hmm_window_results_df = (
    pd.DataFrame(hmm_window_results)
    .sort_values("MAE")
    .reset_index(drop=True)
)

display(hmm_window_results_df)


# ## Statistical significance of the ATM RSBS versus BS difference
# 
# The previous table reports point estimates of MAE and RMSE. A lower MAE for RSBS is economically suggestive, but it does not by itself show whether the improvement is statistically distinguishable from sampling noise.
# 
# This section therefore treats the comparison as a paired forecast-accuracy problem. For each date $t$, both models price the same ATM option, so the relevant object is the paired loss differential:
# 
# $$
# d_t^{MAE} = |e_t^{RSBS}| - |e_t^{BS}|,
# $$
# 
# where $e_t^{model}=C_t^{model}-C_t^{mkt}$. A negative mean loss differential means that RSBS has lower absolute pricing error than the Black-Scholes benchmark.
# 
# Two complementary tests are reported:
# 
# 1. a paired $t$-test on the absolute-error differential, using the daily paired observations;
# 2. a Diebold-Mariano-style test with a Newey-West/HAC correction, which is more appropriate when daily loss differentials are serially correlated.
# 
# The same Diebold-Mariano calculation is also reported for squared-error loss, so that the RMSE comparison is tested directly.
# 

# In[47]:


# ================================================================
# Statistical significance tests: ATM RSBS versus rolling-vol BS
# ================================================================

from scipy.stats import t as student_t






atm_test_df = valid_comparison.dropna(
    subset=["rs_error", "bs_error", "rs_abs_error", "bs_abs_error", "market_price"]
).copy()

atm_test_df = atm_test_df[
    (atm_test_df["market_price"] > 0) & (atm_test_df["success"])
].copy()

atm_test_df["abs_loss_diff_rs_minus_bs"] = (
    atm_test_df["rs_abs_error"] - atm_test_df["bs_abs_error"]
)

atm_test_df["squared_loss_diff_rs_minus_bs"] = (
    atm_test_df["rs_error"] ** 2 - atm_test_df["bs_error"] ** 2
)

n_atm_tests = len(atm_test_df)
mean_abs_diff = atm_test_df["abs_loss_diff_rs_minus_bs"].mean()
std_abs_diff = atm_test_df["abs_loss_diff_rs_minus_bs"].std(ddof=1)
se_abs_diff = std_abs_diff / np.sqrt(n_atm_tests)
paired_t_stat = mean_abs_diff / se_abs_diff
paired_t_p_two_sided = 2.0 * (1.0 - student_t.cdf(abs(paired_t_stat), df=n_atm_tests - 1))
paired_t_p_less = student_t.cdf(paired_t_stat, df=n_atm_tests - 1)
paired_t_ci_95 = (
    mean_abs_diff - student_t.ppf(0.975, df=n_atm_tests - 1) * se_abs_diff,
    mean_abs_diff + student_t.ppf(0.975, df=n_atm_tests - 1) * se_abs_diff,
)

mae_rsbs = atm_test_df["rs_abs_error"].mean()
mae_bs = atm_test_df["bs_abs_error"].mean()
rmse_rsbs = np.sqrt((atm_test_df["rs_error"] ** 2).mean())
rmse_bs = np.sqrt((atm_test_df["bs_error"] ** 2).mean())
mean_market_price = atm_test_df["market_price"].mean()

mae_improvement = mae_bs - mae_rsbs
mae_improvement_pct_vs_bs = mae_improvement / mae_bs
mae_improvement_pct_vs_market = mae_improvement / mean_market_price
rmse_improvement = rmse_bs - rmse_rsbs
rmse_improvement_pct_vs_bs = rmse_improvement / rmse_bs

mae_dm_two_sided = diebold_mariano_hac_test(
    atm_test_df["rs_abs_error"],
    atm_test_df["bs_abs_error"],
    alternative="two-sided",
)

mae_dm_less = diebold_mariano_hac_test(
    atm_test_df["rs_abs_error"],
    atm_test_df["bs_abs_error"],
    alternative="less",
)

rmse_dm_two_sided = diebold_mariano_hac_test(
    atm_test_df["rs_error"] ** 2,
    atm_test_df["bs_error"] ** 2,
    alternative="two-sided",
)

rmse_dm_less = diebold_mariano_hac_test(
    atm_test_df["rs_error"] ** 2,
    atm_test_df["bs_error"] ** 2,
    alternative="less",
)

atm_statistical_test_summary = pd.DataFrame(
    [
        {
            "comparison": "Economic magnitude",
            "loss": "Absolute error / MAE",
            "n_obs": n_atm_tests,
            "RSBS": mae_rsbs,
            "BS": mae_bs,
            "BS_minus_RSBS": mae_improvement,
            "% improvement vs BS": mae_improvement_pct_vs_bs,
            "% of mean market price": mae_improvement_pct_vs_market,
            "test_stat": np.nan,
            "p_value_two_sided": np.nan,
            "p_value_one_sided_RSBS_better": np.nan,
            "HAC_lag": np.nan,
        },
        {
            "comparison": "Economic magnitude",
            "loss": "Squared error / RMSE",
            "n_obs": n_atm_tests,
            "RSBS": rmse_rsbs,
            "BS": rmse_bs,
            "BS_minus_RSBS": rmse_improvement,
            "% improvement vs BS": rmse_improvement_pct_vs_bs,
            "% of mean market price": rmse_improvement / mean_market_price,
            "test_stat": np.nan,
            "p_value_two_sided": np.nan,
            "p_value_one_sided_RSBS_better": np.nan,
            "HAC_lag": np.nan,
        },
        {
            "comparison": "Paired t-test",
            "loss": "|RSBS error| - |BS error|",
            "n_obs": n_atm_tests,
            "RSBS": mae_rsbs,
            "BS": mae_bs,
            "BS_minus_RSBS": mae_improvement,
            "% improvement vs BS": mae_improvement_pct_vs_bs,
            "% of mean market price": mae_improvement_pct_vs_market,
            "test_stat": paired_t_stat,
            "p_value_two_sided": paired_t_p_two_sided,
            "p_value_one_sided_RSBS_better": paired_t_p_less,
            "HAC_lag": np.nan,
        },
        {
            "comparison": "Diebold-Mariano HAC",
            "loss": "Absolute error",
            "n_obs": mae_dm_two_sided["n_obs"],
            "RSBS": mae_rsbs,
            "BS": mae_bs,
            "BS_minus_RSBS": mae_improvement,
            "% improvement vs BS": mae_improvement_pct_vs_bs,
            "% of mean market price": mae_improvement_pct_vs_market,
            "test_stat": mae_dm_two_sided["test_stat"],
            "p_value_two_sided": mae_dm_two_sided["p_value"],
            "p_value_one_sided_RSBS_better": mae_dm_less["p_value"],
            "HAC_lag": mae_dm_two_sided["hac_lag"],
        },
        {
            "comparison": "Diebold-Mariano HAC",
            "loss": "Squared error",
            "n_obs": rmse_dm_two_sided["n_obs"],
            "RSBS": rmse_rsbs,
            "BS": rmse_bs,
            "BS_minus_RSBS": rmse_improvement,
            "% improvement vs BS": rmse_improvement_pct_vs_bs,
            "% of mean market price": rmse_improvement / mean_market_price,
            "test_stat": rmse_dm_two_sided["test_stat"],
            "p_value_two_sided": rmse_dm_two_sided["p_value"],
            "p_value_one_sided_RSBS_better": rmse_dm_less["p_value"],
            "HAC_lag": rmse_dm_two_sided["hac_lag"],
        },
    ]
)

print("ATM RSBS versus rolling-volatility BS statistical comparison")
print("-" * 80)
print(
    "Loss differential is RSBS loss minus BS loss. "
    "Negative values favour RSBS."
)
print(
    f"Paired 95% CI for mean absolute-error differential: "
    f"[{paired_t_ci_95[0]:.4f}, {paired_t_ci_95[1]:.4f}]"
)

display(atm_statistical_test_summary)


# # 6. Out-of-the-money option pricing analysis
# 
# This section repeats the pricing exercise for out-of-the-money SPX call options with target moneyness
# 
# $$
# \frac{S_t}{K_t} \approx 0.90.
# $$
# 
# Since these are call options, the strike is above the spot index level and the option is out of the money at the valuation date.
# 
# The goal is to test whether the historical regime-switching framework remains informative for options that are more sensitive to tail movements, volatility skew, and risk-neutral expectations. The same walk-forward HMM regime estimates are used as in the ATM analysis.
# 

# In[36]:


OTM_DATA_FILE = find_data_file("option_price_otm_90.csv")
OTM_LABEL = "OTM 90%"
USE_ACTUAL_EXPIRY_OTM = False  # bachelor thesis setting: fixed 30-day maturity

print(f"OTM data file: {OTM_DATA_FILE}")

otm_df = pd.read_csv(
    OTM_DATA_FILE,
    sep=";",
    decimal=","
).copy()

otm_df["date"] = pd.to_datetime(otm_df["date"], errors="coerce")
otm_df["expiry_date"] = pd.to_datetime(otm_df["expiry_date"], errors="coerce")

for col in ["spx", "r_3m", "strike", "option_price"]:
    otm_df[col] = pd.to_numeric(otm_df[col], errors="coerce")

otm_df = (
    otm_df[["date", "spx", "r_3m", "strike", "expiry_date", "option_price"]]
    .dropna()
    .sort_values("date")
    .drop_duplicates(subset=["date"], keep="last")
    .reset_index(drop=True)
)

otm_df["moneyness"] = otm_df["spx"] / otm_df["strike"]
otm_df["days_to_expiry"] = (otm_df["expiry_date"] - otm_df["date"]).dt.days

print("OTM dataset:")
print(otm_df.shape)
display(otm_df.head())

display(
    otm_df[["option_price", "moneyness", "days_to_expiry"]].describe()
)


# ## Preparing OTM option inputs
# 
# The pricing dataframe is constructed in the same way as in the ATM exercise. For each date, the model uses the observed index level, strike, short-term interest rate, and market option price.
# 
# The OTM thesis specification fixes maturity at 30 calendar days (`USE_ACTUAL_EXPIRY_OTM = False`). The switch is left in the notebook so the exercise can be rerun with the observed `expiry_date` if required.
# 

# In[37]:




otm_option_df = prepare_option_pricing_df(
    df=otm_df,
    option_price_col="option_price",
    option_type=OPTION_TYPE,
    use_actual_expiry=USE_ACTUAL_EXPIRY_OTM,
    maturity_days=ATM_MATURITY_DAYS,
)

otm_pricing_df = merge_atm_options_with_walkforward(
    atm_option_df=otm_option_df,
    walkforward_regime_df=walkforward_regime_df,
)

print("OTM option dataframe:")
print(otm_option_df.shape)

print("OTM pricing dataframe after merging with walk-forward regimes:")
print(otm_pricing_df.shape)

display(
    otm_pricing_df[
        get_existing_columns(
            otm_pricing_df,
            [
                "date",
                "spx",
                "strike",
                "moneyness",
                "T",
                "r_3m",
                "market_price",
                "sigma_low",
                "sigma_high",
                "p_low_pred",
                "p_high_pred",
            ],
        )
    ].head()
)


# ## Original RSBS and rolling-volatility Black-Scholes pricing
# 
# The first OTM pricing exercise uses exactly the same regime-switching Black-Scholes machinery as before. The HMM regime volatilities, transition intensities, and one-step-ahead predicted regime probabilities are merged with the OTM option data by date.
# 
# The original regime-switching price is then compared with a rolling-volatility Black-Scholes benchmark. This benchmark uses the same strike, spot, maturity, and interest rate, but replaces the regime-switching structure with the 21-day lagged historical volatility.

# In[38]:


priced_otm_options = price_atm_options_walkforward(
    otm_pricing_df,
    n_S=150,
    n_t=150,
    progress_every=50,
)

display(
    priced_otm_options[
        get_existing_columns(
            priced_otm_options,
            [
                "date",
                "spx",
                "strike",
                "moneyness",
                "T",
                "market_price",
                "rs_price",
                "pricing_error",
                "abs_error",
                "relative_error",
                "success",
            ],
        )
    ].head()
)


otm_comparison_df = priced_otm_options.copy()
otm_comparison_df["date"] = pd.to_datetime(otm_comparison_df["date"])

otm_comparison_df = otm_comparison_df.merge(
    bs_vol_df[["date", "bs_rolling_vol"]],
    on="date",
    how="left",
)

otm_comparison_df["bs_price"] = otm_comparison_df.apply(
    lambda row: black_scholes_price(
        S0=row["spx"],
        K=row["strike"],
        T=row["T"],
        r=row["r_3m"],
        sigma=row["bs_rolling_vol"],
        option_type=row["option_type"],
    ),
    axis=1,
)

otm_comparison_df["rs_error"] = (
    otm_comparison_df["rs_price"] - otm_comparison_df["market_price"]
)
otm_comparison_df["bs_error"] = (
    otm_comparison_df["bs_price"] - otm_comparison_df["market_price"]
)

otm_comparison_df["rs_abs_error"] = otm_comparison_df["rs_error"].abs()
otm_comparison_df["bs_abs_error"] = otm_comparison_df["bs_error"].abs()

otm_comparison_df["rs_relative_error"] = (
    otm_comparison_df["rs_abs_error"] / otm_comparison_df["market_price"]
)
otm_comparison_df["bs_relative_error"] = (
    otm_comparison_df["bs_abs_error"] / otm_comparison_df["market_price"]
)

valid_otm_comparison = otm_comparison_df.dropna(
    subset=["market_price", "rs_price", "bs_price"]
).copy()

valid_otm_comparison = valid_otm_comparison[
    valid_otm_comparison["success"]
    & (valid_otm_comparison["market_price"] > 0)
].copy()

print("Valid OTM comparison observations:", len(valid_otm_comparison))
display(valid_otm_comparison.head())


otm_summary = pd.DataFrame(
    [
        summarize_model_errors(
            valid_otm_comparison,
            "Regime-Switching BS",
            "rs_price",
            "rs_error",
            "rs_abs_error",
            "rs_relative_error",
        ),
        summarize_model_errors(
            valid_otm_comparison,
            "Rolling-volatility BS",
            "bs_price",
            "bs_error",
            "bs_abs_error",
            "bs_relative_error",
        ),
    ]
)

display(otm_summary)


otm_extra_summary = pd.DataFrame(
    [
        {
            "dataset": OTM_LABEL,
            "n_obs": len(valid_otm_comparison),
            "mean_moneyness": valid_otm_comparison["moneyness"].mean(),
            "median_moneyness": valid_otm_comparison["moneyness"].median(),
            "mean_T_days": (valid_otm_comparison["T"] * 365).mean(),
            "median_T_days": (valid_otm_comparison["T"] * 365).median(),
            "mean_market_price": valid_otm_comparison["market_price"].mean(),
            "mean_RSBS_price": valid_otm_comparison["rs_price"].mean(),
            "mean_BS_price": valid_otm_comparison["bs_price"].mean(),
            "RSBS_MAE": valid_otm_comparison["rs_abs_error"].mean(),
            "BS_MAE": valid_otm_comparison["bs_abs_error"].mean(),
            "RSBS_RMSE": np.sqrt((valid_otm_comparison["rs_error"] ** 2).mean()),
            "BS_RMSE": np.sqrt((valid_otm_comparison["bs_error"] ** 2).mean()),
        }
    ]
)

display(otm_extra_summary)


# ## Price and error diagnostics
# 
# The next plots compare the observed OTM market price with the original RSBS price and the rolling-volatility Black-Scholes benchmark.
# 
# Because OTM options often have low market prices, relative errors can become mechanically large. For this reason, the analysis focuses not only on relative errors, but also on absolute errors and rolling average absolute errors.

# In[39]:


fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    valid_otm_comparison["date"],
    valid_otm_comparison["market_price"],
    label=f"Market {OTM_LABEL} option price",
)

ax.plot(
    valid_otm_comparison["date"],
    valid_otm_comparison["rs_price"],
    label=f"{N_COMPONENTS}-state RSBS PDE price",
)

ax.plot(
    valid_otm_comparison["date"],
    valid_otm_comparison["bs_price"],
    label=f"BS price ({ROLLING_BS_WINDOW}D rolling volatility)",
    alpha=0.85,
)

ax.set_title(f"{OTM_LABEL} SPX option: market price vs RSBS and Black-Scholes benchmark")
ax.set_xlabel("Date")
ax.set_ylabel("Option price")
ax.legend()

apply_bsic_colors_only(fig, ax)
plt.tight_layout()
plt.show()


valid_otm_comparison["rs_abs_error_roll"] = (
    valid_otm_comparison["rs_abs_error"]
    .rolling(ERROR_ROLLING_WINDOW)
    .mean()
)

valid_otm_comparison["bs_abs_error_roll"] = (
    valid_otm_comparison["bs_abs_error"]
    .rolling(ERROR_ROLLING_WINDOW)
    .mean()
)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    valid_otm_comparison["date"],
    valid_otm_comparison["rs_abs_error_roll"],
    label=f"RSBS {ERROR_ROLLING_WINDOW}D average absolute error",
)

ax.plot(
    valid_otm_comparison["date"],
    valid_otm_comparison["bs_abs_error_roll"],
    label=f"BS {ERROR_ROLLING_WINDOW}D average absolute error",
)

ax.set_title(f"{OTM_LABEL} SPX option: rolling average absolute pricing error")
ax.set_xlabel("Date")
ax.set_ylabel("Average absolute error")
ax.legend()

apply_bsic_colors_only(fig, ax)
plt.tight_layout()
plt.show()


# ## Threshold analysis for low-priced OTM options
# 
# Very cheap OTM options can generate mechanically large relative errors. The table below repeats the RSBS versus rolling-volatility Black-Scholes comparison after excluding observations with market prices below 0.5, 1.0, and 2.0 index points.
# 

# In[40]:


threshold_results = []

for threshold in [0.5, 1.0, 2.0]:
    tmp = valid_otm_comparison[
        valid_otm_comparison["market_price"] >= threshold
    ].copy()

    threshold_results.append(
        summarize_model_errors(
            tmp,
            f"RSBS, market >= {threshold}",
            "rs_price",
            "rs_error",
            "rs_abs_error",
            "rs_relative_error",
        )
    )

    threshold_results.append(
        summarize_model_errors(
            tmp,
            f"Rolling BS, market >= {threshold}",
            "bs_price",
            "bs_error",
            "bs_abs_error",
            "bs_relative_error",
        )
    )

threshold_otm_summary = pd.DataFrame(threshold_results)
display(threshold_otm_summary)


# ## Implied-volatility comparison: market price versus RSBS price
# 
# The final OTM diagnostic compares the Black-Scholes implied volatility backed out from the observed market price with the Black-Scholes implied volatility backed out from the regime-switching Black-Scholes price.
# 
# For each date $t$, the implied volatility is computed by inverting the Black-Scholes formula using the same option inputs used in the OTM pricing exercise:
# 
# $$
# S_t,\quad K_t,\quad T_t,\quad r_t.
# $$
# 
# In the OTM specification, $T_t$ is fixed at 30 calendar days because `USE_ACTUAL_EXPIRY_OTM = False`. If the switch is set to `True`, the same code instead uses the observed maturity implied by `expiry_date`.
# 
# The two implied-volatility series are therefore
# 
# $$
# \sigma^{mkt,IV}_t
# =
# BS^{-1}
# \left(
# C_t^{mkt}; S_t,K_t,T_t,r_t
# \right),
# $$
# 
# and
# 
# $$
# \sigma^{RSBS,IV}_t
# =
# BS^{-1}
# \left(
# C_t^{RSBS}; S_t,K_t,T_t,r_t
# \right).
# $$
# 
# This diagnostic evaluates whether the RSBS price embeds too much or too little volatility relative to the market price. The rolling-volatility Black-Scholes benchmark is also shown as a reference, but the main comparison is between market implied volatility and RSBS price-implied volatility.

# In[41]:


required_otm_iv_cols = [
    "date",
    "spx",
    "strike",
    "T",
    "r_3m",
    "market_price",
    "rs_price",
    "bs_price",
    "bs_rolling_vol",
    "option_type",
]

missing_otm_iv_cols = [
    col for col in required_otm_iv_cols
    if col not in valid_otm_comparison.columns
]

if missing_otm_iv_cols:
    raise ValueError(
        "The following columns are required for the OTM IV comparison "
        f"but are missing from valid_otm_comparison: {missing_otm_iv_cols}"
    )

valid_otm_comparison = valid_otm_comparison.copy()
valid_otm_comparison["date"] = pd.to_datetime(valid_otm_comparison["date"], errors="coerce")

# Check that the IV inversion uses the same maturity T used in OTM pricing.
if "days_to_expiry" in valid_otm_comparison.columns:
    valid_otm_comparison["T_from_days_to_expiry"] = (
        valid_otm_comparison["days_to_expiry"] / 365.0
    )

    max_T_difference = (
        valid_otm_comparison["T"] - valid_otm_comparison["T_from_days_to_expiry"]
    ).abs().max()

    print("OTM implied-volatility input consistency check")
    print("-" * 80)
    print("The IV inversion uses the same S, K, T, and r used for OTM pricing.")
    print(f"USE_ACTUAL_EXPIRY_OTM: {USE_ACTUAL_EXPIRY_OTM}")
    print(f"Maximum |T - days_to_expiry / 365|: {max_T_difference:.12f}")
    print(f"Mean selected maturity in days: {valid_otm_comparison['days_to_expiry'].mean():.2f}")
    print(f"Median selected maturity in days: {valid_otm_comparison['days_to_expiry'].median():.2f}")

    if max_T_difference > 1e-10:
        raise ValueError(
            "Mismatch between stored T and days_to_expiry / 365. "
            "Check prepare_option_pricing_df."
        )

if "expiry_date" in valid_otm_comparison.columns:
    valid_otm_comparison["expiry_date"] = pd.to_datetime(
        valid_otm_comparison["expiry_date"],
        errors="coerce",
    )

    valid_otm_comparison["actual_days_to_expiry"] = (
        valid_otm_comparison["expiry_date"] - valid_otm_comparison["date"]
    ).dt.days

    print(f"Mean actual expiry maturity in days: {valid_otm_comparison['actual_days_to_expiry'].mean():.2f}")
    print(f"Median actual expiry maturity in days: {valid_otm_comparison['actual_days_to_expiry'].median():.2f}")

valid_otm_comparison["market_iv_otm"] = valid_otm_comparison.apply(
    lambda row: black_scholes_implied_vol(
        market_price=row["market_price"],
        S0=row["spx"],
        K=row["strike"],
        T=row["T"],
        r=row["r_3m"],
        option_type=row["option_type"],
    ),
    axis=1,
)

valid_otm_comparison["rsbs_iv_otm"] = valid_otm_comparison.apply(
    lambda row: black_scholes_implied_vol(
        market_price=row["rs_price"],
        S0=row["spx"],
        K=row["strike"],
        T=row["T"],
        r=row["r_3m"],
        option_type=row["option_type"],
    ),
    axis=1,
)

valid_otm_comparison["bs_iv_otm"] = valid_otm_comparison["bs_rolling_vol"]

iv_plot_df = valid_otm_comparison.dropna(
    subset=[
        "market_iv_otm",
        "rsbs_iv_otm",
        "bs_iv_otm",
    ]
).copy()

iv_plot_df["iv_error_otm"] = (
    iv_plot_df["rsbs_iv_otm"] - iv_plot_df["market_iv_otm"]
)

iv_plot_df["abs_iv_error_otm"] = iv_plot_df["iv_error_otm"].abs()

print("\nValid OTM implied-volatility comparison observations:", len(iv_plot_df))

display(
    iv_plot_df[
        get_existing_columns(
            iv_plot_df,
            [
                "date",
                "spx",
                "strike",
                "moneyness",
                "days_to_expiry",
                "actual_days_to_expiry",
                "T",
                "r_3m",
                "market_price",
                "rs_price",
                "bs_price",
                "market_iv_otm",
                "rsbs_iv_otm",
                "bs_iv_otm",
                "iv_error_otm",
                "abs_iv_error_otm",
            ],
        )
    ].head()
)

otm_iv_summary = pd.DataFrame(
    [
        {
            "series": "Market implied volatility",
            "mean": iv_plot_df["market_iv_otm"].mean(),
            "median": iv_plot_df["market_iv_otm"].median(),
            "std": iv_plot_df["market_iv_otm"].std(),
            "min": iv_plot_df["market_iv_otm"].min(),
            "max": iv_plot_df["market_iv_otm"].max(),
        },
        {
            "series": "RSBS price-implied volatility",
            "mean": iv_plot_df["rsbs_iv_otm"].mean(),
            "median": iv_plot_df["rsbs_iv_otm"].median(),
            "std": iv_plot_df["rsbs_iv_otm"].std(),
            "min": iv_plot_df["rsbs_iv_otm"].min(),
            "max": iv_plot_df["rsbs_iv_otm"].max(),
        },
        {
            "series": "Rolling BS volatility",
            "mean": iv_plot_df["bs_iv_otm"].mean(),
            "median": iv_plot_df["bs_iv_otm"].median(),
            "std": iv_plot_df["bs_iv_otm"].std(),
            "min": iv_plot_df["bs_iv_otm"].min(),
            "max": iv_plot_df["bs_iv_otm"].max(),
        },
        {
            "series": "RSBS IV - Market IV",
            "mean": iv_plot_df["iv_error_otm"].mean(),
            "median": iv_plot_df["iv_error_otm"].median(),
            "std": iv_plot_df["iv_error_otm"].std(),
            "min": iv_plot_df["iv_error_otm"].min(),
            "max": iv_plot_df["iv_error_otm"].max(),
        },
        {
            "series": "|RSBS IV - Market IV|",
            "mean": iv_plot_df["abs_iv_error_otm"].mean(),
            "median": iv_plot_df["abs_iv_error_otm"].median(),
            "std": iv_plot_df["abs_iv_error_otm"].std(),
            "min": iv_plot_df["abs_iv_error_otm"].min(),
            "max": iv_plot_df["abs_iv_error_otm"].max(),
        },
    ]
)

display(otm_iv_summary)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    iv_plot_df["date"],
    iv_plot_df["market_iv_otm"],
    label=f"Market implied volatility from {OTM_LABEL} price",
)

ax.plot(
    iv_plot_df["date"],
    iv_plot_df["rsbs_iv_otm"],
    label="RSBS price-implied volatility",
)

ax.plot(
    iv_plot_df["date"],
    iv_plot_df["bs_iv_otm"],
    label=f"Rolling BS volatility ({ROLLING_BS_WINDOW}D)",
    alpha=0.85,
)

ax.set_title(f"{OTM_LABEL} option: market IV vs RSBS-implied IV vs rolling BS volatility")
ax.set_xlabel("Date")
ax.set_ylabel("Annualized implied volatility")
ax.legend()

apply_bsic_colors_only(fig, ax)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    iv_plot_df["date"],
    iv_plot_df["iv_error_otm"],
    label="RSBS implied volatility - market implied volatility",
)

ax.axhline(0, linestyle="--", linewidth=1)

ax.set_title(f"{OTM_LABEL} option: implied-volatility error of RSBS model")
ax.set_xlabel("Date")
ax.set_ylabel("Implied-volatility error")
ax.legend()

apply_bsic_colors_only(fig, ax)
plt.tight_layout()
plt.show()


# ## Walk-forward OTM volatility adjustment
# 
# The implied-volatility comparison above indicates whether the original RSBS price embeds too much or too little volatility relative to the OTM market price.
# 
# To test this directly, this section introduces a simple walk-forward volatility adjustment. For each month, the adjustment coefficient is computed as the median ratio between market implied volatility and RSBS price-implied volatility observed in the previous month:
# 
# $$
# \alpha_m
# =
# \operatorname{median}_{t \in m-1}
# \left(
# \frac{\sigma^{mkt,IV}_t}{\sigma^{RSBS,IV}_t}
# \right).
# $$
# 
# The adjustment is then applied multiplicatively to all HMM regime volatilities during month $m$:
# 
# $$
# \sigma^{adj}_{i,t}
# =
# \alpha_m \sigma_{i,t}.
# $$
# 
# The one-month lag avoids look-ahead bias. The clipping interval prevents the adjustment from becoming too unstable in months with noisy OTM implied-volatility estimates.

# In[42]:


# ================================================================
# Walk-forward OTM volatility adjustment
# ================================================================

valid_otm_comparison["month"] = valid_otm_comparison["date"].dt.to_period("M")

alpha_monthly = (
    valid_otm_comparison
    .replace([np.inf, -np.inf], np.nan)
    .dropna(subset=["market_iv_otm", "rsbs_iv_otm"])
    .assign(alpha_raw=lambda x: x["market_iv_otm"] / x["rsbs_iv_otm"])
    .replace([np.inf, -np.inf], np.nan)
    .dropna(subset=["alpha_raw"])
    .groupby("month")["alpha_raw"]
    .median()
    .shift(1)
    .clip(lower=0.50, upper=1.20)
    .rename("alpha_otm")
    .reset_index()
)

display(alpha_monthly.head())


otm_comparison_df = otm_comparison_df.copy()
otm_comparison_df["date"] = pd.to_datetime(otm_comparison_df["date"], errors="coerce")
otm_comparison_df["month"] = otm_comparison_df["date"].dt.to_period("M")

otm_comparison_alpha_df = otm_comparison_df.merge(
    alpha_monthly,
    on="month",
    how="left",
)

otm_comparison_alpha_df["alpha_otm"] = (
    otm_comparison_alpha_df["alpha_otm"]
    .fillna(1.0)
)



otm_alpha_df = price_otm_options_alpha_adjusted(
    otm_comparison_alpha_df,
    n_S=150,
    n_t=150,
    progress_every=50,
)

valid_otm_alpha = otm_alpha_df.dropna(
    subset=[
        "market_price",
        "rs_price",
        "rs_price_alpha",
        "bs_price",
        "rs_abs_error",
        "rs_abs_error_alpha",
        "bs_abs_error",
    ]
).copy()

valid_otm_alpha = valid_otm_alpha[
    valid_otm_alpha["success"]
    & valid_otm_alpha["success_alpha"]
    & (valid_otm_alpha["market_price"] > 0)
].copy()

print("Valid alpha-adjusted OTM observations:", len(valid_otm_alpha))

display(
    valid_otm_alpha[
        get_existing_columns(
            valid_otm_alpha,
            [
                "date",
                "market_price",
                "rs_price",
                "rs_price_alpha",
                "bs_price",
                "alpha_otm_used",
                "rs_error",
                "rs_error_alpha",
                "bs_error",
                "rs_abs_error",
                "rs_abs_error_alpha",
                "bs_abs_error",
            ],
        )
    ].head()
)


# ## Original versus alpha-adjusted RSBS
# 
# The adjusted model is compared with both the original RSBS price and the rolling-volatility Black-Scholes benchmark.
# 
# This comparison separates two effects. The original RSBS model tests whether historical regime-switching volatility alone is sufficient for OTM pricing. The alpha-adjusted version tests whether a simple lagged correction based on option-implied information can improve OTM pricing without using same-day market prices.
# 

# In[43]:


otm_alpha_summary = pd.DataFrame(
    [
        summarize_model_errors(
            valid_otm_alpha,
            "Original RSBS",
            "rs_price",
            "rs_error",
            "rs_abs_error",
            "rs_relative_error",
        ),
        summarize_model_errors(
            valid_otm_alpha,
            "Alpha-adjusted RSBS",
            "rs_price_alpha",
            "rs_error_alpha",
            "rs_abs_error_alpha",
            "rs_relative_error_alpha",
        ),
        summarize_model_errors(
            valid_otm_alpha,
            "Rolling-volatility BS",
            "bs_price",
            "bs_error",
            "bs_abs_error",
            "bs_relative_error",
        ),
    ]
)

display(otm_alpha_summary)


main_error_metrics = otm_alpha_summary.set_index("model")[["MAE", "RMSE"]]

fig, ax = plt.subplots(figsize=(8, 4))

main_error_metrics.plot(kind="bar", ax=ax)

ax.set_title(f"{OTM_LABEL} option: MAE and RMSE comparison")
ax.set_xlabel("Model")
ax.set_ylabel("Pricing error")
ax.tick_params(axis="x", rotation=20)
ax.legend(title="Metric")

apply_bsic_colors_only(fig, ax)
plt.tight_layout()
plt.show()


relative_error_metrics = otm_alpha_summary.set_index("model")[
    ["mean_relative_error", "median_relative_error"]
]

fig, ax = plt.subplots(figsize=(8, 4))

relative_error_metrics.plot(kind="bar", ax=ax)

ax.set_title(f"{OTM_LABEL} option: relative error comparison")
ax.set_xlabel("Model")
ax.set_ylabel("Relative error")
ax.tick_params(axis="x", rotation=20)
ax.legend(title="Metric")

apply_bsic_colors_only(fig, ax)
plt.tight_layout()
plt.show()


# In[44]:


fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    valid_otm_alpha["date"],
    valid_otm_alpha["market_price"],
    label=f"Market {OTM_LABEL} option price",
)

ax.plot(
    valid_otm_alpha["date"],
    valid_otm_alpha["rs_price_alpha"],
    label="Alpha-adjusted RSBS price",
)

ax.plot(
    valid_otm_alpha["date"],
    valid_otm_alpha["bs_price"],
    label=f"BS price ({ROLLING_BS_WINDOW}D rolling volatility)",
    alpha=0.85,
)

ax.set_title(f"{OTM_LABEL} option: market price vs original and adjusted models")
ax.set_xlabel("Date")
ax.set_ylabel("Option price")
ax.legend()

apply_bsic_colors_only(fig, ax)
plt.tight_layout()
plt.show()

valid_otm_alpha["rs_abs_error_alpha_roll"] = (
    valid_otm_alpha["rs_abs_error_alpha"]
    .rolling(ERROR_ROLLING_WINDOW)
    .mean()
)

valid_otm_alpha["bs_abs_error_roll"] = (
    valid_otm_alpha["bs_abs_error"]
    .rolling(ERROR_ROLLING_WINDOW)
    .mean()
)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    valid_otm_alpha["date"],
    valid_otm_alpha["rs_abs_error_alpha_roll"],
    label="Alpha-adjusted RSBS rolling absolute error",
)

ax.plot(
    valid_otm_alpha["date"],
    valid_otm_alpha["bs_abs_error_roll"],
    label="Rolling-volatility BS rolling absolute error",
)

ax.set_title(f"{OTM_LABEL} option: rolling average absolute pricing error")
ax.set_xlabel("Date")
ax.set_ylabel("Average absolute error")
ax.legend()

apply_bsic_colors_only(fig, ax)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    valid_otm_alpha["date"],
    valid_otm_alpha["alpha_otm_used"],
    label=r"OTM volatility adjustment $\alpha_t$",
)

ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1,
    label="No adjustment",
)

ax.set_title(r"Walk-forward OTM volatility adjustment $\alpha_t$")
ax.set_xlabel("Date")
ax.set_ylabel(r"$\alpha_t$")
ax.legend()

apply_bsic_colors_only(fig, ax)
plt.tight_layout()
plt.show()


# ## Threshold analysis
# 
# Since many OTM options have very low prices, relative errors can become extremely large even when the absolute error is small. As a robustness check, the comparison is repeated after excluding options whose market price is below different thresholds.
# 
# This helps distinguish genuine model misspecification from the mechanical effect of dividing by very small market prices.

# In[45]:


threshold_results = []

for threshold in [0.5, 1.0, 2.0]:
    tmp = valid_otm_alpha[
        valid_otm_alpha["market_price"] >= threshold
    ].copy()

    threshold_results.append(
        summarize_model_errors(
            tmp,
            f"Original RSBS, market >= {threshold}",
            "rs_price",
            "rs_error",
            "rs_abs_error",
            "rs_relative_error",
        )
    )

    threshold_results.append(
        summarize_model_errors(
            tmp,
            f"Alpha-adjusted RSBS, market >= {threshold}",
            "rs_price_alpha",
            "rs_error_alpha",
            "rs_abs_error_alpha",
            "rs_relative_error_alpha",
        )
    )

    threshold_results.append(
        summarize_model_errors(
            tmp,
            f"Rolling BS, market >= {threshold}",
            "bs_price",
            "bs_error",
            "bs_abs_error",
            "bs_relative_error",
        )
    )

threshold_summary = pd.DataFrame(threshold_results)

display(threshold_summary)


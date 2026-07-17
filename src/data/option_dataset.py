"""Build the option-price datasets used by the bachelor thesis.

The module contains the data-loading, filtering, option-selection, and CSV
writing logic for the SPX option datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class OptionSelectionConfig:
    """Configuration for selecting one SPX option per trading date."""

    target_moneyness: float = 1.00
    target_days_to_expiry: int = 30
    min_days_to_expiry: int = 1
    option_type: str = "C"
    min_mid_price: float = 0.05
    min_open_interest: int = 1
    min_volume: int = 1
    moneyness_upper_bound: float = 1.00
    atm_lower_bound: float = 0.98
    atm_upper_bound: float = 1.02


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    return df


def load_base_dataset(path: Path) -> pd.DataFrame:
    """Load and clean the SPX/risk-free-rate base dataset."""
    base_df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        low_memory=False,
    ).copy()
    base_df = _normalise_columns(base_df)

    required_cols = ["date", "spx", "r_3m"]
    missing_cols = [col for col in required_cols if col not in base_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in base file: {missing_cols}. "
            f"Available columns are: {base_df.columns.tolist()}"
        )

    base_df["date"] = pd.to_datetime(base_df["date"], errors="coerce")
    base_df["spx"] = pd.to_numeric(base_df["spx"], errors="coerce")
    base_df["r_3m"] = pd.to_numeric(base_df["r_3m"], errors="coerce")

    return (
        base_df[["date", "spx", "r_3m"]]
        .dropna(subset=["date", "spx", "r_3m"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )


def load_option_chain(path: Path) -> pd.DataFrame:
    """Load and type-cast the raw WRDS OptionMetrics option chain."""
    options = pd.read_csv(path, encoding="latin1", low_memory=False).copy()
    options = _normalise_columns(options)

    required_cols = [
        "date",
        "exdate",
        "cp_flag",
        "strike_price",
        "best_bid",
        "best_offer",
    ]
    missing_cols = [col for col in required_cols if col not in options.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in options CSV: {missing_cols}")

    options["date"] = pd.to_datetime(options["date"], errors="coerce")
    options["exdate"] = pd.to_datetime(options["exdate"], errors="coerce")
    options["cp_flag"] = options["cp_flag"].astype(str).str.upper().str.strip()

    numeric_cols = [
        "strike_price",
        "best_bid",
        "best_offer",
        "volume",
        "open_interest",
        "impl_volatility",
        "delta",
        "vega",
        "forward_price",
    ]
    for col in numeric_cols:
        if col in options.columns:
            options[col] = pd.to_numeric(options[col], errors="coerce")

    options["strike"] = options["strike_price"] / 1000
    options["mid_price"] = (options["best_bid"] + options["best_offer"]) / 2
    options["days_to_expiry"] = (options["exdate"] - options["date"]).dt.days
    return options


def filter_option_chain(options: pd.DataFrame, config: OptionSelectionConfig) -> pd.DataFrame:
    """Apply the exact liquidity, maturity, and option-type filters."""
    filtered = options[
        options["date"].notna()
        & options["exdate"].notna()
        & options["cp_flag"].eq(config.option_type)
        & (options["strike"] > 0)
        & (options["best_bid"] >= 0)
        & (options["best_offer"] > options["best_bid"])
        & (options["mid_price"] >= config.min_mid_price)
        & (options["days_to_expiry"] <= config.target_days_to_expiry)
        & (options["days_to_expiry"] >= config.min_days_to_expiry)
    ].copy()

    if "open_interest" in filtered.columns:
        filtered = filtered[filtered["open_interest"] >= config.min_open_interest].copy()

    return filtered


def select_daily_options(
    options: pd.DataFrame,
    base_df: pd.DataFrame,
    config: OptionSelectionConfig,
) -> pd.DataFrame:
    """Select one option per date using the thesis priority ordering."""
    options = options.merge(base_df[["date", "spx"]], on="date", how="inner")
    options["moneyness"] = options["spx"] / options["strike"]

    if config.target_moneyness < 1.0:
        candidates = options[
            (options["moneyness"] >= config.target_moneyness)
            & (options["moneyness"] < config.moneyness_upper_bound)
        ].copy()
    else:
        candidates = options[
            (options["moneyness"] >= config.atm_lower_bound)
            & (options["moneyness"] <= config.atm_upper_bound)
        ].copy()

    candidates["days_distance"] = config.target_days_to_expiry - candidates["days_to_expiry"]
    candidates["moneyness_distance"] = (candidates["moneyness"] - config.target_moneyness).abs()
    candidates["bid_ask_spread"] = candidates["best_offer"] - candidates["best_bid"]

    sort_cols = ["date", "days_distance", "moneyness_distance", "bid_ask_spread"]
    ascending = [True, True, True, True]

    if "open_interest" in candidates.columns:
        sort_cols.append("open_interest")
        ascending.append(False)
    if "volume" in candidates.columns:
        sort_cols.append("volume")
        ascending.append(False)

    return candidates.sort_values(sort_cols, ascending=ascending).groupby("date", as_index=False).first()


def build_final_dataset(base_df: pd.DataFrame, selected_options: pd.DataFrame) -> pd.DataFrame:
    """Build the final CSV with the thesis columns and rounding conventions."""
    selected_prices = selected_options[["date", "mid_price", "strike", "exdate"]].copy()
    selected_prices = selected_prices.rename(
        columns={"mid_price": "option_price", "exdate": "expiry_date"}
    )

    final_df = base_df.merge(selected_prices, on="date", how="left")
    final_df = final_df[["date", "spx", "r_3m", "strike", "expiry_date", "option_price"]].copy()

    final_df["date"] = pd.to_datetime(final_df["date"]).dt.strftime("%Y-%m-%d")
    final_df["expiry_date"] = pd.to_datetime(final_df["expiry_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    final_df["spx"] = final_df["spx"].round(6)
    final_df["r_3m"] = final_df["r_3m"].round(6)
    final_df["strike"] = final_df["strike"].round(2)
    final_df["option_price"] = final_df["option_price"].round(2)
    return final_df


def create_option_price_dataset(
    base_file: Path,
    options_csv: Path,
    output_file: Path,
    config: OptionSelectionConfig | None = None,
) -> pd.DataFrame:
    """Create and save the option-price dataset used by the pricing notebook."""
    config = config or OptionSelectionConfig()
    base_df = load_base_dataset(base_file)
    options = load_option_chain(options_csv)
    filtered_options = filter_option_chain(options, config)
    selected_options = select_daily_options(filtered_options, base_df, config)
    final_df = build_final_dataset(base_df, selected_options)
    final_df.to_csv(output_file, index=False, sep=";", decimal=",")
    return final_df

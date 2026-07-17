from rsbs.core import black_scholes_price, hmm_parameter_count
from data.option_dataset import OptionSelectionConfig


def test_hmm_parameter_count_two_states():
    assert hmm_parameter_count(2) == 7


def test_option_config_defaults():
    config = OptionSelectionConfig()
    assert config.target_moneyness == 1.0
    assert config.target_days_to_expiry == 30


def test_black_scholes_price_positive():
    price = black_scholes_price(S0=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
    assert price > 0

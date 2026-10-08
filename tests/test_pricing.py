import numpy as np
import pytest

from rsbs.core import (
    black_scholes_implied_vol,
    black_scholes_price,
    diebold_mariano_hac_test,
    find_data_file,
    price_rsbs_pde_nstate,
    solve_rsbs_pde_2state,
    solve_rsbs_pde_nstate,
)


def test_black_scholes_put_call_parity():
    S0, K, T, r, sigma = 100.0, 95.0, 0.5, 0.03, 0.25
    call = black_scholes_price(S0, K, T, r, sigma, "call")
    put = black_scholes_price(S0, K, T, r, sigma, "put")
    assert call - put == pytest.approx(S0 - K * np.exp(-r * T), abs=1e-10)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_implied_vol_round_trip(option_type):
    price = black_scholes_price(100.0, 100.0, 30 / 365, 0.02, 0.18, option_type)
    implied = black_scholes_implied_vol(price, 100.0, 100.0, 30 / 365, 0.02, option_type)
    assert implied == pytest.approx(0.18, abs=1e-8)


def test_pde_with_equal_vols_matches_black_scholes():
    S0, K, T, r, sigma = 100.0, 100.0, 30 / 365, 0.02, 0.2
    Q = np.array([[-3.0, 3.0], [5.0, -5.0]])
    values = solve_rsbs_pde_nstate(S0, K, T, r, [sigma, sigma], Q, n_S=400, n_t=200)
    bs = black_scholes_price(S0, K, T, r, sigma)
    np.testing.assert_allclose(values, bs, rtol=5e-3)


def test_regime_prices_are_ordered_by_volatility():
    Q = np.array([[-2.0, 2.0], [6.0, -6.0]])
    v_low, v_high = solve_rsbs_pde_nstate(100.0, 100.0, 0.25, 0.02, [0.12, 0.35], Q)
    bs_low = black_scholes_price(100.0, 100.0, 0.25, 0.02, 0.12)
    bs_high = black_scholes_price(100.0, 100.0, 0.25, 0.02, 0.35)
    assert bs_low < v_low < v_high < bs_high


def test_two_state_and_nstate_solvers_agree():
    args = dict(S0=100.0, K=105.0, T=0.25, r=0.02)
    v2 = solve_rsbs_pde_2state(**args, sigma_low=0.12, sigma_high=0.35, q_lh=2.0, q_hl=6.0)
    vn = solve_rsbs_pde_nstate(**args, sigmas=[0.12, 0.35], Q=[[-2.0, 2.0], [6.0, -6.0]])
    np.testing.assert_allclose(v2, vn, rtol=1e-10)


def test_rs_price_is_probability_weighted():
    out = price_rsbs_pde_nstate(
        100.0, 100.0, 0.25, 0.02, [0.12, 0.35], [[-2.0, 2.0], [6.0, -6.0]], [0.7, 0.3]
    )
    expected = 0.7 * out["V_state_0"] + 0.3 * out["V_state_1"]
    assert out["rs_price"] == pytest.approx(expected)


def test_diebold_mariano_runs():
    rng = np.random.default_rng(0)
    result = diebold_mariano_hac_test(rng.normal(0, 1, 200), rng.normal(0.5, 1, 200), alternative="less")
    assert result["mean_loss_diff"] < 0
    assert 0 <= result["p_value"] < 0.05


def test_find_data_file_locates_processed_data():
    assert find_data_file("option_price_atm.csv").name == "option_price_atm.csv"

"""Tests for analysis/greeks.py — Black-Scholes Greeks."""
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.greeks import (
    black_scholes_price,
    compute_greeks,
    portfolio_greeks,
    estimate_iv,
)

# --- Reference parameters ---
# ATM call: S=100, K=100, T=1yr, r=5%, sigma=20%
# Textbook BS call price ≈ 10.4506
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20


def test_bs_price_atm_call():
    price = black_scholes_price(S, K, T, r, sigma, 'call')
    assert abs(price - 10.4506) < 0.01, f"ATM call price {price:.4f} not near 10.4506"


def test_bs_price_atm_put():
    price = black_scholes_price(S, K, T, r, sigma, 'put')
    # Textbook ATM put ≈ 5.5735
    assert abs(price - 5.5735) < 0.01, f"ATM put price {price:.4f} not near 5.5735"


def test_put_call_parity():
    call = black_scholes_price(S, K, T, r, sigma, 'call')
    put  = black_scholes_price(S, K, T, r, sigma, 'put')
    lhs = call - put
    rhs = S - K * math.exp(-r * T)
    assert abs(lhs - rhs) < 1e-6, f"Put-call parity violated: C-P={lhs:.6f}, S-Ke^(-rT)={rhs:.6f}"


def test_delta_deep_itm_call():
    # Deep ITM: S >> K
    g = compute_greeks(200.0, 100.0, T, r, sigma, 'call')
    assert g.delta > 0.99, f"Deep ITM call delta should be ~1.0, got {g.delta:.4f}"


def test_delta_deep_otm_call():
    # Deep OTM: S << K
    g = compute_greeks(50.0, 100.0, T, r, sigma, 'call')
    assert g.delta < 0.01, f"Deep OTM call delta should be ~0.0, got {g.delta:.4f}"


def test_delta_call_plus_put_same_strike():
    """For same strike, |delta_call| - |delta_put| ≈ 1 (from put-call parity delta)."""
    g_call = compute_greeks(S, K, T, r, sigma, 'call')
    g_put  = compute_greeks(S, K, T, r, sigma, 'put')
    # delta_call - delta_put = 1 (in absolute terms, call delta + |put delta| ≈ 1)
    total = abs(g_call.delta) + abs(g_put.delta)
    assert abs(total - 1.0) < 0.001, f"delta_call + |delta_put| = {total:.4f}, expected ~1.0"


def test_theta_negative_long_call():
    g = compute_greeks(S, K, T, r, sigma, 'call')
    assert g.theta < 0, f"Long call theta should be negative, got {g.theta:.4f}"


def test_theta_negative_long_put():
    g = compute_greeks(S, K, T, r, sigma, 'put')
    assert g.theta < 0, f"Long put theta should be negative, got {g.theta:.4f}"


def test_vega_positive_call():
    g = compute_greeks(S, K, T, r, sigma, 'call')
    assert g.vega > 0, f"Call vega should be positive, got {g.vega:.4f}"


def test_vega_positive_put():
    g = compute_greeks(S, K, T, r, sigma, 'put')
    assert g.vega > 0, f"Put vega should be positive, got {g.vega:.4f}"


def test_portfolio_greeks_aggregation():
    """Two long calls (qty=2) and one short put (qty=-1)."""
    positions = [
        {'S': 100.0, 'K': 100.0, 'T': 1.0, 'r': 0.05, 'sigma': 0.20,
         'option_type': 'call', 'qty': 2, 'contract_price': 10.45},
        {'S': 100.0, 'K': 100.0, 'T': 1.0, 'r': 0.05, 'sigma': 0.20,
         'option_type': 'put',  'qty': -1, 'contract_price': 5.57},
    ]
    port = portfolio_greeks(positions)
    g_call = compute_greeks(100.0, 100.0, 1.0, 0.05, 0.20, 'call')
    g_put  = compute_greeks(100.0, 100.0, 1.0, 0.05, 0.20, 'put')

    expected_delta = (g_call.delta * 2 * 100) + (g_put.delta * -1 * 100)
    expected_theta = (g_call.theta * 2) + (g_put.theta * -1)
    expected_vega  = (g_call.vega * 2) + (g_put.vega * -1)

    assert abs(port['delta'] - expected_delta) < 1e-6, \
        f"Portfolio delta mismatch: {port['delta']:.4f} vs {expected_delta:.4f}"
    assert abs(port['theta'] - expected_theta) < 1e-6, \
        f"Portfolio theta mismatch: {port['theta']:.4f} vs {expected_theta:.4f}"
    assert abs(port['vega'] - expected_vega) < 1e-6, \
        f"Portfolio vega mismatch: {port['vega']:.4f} vs {expected_vega:.4f}"
    assert port['daily_theta_dollars'] == port['theta'], \
        "daily_theta_dollars should equal theta"


def test_portfolio_greeks_signs():
    """Short position should flip the sign of theta relative to long."""
    long_pos  = [{'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma,
                  'option_type': 'call', 'qty': 1, 'contract_price': 10.45}]
    short_pos = [{'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma,
                  'option_type': 'call', 'qty': -1, 'contract_price': 10.45}]
    long_port  = portfolio_greeks(long_pos)
    short_port = portfolio_greeks(short_pos)
    assert long_port['theta'] < 0,  "Long call theta should be negative"
    assert short_port['theta'] > 0, "Short call theta should be positive"
    assert abs(long_port['theta'] + short_port['theta']) < 1e-6, \
        "Long + short theta should cancel to zero"


def test_estimate_iv_recovers_known_sigma():
    """Round-trip: price an option, then recover sigma from the price."""
    known_sigma = 0.25
    price = black_scholes_price(S, K, T, r, known_sigma, 'call')
    recovered = estimate_iv(price, S, K, T, r, 'call')
    assert abs(recovered - known_sigma) < 0.001, \
        f"IV estimation error too large: {recovered:.6f} vs {known_sigma}"


def test_estimate_iv_put():
    known_sigma = 0.30
    price = black_scholes_price(S, K, T, r, known_sigma, 'put')
    recovered = estimate_iv(price, S, K, T, r, 'put')
    assert abs(recovered - known_sigma) < 0.001, \
        f"Put IV estimation error too large: {recovered:.6f} vs {known_sigma}"


def test_edge_case_zero_time():
    """T=0: price equals intrinsic, no greeks."""
    price = black_scholes_price(S, 90.0, 0.0, r, sigma, 'call')
    assert abs(price - 10.0) < 1e-6, f"ITM call at expiry should be 10.0, got {price}"
    price_otm = black_scholes_price(S, 110.0, 0.0, r, sigma, 'call')
    assert abs(price_otm) < 1e-6, f"OTM call at expiry should be 0.0, got {price_otm}"


def test_intrinsic_and_time_value():
    """For ITM call: intrinsic = S-K, time_value = price - intrinsic."""
    S2, K2 = 110.0, 100.0
    price = black_scholes_price(S2, K2, T, r, sigma, 'call')
    g = compute_greeks(S2, K2, T, r, sigma, 'call', contract_price=price)
    assert abs(g.intrinsic - 10.0) < 1e-6
    assert abs(g.time_value - (price - 10.0)) < 1e-6


def test_put_intrinsic_otm_is_zero():
    """OTM put: intrinsic = 0, time_value = price."""
    price = black_scholes_price(S, 90.0, T, r, sigma, 'put')
    g = compute_greeks(S, 90.0, T, r, sigma, 'put', contract_price=price)
    assert g.intrinsic == 0.0
    assert abs(g.time_value - price) < 1e-6

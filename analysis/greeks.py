"""Options Greeks — Black-Scholes implementation using pure math (no numpy/scipy)."""
import math
from dataclasses import dataclass


@dataclass
class Greeks:
    delta: float      # directional exposure [-1, 1]
    gamma: float      # rate of delta change
    theta: float      # daily time decay in $ per contract (negative for long options)
    vega: float       # $ change per 1% move in IV
    rho: float        # $ change per 1% move in risk-free rate
    iv: float         # implied volatility used (input)
    option_type: str  # 'call' | 'put'
    intrinsic: float  # max(0, S-K) for call, max(0, K-S) for put
    time_value: float # price - intrinsic


def _norm_cdf(x: float) -> float:
    """Rational approximation of the normal CDF (Abramowitz & Stegun 26.2.17).
    Maximum error: ~7.5e-8.
    """
    # Constants
    a1 =  0.319381530
    a2 = -0.356563782
    a3 =  1.781477937
    a4 = -1.821255978
    a5 =  1.330274429
    p  =  0.2316419

    x_abs = abs(x)
    t = 1.0 / (1.0 + p * x_abs)
    poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    cdf_pos = 1.0 - _norm_pdf(x_abs) * poly
    if x >= 0:
        return cdf_pos
    return 1.0 - cdf_pos


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float):
    """Compute d1 and d2 for Black-Scholes."""
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def black_scholes_price(S: float, K: float, T: float, r: float,
                         sigma: float, option_type: str) -> float:
    """
    S: spot price, K: strike, T: time to expiry in years,
    r: risk-free rate (e.g. 0.05), sigma: IV (e.g. 0.25),
    option_type: 'call' | 'put'
    """
    if T <= 0:
        if option_type == 'call':
            return max(0.0, S - K)
        return max(0.0, K - S)
    if sigma <= 0:
        return 0.0

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    discount = math.exp(-r * T)

    if option_type == 'call':
        return S * _norm_cdf(d1) - K * discount * _norm_cdf(d2)
    else:  # put
        return K * discount * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def compute_greeks(S: float, K: float, T: float, r: float,
                   sigma: float, option_type: str,
                   contract_price: float | None = None) -> Greeks:
    """Returns Greeks dataclass. Uses pure math (no scipy).

    Theta is expressed as daily $ decay per single contract (100 shares).
    Vega is $ change per 1% move in IV per contract.
    Rho is $ change per 1% move in risk-free rate per contract.
    """
    # Intrinsic value
    if option_type == 'call':
        intrinsic = max(0.0, S - K)
    else:
        intrinsic = max(0.0, K - S)

    # Edge cases
    if T <= 0 or sigma <= 0:
        price = intrinsic
        time_value = (contract_price - intrinsic) if contract_price is not None else 0.0
        delta = 1.0 if (option_type == 'call' and S > K) else (0.0 if (option_type == 'call' and S <= K) else (-1.0 if S < K else 0.0))
        return Greeks(
            delta=delta, gamma=0.0, theta=0.0, vega=0.0, rho=0.0,
            iv=sigma, option_type=option_type,
            intrinsic=intrinsic, time_value=time_value,
        )

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    sqrt_T = math.sqrt(T)
    discount = math.exp(-r * T)
    nd1 = _norm_pdf(d1)

    # Delta
    if option_type == 'call':
        delta = _norm_cdf(d1)
    else:
        delta = _norm_cdf(d1) - 1.0

    # Gamma (same for call and put)
    gamma = nd1 / (S * sigma * sqrt_T)

    # Theta (annualised, then convert to daily by /365)
    # Per-share theta
    if option_type == 'call':
        theta_annual = (-(S * nd1 * sigma) / (2.0 * sqrt_T)
                        - r * K * discount * _norm_cdf(d2))
    else:
        theta_annual = (-(S * nd1 * sigma) / (2.0 * sqrt_T)
                        + r * K * discount * _norm_cdf(-d2))
    theta_daily_per_share = theta_annual / 365.0
    # Per contract (100 shares)
    theta = theta_daily_per_share * 100.0

    # Vega: per-share $ change per 1 unit (100%) move in sigma
    # Scale to per 1% move, then per contract
    vega_per_share = S * nd1 * sqrt_T          # $ per 100% IV change
    vega = (vega_per_share / 100.0) * 100.0    # $ per 1% IV change, per contract

    # Rho: per-share $ change per 1 unit (100%) move in r
    if option_type == 'call':
        rho_per_share = K * T * discount * _norm_cdf(d2)
    else:
        rho_per_share = -K * T * discount * _norm_cdf(-d2)
    rho = (rho_per_share / 100.0) * 100.0      # $ per 1% rate change, per contract

    # Price and time value
    price = black_scholes_price(S, K, T, r, sigma, option_type)
    ref_price = contract_price if contract_price is not None else price
    time_value = ref_price - intrinsic

    return Greeks(
        delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho,
        iv=sigma, option_type=option_type,
        intrinsic=intrinsic, time_value=time_value,
    )


def portfolio_greeks(positions: list[dict]) -> dict:
    """
    positions: list of {'S': float, 'K': float, 'T': float, 'r': float,
                         'sigma': float, 'option_type': str, 'qty': int,
                         'contract_price': float}
    Returns aggregate: {'delta': float, 'gamma': float, 'theta': float,
                        'vega': float, 'daily_theta_dollars': float}
    qty is signed (positive = long, negative = short)
    Each option contract = 100 shares
    """
    total_delta = 0.0
    total_gamma = 0.0
    total_theta = 0.0
    total_vega = 0.0

    for pos in positions:
        qty = pos['qty']
        g = compute_greeks(
            S=pos['S'], K=pos['K'], T=pos['T'], r=pos['r'],
            sigma=pos['sigma'], option_type=pos['option_type'],
            contract_price=pos.get('contract_price'),
        )
        # Delta exposure in shares: delta * qty * 100
        total_delta += g.delta * qty * 100.0
        # Gamma per $1 move: gamma * qty * 100
        total_gamma += g.gamma * qty * 100.0
        # Theta already $ per day per contract; scale by qty
        total_theta += g.theta * qty
        # Vega already $ per 1% IV; scale by qty
        total_vega += g.vega * qty

    return {
        'delta': total_delta,
        'gamma': total_gamma,
        'theta': total_theta,
        'vega': total_vega,
        'daily_theta_dollars': total_theta,
    }


def estimate_iv(market_price: float, S: float, K: float, T: float,
                r: float, option_type: str,
                max_iter: int = 100, tol: float = 1e-6) -> float:
    """Newton-Raphson IV solver. Returns sigma estimate."""
    if T <= 0:
        return 0.0

    # Initial guess: simple approximation (Brenner-Subrahmanyam)
    sigma = math.sqrt(2.0 * math.pi / T) * market_price / S
    sigma = max(0.001, min(sigma, 10.0))  # clamp to sane range

    for _ in range(max_iter):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma

        # Vega per-share (derivative of price w.r.t. sigma)
        d1, _ = _d1_d2(S, K, T, r, sigma)
        vega = S * _norm_pdf(d1) * math.sqrt(T)
        if vega < 1e-10:
            break
        sigma -= diff / vega
        sigma = max(1e-6, sigma)  # keep positive

    return sigma

from analysis.risk_metrics import (
    RiskMetrics,
    compute_risk_metrics,
    historical_cvar,
    historical_var,
    monte_carlo_var,
)

RETURNS = [
    -0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.02, 0.01,
    -0.02, -0.04, 0.03, 0.01, -0.01, 0.02, 0.05, -0.03, 0.01, 0.02,
]


def test_historical_var_positive():
    assert historical_var(RETURNS, 0.95) > 0


def test_historical_var_95_lt_99():
    assert historical_var(RETURNS, 0.95) <= historical_var(RETURNS, 0.99)


def test_cvar_gte_var():
    assert historical_cvar(RETURNS, 0.95) >= historical_var(RETURNS, 0.95)


def test_monte_carlo_var_deterministic():
    assert monte_carlo_var(RETURNS, seed=42) == monte_carlo_var(RETURNS, seed=42)


def test_compute_risk_metrics_returns_dataclass():
    vals = [100.0]
    for r in RETURNS:
        vals.append(vals[-1] * (1 + r))
    metrics = compute_risk_metrics(vals)
    assert isinstance(metrics, RiskMetrics)
    assert metrics.n_observations == len(RETURNS)
    assert metrics.var_95 > 0
    assert metrics.cvar_95 >= metrics.var_95


def test_compute_risk_metrics_insufficient_data():
    assert compute_risk_metrics([100, 101]) is None


def test_empty_returns():
    assert historical_var([]) == 0.0
    assert historical_cvar([]) == 0.0
    assert monte_carlo_var([]) == 0.0

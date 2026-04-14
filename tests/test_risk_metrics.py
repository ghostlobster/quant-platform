from analysis.risk_metrics import (
    RiskMetrics,
    compute_risk_metrics,
    historical_cvar,
    historical_var,
    monte_carlo_var,
)

RETURNS = [-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.02, 0.01,
           -0.02, -0.04, 0.03, 0.01, -0.01, 0.02, 0.05, -0.03, 0.01, 0.02]


def test_historical_var_positive():
    var = historical_var(RETURNS, 0.95)
    assert var > 0

def test_historical_var_95_lt_99():
    var95 = historical_var(RETURNS, 0.95)
    var99 = historical_var(RETURNS, 0.99)
    assert var95 <= var99

def test_cvar_gte_var():
    var = historical_var(RETURNS, 0.95)
    cvar = historical_cvar(RETURNS, 0.95)
    assert cvar >= var

def test_monte_carlo_var_deterministic():
    v1 = monte_carlo_var(RETURNS, seed=42)
    v2 = monte_carlo_var(RETURNS, seed=42)
    assert v1 == v2

def test_compute_risk_metrics_returns_dataclass():
    # build cumulative prices
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

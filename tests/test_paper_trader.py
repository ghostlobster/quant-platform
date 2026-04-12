import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import data.db as db_module
import broker.paper_trader as pt


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Redirect all DB operations to a per-test temporary file."""
    test_db = tmp_path / "test_paper.db"
    monkeypatch.setattr(db_module, "_DB_PATH", test_db)
    monkeypatch.setattr(pt, "STARTING_CASH", 100_000.0)
    pt.init_paper_tables()
    yield test_db


def test_initial_balance():
    account = pt.get_account()
    assert account["cash"] > 0
    assert account["cash"] == pytest.approx(100_000.0)


def test_buy_creates_position():
    pt.buy("AAPL", shares=10, price=150.0)
    portfolio = pt.get_portfolio(current_prices={"AAPL": 150.0})
    assert "AAPL" in portfolio["Ticker"].values


def test_buy_reduces_cash():
    before = pt.get_account()["cash"]
    pt.buy("AAPL", shares=10, price=150.0)
    after = pt.get_account()["cash"]
    assert after == pytest.approx(before - 10 * 150.0)


def test_buy_raises_insufficient_cash():
    with pytest.raises(ValueError, match="Insufficient cash"):
        pt.buy("AAPL", shares=100_000, price=200.0)


def test_sell_reduces_position():
    pt.buy("AAPL", shares=10, price=150.0)
    pt.sell("AAPL", shares=5, price=160.0)
    portfolio = pt.get_portfolio()
    row = portfolio[portfolio["Ticker"] == "AAPL"]
    assert float(row["Shares"].iloc[0]) == pytest.approx(5.0)


def test_sell_raises_no_position():
    with pytest.raises(ValueError, match="No open position"):
        pt.sell("AAPL", shares=5, price=150.0)


def test_get_portfolio_returns_dataframe():
    portfolio = pt.get_portfolio()
    assert hasattr(portfolio, "columns")
    assert "Ticker" in portfolio.columns
    assert "Shares" in portfolio.columns


def test_reset_restores_cash():
    original_cash = pt.get_account()["cash"]
    pt.buy("TSLA", shares=5, price=200.0)
    pt.reset_account()
    restored_cash = pt.get_account()["cash"]
    assert abs(restored_cash - original_cash) < 1.0


def test_trade_history_recorded():
    pt.buy("MSFT", shares=3, price=300.0)
    history = pt.get_trade_history()
    assert len(history) >= 1


def test_circuit_breaker_triggers_on_drawdown():
    """Simulate >20% drawdown by directly reducing cash_balance, then assert buy() raises."""
    conn = db_module.get_connection()
    # Set cash to 70 % of starting capital → 30 % drawdown, exceeds 20 % threshold
    conn.execute("UPDATE paper_account SET cash_balance = 70000 WHERE id = 1")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="Circuit breaker"):
        pt.buy("AAPL", shares=1, price=10.0)

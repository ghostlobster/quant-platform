import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from risk.kelly import kelly_fraction, kelly_from_backtest


def test_kelly_positive_edge():
    f = kelly_fraction(win_rate=0.55, avg_win=0.05, avg_loss=0.03)
    assert 0 < f <= 0.25

def test_kelly_no_edge_returns_zero():
    f = kelly_fraction(win_rate=0.40, avg_win=0.02, avg_loss=0.05)
    assert f == 0.0

def test_kelly_capped_at_max():
    f = kelly_fraction(win_rate=0.90, avg_win=0.20, avg_loss=0.01, max_fraction=0.25)
    assert f <= 0.25

def test_kelly_zero_loss_returns_zero():
    assert kelly_fraction(0.6, 0.05, 0.0) == 0.0

def test_kelly_from_backtest():
    f = kelly_from_backtest(total_return=0.15, trade_count=30, win_rate=0.55)
    assert 0.0 <= f <= 0.25

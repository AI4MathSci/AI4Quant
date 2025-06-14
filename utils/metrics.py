# utils/metrics.py

import numpy as np
import pandas as pd

def calculate_cumulative_returns(prices: pd.Series) -> pd.Series:
    """Calculate cumulative returns from a price series."""
    returns = prices.pct_change().fillna(0)
    cumulative = (1 + returns).cumprod()
    return cumulative

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sharpe ratio of returns."""
    excess_returns = returns - risk_free_rate / 252
    annualized_return = excess_returns.mean() * 252
    annualized_volatility = excess_returns.std() * np.sqrt(252)
    if annualized_volatility == 0:
        return np.nan
    return annualized_return / annualized_volatility

def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """Calculate the maximum drawdown from a cumulative return series."""
    rolling_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / rolling_max - 1
    return drawdown.min()

def calculate_win_loss_ratio(signals: pd.Series, returns: pd.Series) -> float:
    """Calculate win/loss ratio based on signal returns."""
    signal_returns = signals.shift(1) * returns
    wins = signal_returns[signal_returns > 0].count()
    losses = signal_returns[signal_returns < 0].count()
    if losses == 0:
        return float('inf')
    return wins / losses

import numpy as np
import pandas as pd

def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate=0.0) -> float:
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def calculate_win_loss_ratio(signals: pd.Series, returns: pd.Series) -> float:
    applied_returns = signals.shift(1).fillna(0) * returns
    wins = applied_returns[applied_returns > 0].count()
    losses = applied_returns[applied_returns < 0].count()
    return wins / losses if losses > 0 else float('inf')


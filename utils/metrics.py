# utils/metrics.py

import numpy as np
import pandas as pd

def calculate_cumulative_returns(prices: pd.Series) -> pd.Series:
    return (1 + prices.pct_change().fillna(0)).cumprod()

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess_returns = returns - risk_free_rate
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() else 0.0

def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def calculate_win_loss_ratio(signals: pd.Series, returns: pd.Series) -> float:
    correct = ((signals.shift(1).fillna(0) > 0) & (returns > 0)) | ((signals.shift(1) < 0) & (returns < 0))
    wins = correct.sum()
    losses = (~correct).sum()
    return wins / losses if losses > 0 else float('inf')


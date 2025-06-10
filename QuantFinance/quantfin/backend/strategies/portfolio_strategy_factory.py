from typing import Dict, Any, List
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns

def get_portfolio_strategy(strategy_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    strategies = {
        "Equal Weight": lambda prices, symbols: {s: 1/len(symbols) for s in symbols},
        "Mean Variance Optimization": lambda prices, symbols: {
            "mu": expected_returns.mean_historical_return(prices),
            "S": risk_models.sample_cov(prices),
        },
        "Momentum": lambda prices, symbols: {
            "returns": prices.pct_change().mean(),
            "top": prices.pct_change().mean().sort_values(ascending=False).index.tolist()[:max(1, len(symbols)//3)],
        },
    }
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    return strategies[strategy_name](params.get("prices"), params.get("asset_symbols")) 
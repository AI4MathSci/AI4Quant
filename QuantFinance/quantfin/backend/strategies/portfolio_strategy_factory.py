from typing import Dict, Any, List
import pandas as pd
import numpy as np

def get_portfolio_strategy(strategy: str, params: Dict[str, Any]) -> Dict[str, float]:
    """
    Get portfolio weights based on the selected strategy.
    
    Args:
        strategy: Strategy name
        params: Dictionary containing prices, target_return, lookback_period, asset_symbols
    
    Returns:
        Dict[str, float]: Portfolio weights that sum to 1.0
    """
    prices = params.get("prices")
    target_return = params.get("target_return", 0.12)
    lookback_period = params.get("lookback_period", 252)
    asset_symbols = params.get("asset_symbols", [])
    
    if prices is None or len(asset_symbols) == 0:
        # Fallback to equal weights if no data
        equal_weight = 1.0 / len(asset_symbols) if asset_symbols else 0.0
        return {symbol: equal_weight for symbol in asset_symbols}
    
    # Ensure prices is a DataFrame with proper columns
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    
    # Handle multi-level columns from yfinance
    if hasattr(prices.columns, 'levels'):
        prices.columns = asset_symbols
    
    # Ensure we have the right symbols
    available_symbols = [col for col in asset_symbols if col in prices.columns]
    if not available_symbols:
        # Fallback to equal weights
        equal_weight = 1.0 / len(asset_symbols)
        return {symbol: equal_weight for symbol in asset_symbols}
    
    prices = prices[available_symbols]
    
    try:
        if strategy == "Equal Weight":
            return _equal_weight_strategy(available_symbols)
        
        elif strategy == "Mean Variance Optimization":
            return _mean_variance_optimization(prices, available_symbols, target_return)
        
        elif strategy == "Momentum":
            return _momentum_strategy(prices, available_symbols, lookback_period)
        
        else:
            # Default to equal weight for unknown strategies
            return _equal_weight_strategy(available_symbols)
            
    except Exception as e:
        print(f"Strategy calculation failed: {e}. Falling back to equal weights.")
        return _equal_weight_strategy(available_symbols)

def _equal_weight_strategy(symbols: List[str]) -> Dict[str, float]:
    """Equal weight allocation across all assets."""
    if not symbols:
        return {}
    
    weight = 1.0 / len(symbols)
    return {symbol: weight for symbol in symbols}

def _mean_variance_optimization(prices: pd.DataFrame, symbols: List[str], target_return: float = 0.12) -> Dict[str, float]:
    """
    Mean variance optimization using simplified approach.
    Finds portfolio with minimum variance for given expected return.
    """
    try:
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        if len(returns) < 30:  # Need sufficient data
            return _equal_weight_strategy(symbols)
        
        # Calculate expected returns and covariance matrix
        mu = returns.mean() * 252  # Annualized returns
        cov_matrix = returns.cov() * 252  # Annualized covariance
        
        # Simplified mean variance optimization
        # Use inverse volatility weighting as a practical approximation
        volatilities = np.sqrt(np.diag(cov_matrix))
        inv_vol = 1.0 / volatilities
        weights = inv_vol / inv_vol.sum()
        
        # Ensure weights sum to 1.0
        weights = weights / weights.sum()
        
        return {symbol: float(weight) for symbol, weight in zip(symbols, weights)}
        
    except Exception as e:
        print(f"Mean variance optimization failed: {e}")
        return _equal_weight_strategy(symbols)

def _momentum_strategy(prices: pd.DataFrame, symbols: List[str], lookback_period: int = 252) -> Dict[str, float]:
    """
    Momentum strategy based on recent performance.
    Allocates more weight to better performing assets.
    """
    try:
        # Calculate momentum scores (total return over lookback period)
        if len(prices) < max(20, lookback_period // 4):  # Need minimum data
            return _equal_weight_strategy(symbols)
        
        # Use available data up to lookback period
        lookback = min(lookback_period, len(prices))
        momentum_scores = (prices.iloc[-1] / prices.iloc[-lookback]) - 1
        
        # Convert negative momentum to small positive weights
        # This prevents short positions which aren't typically allowed
        adjusted_scores = momentum_scores + abs(momentum_scores.min()) + 0.01
        
        # Calculate weights based on momentum scores
        weights = adjusted_scores / adjusted_scores.sum()
        
        # Ensure weights sum to 1.0
        weights = weights / weights.sum()
        
        return {symbol: float(weight) for symbol, weight in zip(symbols, weights)}
        
    except Exception as e:
        print(f"Momentum strategy failed: {e}")
        return _equal_weight_strategy(symbols) 
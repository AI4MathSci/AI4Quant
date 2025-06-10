from pydantic import BaseModel
from typing import Dict, Any, List
import yfinance as yf
from datetime import datetime
import pandas as pd

from quantfin.backend.strategies.portfolio_strategy_factory import get_portfolio_strategy

class PortfolioModel(BaseModel):
    """
    Represents a portfolio management model.
    """
    asset_symbols: List[str] = ["AAPL", "GOOGL", "MSFT", "AMZN"]
    strategy: str = "Equal Weight"  # Options: "Equal Weight", "Mean Variance Optimization", "Momentum"
    initial_capital: float = 100000.0
    target_return: float = 0.12  # Only used for Mean Variance Optimization
    lookback_period: int = 252  # Only used for Momentum strategy
    start_date: str
    end_date: str

    async def simulate(self) -> Dict[str, Any]:
        """
        Simulates the portfolio management model.
        """
        prices = yf.download(self.asset_symbols, start=self.start_date, end=self.end_date)["Close"]
        weights = get_portfolio_strategy(self.strategy, {
            "prices": prices,
            "target_return": self.target_return,
            "lookback_period": self.lookback_period,
            "asset_symbols": self.asset_symbols,
        })
        
        # Calculate allocation amounts in dollars
        allocation_amounts = {symbol: round(weight * self.initial_capital, 2) 
                            for symbol, weight in weights.items()}
        
        # Calculate additional optimization metrics (added from optimize method)
        returns = prices.pct_change().dropna()
        portfolio_returns = (returns * list(weights.values())).sum(axis=1)
        expected_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * (252 ** 0.5)
        sharpe_ratio = expected_return / volatility if volatility > 0 else 0
        
        # Calculate projected final values
        time_period_years = (pd.to_datetime(self.end_date) - pd.to_datetime(self.start_date)).days / 365.25
        projected_total_return = expected_return * time_period_years
        projected_final_value = round(self.initial_capital * (1 + projected_total_return), 2)
        projected_return_dollars = round(projected_final_value - self.initial_capital, 2)
        projected_return_percent = round(projected_total_return * 100, 2)
        
        # Calculate individual asset expected returns and final values
        individual_returns = returns.mean() * 252  # Annualized returns per asset
        individual_projected_returns = individual_returns * time_period_years
        final_asset_values = {symbol: round(allocation_amounts[symbol] * (1 + individual_projected_returns[symbol]), 2)
                            for symbol in self.asset_symbols}
        
        return {
            "strategy": self.strategy,
            "initial_capital": self.initial_capital,
            "weights": {symbol: round(weight, 4) for symbol, weight in weights.items()},
            "allocation_amounts": allocation_amounts,
            "projected_final_value": projected_final_value,
            "projected_return_dollars": projected_return_dollars,
            "projected_return_percent": projected_return_percent,
            "final_asset_values": final_asset_values,
            "expected_annual_return": round(expected_return, 4),
            "volatility": round(volatility, 4),
            "sharpe_ratio": round(sharpe_ratio, 4),
            "message": f"Portfolio simulation successful using {self.strategy}."
        }

    async def backtest(self) -> Dict[str, Any]:
        """
        Backtests the portfolio management model with historical price data over a chosen period of time.

        Returns:
            Dict[str, Any]: The results of the backtest including performance metrics.
        """
        prices = yf.download(self.asset_symbols, start=self.start_date, end=self.end_date)["Close"]
        weights = get_portfolio_strategy(self.strategy, {
            "prices": prices,
            "target_return": self.target_return,
            "lookback_period": self.lookback_period,
            "asset_symbols": self.asset_symbols,
        })
        
        # Calculate allocation amounts in dollars
        allocation_amounts = {symbol: round(weight * self.initial_capital, 2) 
                            for symbol, weight in weights.items()}
        
        # Calculate portfolio returns for backtesting
        returns = prices.pct_change().dropna()
        portfolio_returns = (returns * list(weights.values())).sum(axis=1)
        total_return = (1 + portfolio_returns).prod() - 1
        annual_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * (252 ** 0.5)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Calculate portfolio values
        final_portfolio_value = round(self.initial_capital * (1 + total_return), 2)
        total_return_dollars = round(final_portfolio_value - self.initial_capital, 2)
        total_return_percent = round(total_return * 100, 2)
        
        # Calculate individual asset actual returns and final values
        individual_total_returns = (prices.iloc[-1] / prices.iloc[0]) - 1
        final_asset_values = {symbol: round(allocation_amounts[symbol] * (1 + individual_total_returns[symbol]), 2)
                            for symbol in self.asset_symbols}
        
        return {
            "strategy": self.strategy,
            "initial_capital": self.initial_capital,
            "weights": {symbol: round(weight, 4) for symbol, weight in weights.items()},
            "allocation_amounts": allocation_amounts,
            "final_portfolio_value": final_portfolio_value,
            "total_return_dollars": total_return_dollars,
            "total_return_percent": total_return_percent,
            "final_asset_values": final_asset_values,
            "total_return": round(total_return, 4),
            "annual_return": round(annual_return, 4),
            "volatility": round(volatility, 4),
            "sharpe_ratio": round(sharpe_ratio, 4),
            "message": f"Portfolio backtesting successful using {self.strategy}."
        }
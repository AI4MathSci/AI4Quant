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

    async def optimize(self) -> Dict[str, Any]:
        """
        Optimizes portfolio parameters by testing different strategies and configurations.
        Finds the best strategy based on risk-adjusted returns (Sharpe ratio).

        Returns:
            Dict[str, Any]: Optimization results with best strategy and performance comparison.
        """
        prices = yf.download(self.asset_symbols, start=self.start_date, end=self.end_date, progress=False)["Close"]
        
        # Define strategies and their parameter ranges to test
        strategies_to_test = [
            {"strategy": "Equal Weight", "params": {}},
            {"strategy": "Mean Variance Optimization", "params": {"target_return": 0.08}},
            {"strategy": "Mean Variance Optimization", "params": {"target_return": 0.12}},
            {"strategy": "Mean Variance Optimization", "params": {"target_return": 0.16}},
            {"strategy": "Momentum", "params": {"lookback_period": 126}},  # 6 months
            {"strategy": "Momentum", "params": {"lookback_period": 252}},  # 1 year
            {"strategy": "Momentum", "params": {"lookback_period": 504}},  # 2 years
        ]
        
        optimization_results = []
        
        for strategy_config in strategies_to_test:
            try:
                strategy_name = strategy_config["strategy"]
                strategy_params = strategy_config["params"]
                
                # Get strategy-specific parameters
                target_return = strategy_params.get("target_return", self.target_return)
                lookback_period = strategy_params.get("lookback_period", self.lookback_period)
                
                # Calculate weights for this strategy configuration
                weights = get_portfolio_strategy(strategy_name, {
                    "prices": prices,
                    "target_return": target_return,
                    "lookback_period": lookback_period,
                    "asset_symbols": self.asset_symbols,
                })
                
                # Calculate performance metrics
                returns = prices.pct_change().dropna()
                portfolio_returns = (returns * list(weights.values())).sum(axis=1)
                
                # Performance calculations
                total_return = (1 + portfolio_returns).prod() - 1
                annual_return = portfolio_returns.mean() * 252
                volatility = portfolio_returns.std() * (252 ** 0.5)
                sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                
                # Portfolio values
                final_portfolio_value = self.initial_capital * (1 + total_return)
                total_return_dollars = final_portfolio_value - self.initial_capital
                
                # Store results
                result = {
                    "strategy": strategy_name,
                    "parameters": strategy_params,
                    "weights": {symbol: round(weight, 4) for symbol, weight in weights.items()},
                    "final_portfolio_value": round(final_portfolio_value, 2),
                    "total_return_dollars": round(total_return_dollars, 2),
                    "total_return": round(total_return, 4),
                    "annual_return": round(annual_return, 4),
                    "volatility": round(volatility, 4),
                    "sharpe_ratio": round(sharpe_ratio, 4),
                }
                optimization_results.append(result)
                
            except Exception as e:
                print(f"Error testing {strategy_config}: {e}")
                continue
        
        # Find best strategy based on Sharpe ratio
        if optimization_results:
            best_strategy = max(optimization_results, key=lambda x: x["sharpe_ratio"])
            
            # Calculate improvement vs equal weight baseline
            equal_weight_result = next((r for r in optimization_results if r["strategy"] == "Equal Weight"), None)
            improvement = 0
            if equal_weight_result:
                improvement = best_strategy["sharpe_ratio"] - equal_weight_result["sharpe_ratio"]
            
            return {
                "methodology": "Portfolio Optimization",
                "optimization_complete": True,
                "strategies_tested": len(optimization_results),
                "best_strategy": best_strategy,
                "improvement_over_equal_weight": round(improvement, 4),
                "all_results": sorted(optimization_results, key=lambda x: x["sharpe_ratio"], reverse=True),
                "recommendation": {
                    "strategy": best_strategy["strategy"],
                    "parameters": best_strategy["parameters"],
                    "expected_return": best_strategy["annual_return"],
                    "risk": best_strategy["volatility"],
                    "sharpe_ratio": best_strategy["sharpe_ratio"],
                },
                "message": f"Optimization complete. Best strategy: {best_strategy['strategy']} with Sharpe ratio of {best_strategy['sharpe_ratio']:.4f}"
            }
        else:
            # Fallback if optimization fails
            return {
                "methodology": "Portfolio Optimization",
                "optimization_complete": False,
                "error": "Unable to optimize portfolio - insufficient data or calculation errors",
                "fallback_strategy": "Equal Weight",
                "message": "Optimization failed. Using equal weight allocation as fallback."
            }
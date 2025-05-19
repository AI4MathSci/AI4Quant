from pydantic import BaseModel
from typing import Dict, Any, List
import numpy as np
from quantfin.backend.utils.data_loader import load_historical_data  # Import the data loader

class PortfolioModel(BaseModel):
    """
    Represents a portfolio management model.
    """
    initial_capital: float = 1000000.0
    allocation_strategy: str = "Equal Weight"
    risk_tolerance: str = "Medium"
    asset_classes: List[str] = ["Stocks", "Bonds", "Cash"]
    # Add more portfolio-specific parameters if needed

    async def simulate(self) -> Dict[str, Any]:
        """
        Simulates the portfolio management model by projecting the final portfolio value
        over a fixed holding period (e.g., 5 years) using assumed annual return rates
        that vary by asset class and risk tolerance.
        """
        print(f"Simulating portfolio with strategy: {self.allocation_strategy}, "
              f"risk_tolerance: {self.risk_tolerance}, and initial capital: {self.initial_capital}")

        # Determine allocation weights.
        # Currently, if allocation_strategy is "Equal Weight" or unrecognized, use equal allocation.
        num_assets = len(self.asset_classes)
        weights = {asset: 1/num_assets for asset in self.asset_classes}

        # Define assumed annual returns for each asset based on risk tolerance.
        # These numbers are for simulation purposes and can be adjusted.
        if self.risk_tolerance == "High":
            expected_returns = {"Stocks": 0.10, "Bonds": 0.04, "Cash": 0.02}
        elif self.risk_tolerance == "Low":
            expected_returns = {"Stocks": 0.05, "Bonds": 0.02, "Cash": 0.005}
        else:  # "Medium" risk tolerance or any other value defaults here
            expected_returns = {"Stocks": 0.07, "Bonds": 0.03, "Cash": 0.01}

        # For any asset classes not explicitly listed in expected_returns, assign a default return.
        for asset in self.asset_classes:
            if asset not in expected_returns:
                expected_returns[asset] = 0.03  # fallback value

        # Define the simulation horizon (e.g., 5 years)
        years = 5
        asset_final_values = {}

        print(f"self.asset_classes = {self.asset_classes}")
        for asset in self.asset_classes:
            weight = weights[asset]
            allocated_capital = self.initial_capital * weight
            # Compound growth: final value = initial allocation * (1 + annual_return)^years
            annual_ret = expected_returns[asset]
            final_value = allocated_capital * ((1 + annual_ret) ** years)
            asset_final_values[asset] = round(final_value, 2)
 
        final_portfolio_value = sum(asset_final_values.values())
 
        return {
            "strategy": self.allocation_strategy,
            "risk_tolerance": self.risk_tolerance,
            "initial_capital": self.initial_capital,
            "final_portfolio_value": round(final_portfolio_value, 2),
            "asset_allocation": weights,
            "asset_final_values": asset_final_values,
            "message": "Portfolio simulation successful"
        }

    async def backtest(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Backtests the portfolio management model with historical price data over a chosen period of time

        Args:
            symbol (str): The stock symbol.
            start_date (str): The start date for historical data.
            end_date (str): The end date for historical data.

        Returns:
            Dict[str, Any]: The results of the backtest.
        """

        print(f"Backtesting portfolio model with strategy: {self.allocation_strategy} for symbol: {symbol} from {start_date} to {end_date}")

        # Load historical data
        historical_data = load_historical_data(symbol, start_date, end_date)

        # refer to the comment for same line of code in the backtest function of the trading model
        prices = historical_data[symbol][('Close', symbol)] 

        if prices is None or prices.empty:
            return {
                "strategy": self.allocation_strategy,
                "backtest_results": None,
                "message": "No 'prices' data provided for backtesting."
            }

        # Calculate total return for each asset: (final_price / initial_price) - 1
        asset_returns = {}
        for asset in self.asset_classes:
            price_series = prices.get(asset, None)
            if price_series is None or len(price_series) < 2:
                continue
            start_price = price_series[0]
            end_price = price_series[-1]
            total_return = (end_price / start_price) - 1
            asset_returns[asset] = total_return

        if not asset_returns:
            return {
                "message": "No valid price data for the specified asset classes in backtest."
            }

        # Again, use the allocation strategy (here, "Equal Weight" by default) to allocate capital.
        num_assets = len(self.asset_classes)
        allocation = 1.0 / num_assets  # Equal weight allocation

        # Calculate portfolio return
        portfolio_return = sum(return_value * allocation for return_value in asset_returns.values())
        final_value = self.initial_capital * (1 + portfolio_return)

        backtest_results = {
            "final_portfolio_value": round(final_value, 2),
            "portfolio_return": round(portfolio_return, 4),
            "asset_returns": {asset: round(return_value, 4) for asset, return_value in asset_returns.items()},
            "allocation": allocation
        }

        return {
            "strategy": self.allocation_strategy,
            "backtest_results": backtest_results,
            "historical_data_keys": list(historical_data.keys()),
            "message": "Portfolio model backtest successful"
        }
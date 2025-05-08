from pydantic import BaseModel
from typing import Dict, Any, List
import numpy as np

class PortfolioModel(BaseModel):
    """
    Represents a portfolio management model.
    """
    initial_capital: float = 1000000.0
    allocation_strategy: str = "Equal Weight"
    risk_tolerance: str = "Medium"
    asset_classes: List[str] = ["Stocks", "Bonds", "Cash"]
    # Add more portfolio-specific parameters if needed

    def simulate(self) -> Dict[str, Any]:
        """
        Simulates the portfolio management model by projecting the final portfolio value
        over a fixed holding period (e.g., 5 years) using assumed annual return rates
        that vary by asset class and risk tolerance.
        """
        print(f"Simulating portfolio with strategy: {self.allocation_strategy}, "
              f"risk tolerance: {self.risk_tolerance}, and initial capital: {self.initial_capital}")

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

            print(f"----final_value = {final_value}")

            asset_final_values[asset] = round(final_value, 2)
 
        final_portfolio_value = sum(asset_final_values.values())
 
        print(f"----final_portfolio_value = {final_portfolio_value}")

        print(f"----asset_final_values = {asset_final_values}")

        return {
            "strategy": self.allocation_strategy,
            "risk_tolerance": self.risk_tolerance,
            "initial_capital": self.initial_capital,
            "final_portfolio_value": round(final_portfolio_value, 2),
            "asset_allocation": weights,
            "asset_final_values": asset_final_values,
            "message": "Portfolio simulation successful"
        }

    def backtest(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Backtests the portfolio management model using provided historical price data.
        The historical_data dictionary is expected to include a key 'prices' which maps
        asset class names to lists of historical prices over a given period.
        """
        print(f"Backtesting portfolio model with strategy: {self.allocation_strategy} "
              f"and data: {historical_data.keys()}")

        prices_data = historical_data.get("prices", {})
        if not prices_data:
            return {
                "message": "No 'prices' data provided for portfolio backtest."
            }

        # Calculate total return for each asset: (final_price / initial_price) - 1
        asset_returns = {}
        for asset in self.asset_classes:
            price_series = prices_data.get(asset, None)
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
        weights = {asset: 1/num_assets for asset in self.asset_classes}

        # Compute the final portfolio value using the historical return for each asset.
        asset_final_values = {}
        for asset in self.asset_classes:
            weight = weights.get(asset, 0)
            allocated_capital = self.initial_capital * weight
            if asset in asset_returns:
                # Apply the asset’s total return over the backtest period.
                final_value = allocated_capital * (1 + asset_returns[asset])
            else:
                final_value = allocated_capital  # assume no change if no data available
            asset_final_values[asset] = round(final_value, 2)

        final_portfolio_value = sum(asset_final_values.values())
        return {
            "strategy": self.allocation_strategy,
            "initial_capital": self.initial_capital,
            "final_portfolio_value": round(final_portfolio_value, 2),
            "asset_allocation": weights,
            "asset_final_values": asset_final_values,
            "historical_data_keys": list(historical_data.keys()),
            "message": "Portfolio backtest successful"
        }
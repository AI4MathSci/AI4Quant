from pydantic import BaseModel
from typing import Dict, Any
import numpy as np
import pandas as pd

class TradingModel(BaseModel):
    """
    Represents a trading model with its parameters.
    """
    initial_capital: float = 100000.0
    strategy: str = "Simple Moving Average"
    sma_window: int = 20
    # Add more trading-specific parameters as needed

    def simulate(self) -> Dict[str, Any]:
        """
        Simulates the trading model.
        """
        if self.strategy == "Simple Moving Average":
            print(f"Running Simple Moving Average strategy with initial capital: {self.initial_capital} and window: {self.sma_window}")
            # Generate synthetic price data (geometric random walk)
            np.random.seed(42)
            days = 100
            dt = 1/252
            mu = 0.1
            sigma = 0.2
            price0 = 100.0
            prices = [price0]
            for _ in range(1, days):
                drift = (mu - 0.5 * sigma ** 2) * dt
                shock = sigma * np.sqrt(dt) * np.random.normal()
                prices.append(prices[-1] * np.exp(drift + shock))
            prices_series = pd.Series(prices)
            
            # Compute simple moving average
            sma = prices_series.rolling(window=self.sma_window, min_periods=1).mean()
            # Generate signals: +1 when price is above SMA, -1 when below
            signals = (prices_series > sma).astype(int)
            signals[signals == 0] = -1
            
            # Count transactions: each signal change is a transaction
            transactions = int(signals.diff().abs().sum())
            
            # Simulate portfolio evolution: assume fully invested when signal==1, otherwise in cash.
            # For simplicity, assume the portfolio value follows the price evolution if invested.
            position_value = self.initial_capital * (prices_series.iloc[-1] / prices_series.iloc[0]) if signals.iloc[-1] == 1 else self.initial_capital
            result = {
                "strategy": self.strategy,
                "initial_capital": self.initial_capital,
                "final_portfolio_value": round(position_value, 2),
                "transactions": transactions,
                "message": "Trading simulation successful (Simple Moving Average)"
            }
            return result

        elif self.strategy == "Buy and Hold":
            print(f"Running Buy and Hold strategy with initial capital: {self.initial_capital}")
            # Generate synthetic price data (geometric random walk)
            np.random.seed(42)
            days = 100
            dt = 1/252
            mu = 0.1
            sigma = 0.2
            price0 = 100.0
            prices = [price0]
            for _ in range(1, days):
                drift = (mu - 0.5 * sigma ** 2) * dt
                shock = sigma * np.sqrt(dt) * np.random.normal()
                prices.append(prices[-1] * np.exp(drift + shock))
            prices_series = pd.Series(prices)
            
            # For Buy and Hold, always fully invested from the start
            final_value = self.initial_capital * (prices_series.iloc[-1] / prices_series.iloc[0])
            return {
                "strategy": self.strategy,
                "initial_capital": self.initial_capital,
                "final_portfolio_value": round(final_value, 2),
                "transactions": 1,
                "message": "Trading simulation successful (Buy and Hold)"
            }
        else:
            return {
                "strategy": self.strategy,
                "initial_capital": self.initial_capital,
                "final_portfolio_value": self.initial_capital,
                "transactions": 0,
                "message": "Trading simulation successful (Unknown Strategy)"
            }

    def backtest(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Backtests the trading model with historical data.

        Args:
            historical_data (Dict[str, Any]): The historical data to use for backtesting.
                Expect a key "prices" with a list or pandas Series of prices.
        """
        print(f"Backtesting trading model with strategy: {self.strategy} and data: {historical_data.keys()}")
        prices = historical_data.get("prices", None)
        if prices is None:
            return {
                "strategy": self.strategy,
                "backtest_results": None,
                "historical_data_keys": list(historical_data.keys()),
                "message": "No 'prices' data provided for backtesting."
            }
        
        # Convert prices to a pandas Series if not already
        if not isinstance(prices, pd.Series):
            prices_series = pd.Series(prices)
        else:
            prices_series = prices.copy()
        
        if self.strategy == "Simple Moving Average":
            sma = prices_series.rolling(window=self.sma_window, min_periods=1).mean()
            signals = (prices_series > sma).astype(int)
            signals[signals == 0] = -1
            transactions = int(signals.diff().abs().sum())
            final_value = self.initial_capital * (prices_series.iloc[-1] / prices_series.iloc[0]) if signals.iloc[-1] == 1 else self.initial_capital
            backtest_results = {
                "final_portfolio_value": round(final_value, 2),
                "transactions": transactions
            }
        elif self.strategy == "Buy and Hold":
            final_value = self.initial_capital * (prices_series.iloc[-1] / prices_series.iloc[0])
            backtest_results = {
                "final_portfolio_value": round(final_value, 2),
                "transactions": 1
            }
        else:
            backtest_results = {
                "final_portfolio_value": self.initial_capital,
                "transactions": 0
            }
        
        return {
            "strategy": self.strategy,
            "backtest_results": backtest_results,
            "historical_data_keys": list(historical_data.keys()),
            "message": "Trading model backtest successful"
        }
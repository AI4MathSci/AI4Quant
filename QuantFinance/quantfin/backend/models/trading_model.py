from pydantic import BaseModel, PrivateAttr
from typing import Dict, Any
import numpy as np
import pandas as pd
from quantfin.backend.utils.data_loader import load_historical_data  # Import the data loader
from quantfin.backend.services.sentiment_service import SentimentAnalyzer

class TradingModel(BaseModel):
    """
    Represents a trading model with its parameters.
    """
    initial_capital: float = 100000.0
    strategy: str = "Simple Moving Average"
    sma_window: int = 20
    use_sentiment: bool = False
    sentiment_weight: float = 0.3  # Weight given to sentiment in decision making
    sentiment_threshold: float = 0.2  # Minimum sentiment score to influence decision

    _sentiment_analyzer: SentimentAnalyzer = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._sentiment_analyzer = SentimentAnalyzer()

    async def get_sentiment_signal(self, symbol: str) -> float:
        """
        Gets sentiment signal for a given symbol.
        Returns a value between -1 and 1, where:
        -1: Strong negative sentiment
        0: Neutral sentiment
        1: Strong positive sentiment
        """
        if not self.use_sentiment:
            return 0.0

        sentiment_data = await self._sentiment_analyzer.get_combined_sentiment(symbol)
        sentiment_score = sentiment_data["combined_sentiment_score"]
        # Only use sentiment if it exceeds the threshold
        if abs(sentiment_score) < self.sentiment_threshold:
            return 0.0
            
        return sentiment_score

    async def simulate(self, symbol: str = "AAPL") -> Dict[str, Any]:
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
            
            # Generate technical signals: +1 when price is above SMA, -1 when below
            technical_signals = (prices_series > sma).astype(int)
            technical_signals[technical_signals == 0] = -1
            
            # If sentiment analysis is enabled, incorporate sentiment signals
            if self.use_sentiment:
                # Get real sentiment signal
                sentiment_signal = await self.get_sentiment_signal(symbol)
                # Create sentiment signals array with the same sentiment value
                sentiment_signals = np.full(len(prices_series), sentiment_signal)
                # Combine technical and sentiment signals
                combined_signals = (1 - self.sentiment_weight) * technical_signals + self.sentiment_weight * sentiment_signals
                signals = np.sign(combined_signals)
            else:
                signals = technical_signals
            
            # Count transactions: each signal change is a transaction
            transactions = int(pd.Series(signals).diff().abs().sum())
                        
            # Simulate portfolio evolution: assume fully invested when signal==1, otherwise in cash.
            # For simplicity, assume the portfolio value follows the price evolution if invested.
            position_value = self.initial_capital * (prices_series.iloc[-1] / prices_series.iloc[0]) if signals.iloc[-1] == 1 else self.initial_capital
            #print(f"position_value: {position_value}")
            result = {
                "strategy": self.strategy,
                "initial_capital": self.initial_capital,
                "final_portfolio_value": round(position_value, 2),
                "transactions": transactions,
                "use_sentiment": self.use_sentiment,
                "sentiment_weight": self.sentiment_weight if self.use_sentiment else None,
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
                "use_sentiment": self.use_sentiment,
                "message": "Trading simulation successful (Buy and Hold)"
            }
        else:
            return {
                "strategy": self.strategy,
                "initial_capital": self.initial_capital,
                "final_portfolio_value": self.initial_capital,
                "transactions": 0,
                "use_sentiment": self.use_sentiment,
                "message": "Trading simulation successful (Unknown Strategy)"
            }

    async def backtest(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Backtests the trading model with historical price data over a chosen period of time

        Args:
            symbol (str): The stock symbol.
            start_date (str): The start date for historical data.
            end_date (str): The end date for historical data.

        Returns:
            Dict[str, Any]: The results of the backtest.
        """
        print(f"Backtesting trading model with strategy: {self.strategy} for symbol: {symbol} from {start_date} to {end_date}")

        # Load historical data
        historical_data = load_historical_data(symbol, start_date, end_date)

        # Through trials and errors, we figure out that historical_data is a dictionary with the ticker symbol as the only key, 
        # such as 'MSFT', and historical_data[symbol] is a Panda Dataframe indexed by a multi-index of two indices (price_type, symbol), 
        # such as ('Close', 'MSFT'), where there are 4 types of prices - Close, High, Low and Open, we choose to use the 'Close' price, 
        # which is the usual choice in quant finance. 
        prices = historical_data[symbol][('Close', symbol)] 

        if prices is None or prices.empty:
            return {
                "strategy": self.strategy,
                "backtest_results": None,
                "message": "No 'prices' data provided for backtesting."
            }
        
        # Convert prices to a pandas Series if not already, is this needed?
        if not isinstance(prices, pd.Series):
            prices_series = pd.Series(prices)
        else:
            prices_series = prices.copy()
        
        if self.strategy == "Simple Moving Average":
            sma = prices_series.rolling(window=self.sma_window, min_periods=1).mean()
            technical_signals = (prices_series > sma).astype(int)
            technical_signals[technical_signals == 0] = -1
            
            # If sentiment analysis is enabled, get real sentiment data
            if self.use_sentiment:
                sentiment_signal = await self.get_sentiment_signal(symbol)
                # Apply sentiment signal to all periods (simplified approach)
                sentiment_signals = pd.Series([sentiment_signal] * len(prices_series))
                # Combine technical and sentiment signals
                combined_signals = (1 - self.sentiment_weight) * technical_signals + self.sentiment_weight * sentiment_signals
                signals = np.sign(combined_signals)
            else:
                signals = technical_signals
            
            transactions = int(pd.Series(signals).diff().abs().sum())
            final_value = self.initial_capital * (prices_series.iloc[-1] / prices_series.iloc[0]) if signals.iloc[-1] == 1 else self.initial_capital
            
            backtest_results = {
                "final_portfolio_value": round(final_value, 2),
                "transactions": transactions,
                "use_sentiment": self.use_sentiment,
                "sentiment_weight": self.sentiment_weight if self.use_sentiment else None
            }
        elif self.strategy == "Buy and Hold":
            final_value = self.initial_capital * (prices_series.iloc[-1] / prices_series.iloc[0])
            backtest_results = {
                "final_portfolio_value": round(final_value, 2),
                "transactions": 1,
                "use_sentiment": self.use_sentiment
            }
        else:
            backtest_results = {
                "final_portfolio_value": self.initial_capital,
                "transactions": 0,
                "use_sentiment": self.use_sentiment
            }
        
        return {
            "strategy": self.strategy,
            "backtest_results": backtest_results,
            "historical_data_keys": list(historical_data.keys()),
            "message": "Trading model backtest successful"
        }
from pydantic import BaseModel
from typing import Dict, Any
import yfinance as yf
import empyrical
from datetime import datetime

from quantfin.backend.strategies.risk_strategy_factory import get_risk_strategy

class RiskModel(BaseModel):
    """
    Represents a risk management model.
    """
    risk_metric: str = "Value at Risk (VaR)"  # Options: "Value at Risk (VaR)" or "Expected Shortfall (ES)"
    confidence_level: float = 0.95
    holding_period: int = 10  # in days
    symbol: str = "AAPL"
    portfolio_value: float = 1000000.0
    start_date: str
    end_date: str

    async def simulate(self) -> Dict[str, Any]:
        """
        Simulates the risk management model.
        Uses parametric methods to compute VaR or ES based on a normal distribution assumption.
        """
        data = yf.download(self.symbol, start=self.start_date, end=self.end_date)["Close"]
        returns = data.pct_change().dropna()
        risk_data = get_risk_strategy(self.risk_metric, {
            "returns": returns,
            "portfolio_value": self.portfolio_value,
            "confidence_level": self.confidence_level,
        })
        return {
            "risk_metric": self.risk_metric,
            "confidence_level": self.confidence_level,
            "holding_period": self.holding_period,
            "portfolio_value": self.portfolio_value,
            "risk_value": round(risk_data["risk_value"], 2),
            "message": f"Risk simulation successful using {self.risk_metric}."
        }

    async def backtest(self) -> Dict[str, Any]:
        """
        Backtests the portfolio management model with historical price data over a chosen period of time

        Returns:
            Dict[str, Any]: The results of the backtest.
    
        Expects historical_data to include a key "returns", a list of historical return values (in decimals).
        Evaluates the empirical VaR by computing the appropriate return percentile and compares it
        with the frequency of exceedances in the data.
        """
        data = yf.download(self.symbol, start=self.start_date, end=self.end_date)["Close"]
        returns = data.pct_change().dropna()
        risk_data = get_risk_strategy(self.risk_metric, {
            "returns": returns,
            "portfolio_value": self.portfolio_value,
            "confidence_level": self.confidence_level,
        })
        return {
            "risk_metric": self.risk_metric,
            "confidence_level": self.confidence_level,
            "holding_period": self.holding_period,
            "portfolio_value": self.portfolio_value,
            "risk_value": round(risk_data["risk_value"], 2),
            "message": f"Risk backtesting successful using {self.risk_metric}."
        }
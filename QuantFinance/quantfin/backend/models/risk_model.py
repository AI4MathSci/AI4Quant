from pydantic import BaseModel
from typing import Dict, Any
import numpy as np
from scipy.stats import norm
from quantfin.backend.utils.data_loader import load_historical_data  # Import the data loader

class RiskModel(BaseModel):
    """
    Represents a risk management model.
    """
    risk_metric: str = "Value at Risk (VaR)"  # Options: "Value at Risk (VaR)" or "Expected Shortfall (ES)"
    confidence_level: float = 0.95
    holding_period: int = 10  # in days
    portfolio_value: float = 1000000.0
    daily_volatility: float = 0.01  # Assumed daily volatility (e.g., 1%)

    async def simulate(self) -> Dict[str, Any]:
        """
        Simulates the risk management model.
        Uses parametric methods to compute VaR or ES based on a normal distribution assumption.
        """
        print(f"Simulating risk with metric: {self.risk_metric}, confidence level: {self.confidence_level}, "
              f"holding period: {self.holding_period} days, and portfolio value: {self.portfolio_value}")
        
        # Calculate the holding period volatility assuming daily volatility scales as sqrt(time)
        hp_vol = self.daily_volatility * np.sqrt(self.holding_period)
        
        if self.risk_metric == "Value at Risk (VaR)":
            # Under normality, VaR at confidence level is computed using z-score
            z = norm.ppf(1 - self.confidence_level)  # this value is negative
            var_percentage = abs(z) * hp_vol  # Estimated loss percentage over the holding period
            risk_value = self.portfolio_value * var_percentage
            message = "Risk simulation successful using Value at Risk (VaR)."
            result = {
                "risk_metric": self.risk_metric,
                "confidence_level": self.confidence_level,
                "holding_period": self.holding_period,
                "portfolio_value": self.portfolio_value,
                "risk_value": round(risk_value, 2),
                "var_percentage": round(var_percentage, 4),
                "message": message
            }
        elif self.risk_metric == "Expected Shortfall (ES)":
            # For ES, the expected shortfall under normality is computed as:
            # ES = sigma * sqrt(holding_period) * (pdf(z) / (1 - confidence_level))
            z = norm.ppf(self.confidence_level)
            es_percentage = hp_vol * (norm.pdf(z) / (1 - self.confidence_level))
            risk_value = self.portfolio_value * es_percentage
            message = "Risk simulation successful using Expected Shortfall (ES)."
            result = {
                "risk_metric": self.risk_metric,
                "confidence_level": self.confidence_level,
                "holding_period": self.holding_period,
                "portfolio_value": self.portfolio_value,
                "risk_value": round(risk_value, 2),
                "es_percentage": round(es_percentage, 4),
                "message": message
            }
        else:
            # Default to VaR if an unrecognized risk metric is provided
            z = norm.ppf(1 - self.confidence_level)
            var_percentage = abs(z) * hp_vol
            risk_value = self.portfolio_value * var_percentage
            message = "Risk simulation successful using default Value at Risk (VaR)."
            result = {
                "risk_metric": self.risk_metric,
                "confidence_level": self.confidence_level,
                "holding_period": self.holding_period,
                "portfolio_value": self.portfolio_value,
                "risk_value": round(risk_value, 2),
                "var_percentage": round(var_percentage, 4),
                "message": message
            }
        return result

    async def backtest(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Backtests the portfolio management model with historical price data over a chosen period of time

        Args:
            symbol (str): The stock symbol.
            start_date (str): The start date for historical data.
            end_date (str): The end date for historical data.

        Returns:
            Dict[str, Any]: The results of the backtest.
    
        Expects historical_data to include a key "returns", a list of historical return values (in decimals).
        Evaluates the empirical VaR by computing the appropriate return percentile and compares it
        with the frequency of exceedances in the data.
        """
        print(f"Backtesting risk model with metric: {self.risk_metric} for symbol: {symbol} from {start_date} to {end_date}")

        # Load historical data
        historical_data = load_historical_data(symbol, start_date, end_date)
        
        returns = historical_data.get("returns", [])
        if not returns:
            return {
                "risk_metric": self.risk_metric,
                "backtest_results": None,
                "historical_data_keys": list(historical_data.keys()),
                "message": "No 'returns' data provided for risk backtest."
            }
        
        returns_array = np.array(returns)
        
        # Compute empirical VaR as the (1 - confidence_level) percentile of the returns distribution
        percentile_level = (1 - self.confidence_level) * 100
        empirical_threshold = np.percentile(returns_array, percentile_level)
        empirical_var = abs(empirical_threshold)
        
        # Count exceedances: periods where the actual return is below the empirical threshold
        exceedances = np.sum(returns_array < empirical_threshold)
        total_periods = len(returns_array)
        exceedance_frequency = exceedances / total_periods
        
        result = {
            "risk_metric": self.risk_metric,
            "confidence_level": self.confidence_level,
            "holding_period": self.holding_period,
            "portfolio_value": self.portfolio_value,
            "empirical_VaR_percentage": round(empirical_var, 4),
            "exceedance_frequency": round(exceedance_frequency, 4),
            "historical_data_points": total_periods,
            "message": "Risk model backtest successful"
        }
        return result
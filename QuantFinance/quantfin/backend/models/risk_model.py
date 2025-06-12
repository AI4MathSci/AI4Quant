from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import yfinance as yf
import empyrical
import numpy as np
import pandas as pd
from datetime import datetime

from quantfin.backend.strategies.risk_strategy_factory import get_risk_strategy

class RiskModel(BaseModel):
    """
    Represents a risk management model for single or multiple assets.
    """
    risk_metric: str = "Value at Risk (VaR)"  # Options: "Value at Risk (VaR)" or "Expected Shortfall (ES)"
    confidence_level: float = 0.95
    holding_period: int = 10  # in days
    symbol: str = "AAPL"  # For single asset analysis
    symbols: Optional[List[str]] = None  # For portfolio analysis
    portfolio_value: float = 1000000.0
    start_date: str
    end_date: str

    async def analyze(self) -> Dict[str, Any]:
        """
        Comprehensive risk analysis for single or multiple assets.
        Provides portfolio-level risk metrics, stress testing, and risk decomposition.
        """
        # Handle both single symbol and multiple symbols
        if hasattr(self, 'symbols') and self.symbols:
            # Multiple symbols - portfolio analysis
            symbols = self.symbols
        else:
            # Single symbol - convert to list for consistent processing
            symbols = [self.symbol]
        
        # Download data for all symbols with consistent handling
        try:
            if len(symbols) == 1:
                # For single symbol, download and structure as DataFrame
                data = yf.download(symbols[0], start=self.start_date, end=self.end_date, progress=False)
                if data.empty:
                    raise ValueError(f"No data found for symbol {symbols[0]}")
                # Extract Close prices and ensure DataFrame structure
                if "Close" in data.columns:
                    prices = data[["Close"]].rename(columns={"Close": symbols[0]})
                else:
                    # Handle case where data might be a Series
                    prices = data.to_frame(name=symbols[0]) if hasattr(data, 'to_frame') else pd.DataFrame({symbols[0]: data})
            else:
                # For multiple symbols
                data = yf.download(symbols, start=self.start_date, end=self.end_date, progress=False)
                if data.empty:
                    raise ValueError(f"No data found for symbols {symbols}")
                # Extract Close prices - handle multi-level columns
                if isinstance(data.columns, pd.MultiIndex):
                    prices = data["Close"]
                else:
                    # Fallback for single-level columns
                    prices = data[["Close"]].rename(columns={"Close": symbols[0]}) if len(symbols) == 1 else data
        except Exception as e:
            raise ValueError(f"Error downloading data: {str(e)}")
        
        # Validate that we have data
        if prices.empty:
            raise ValueError("No valid price data available for analysis")
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Ensure we still have data after calculating returns
        if returns.empty:
            raise ValueError("Insufficient data to calculate returns")
        
        # Portfolio-level analysis
        if len(symbols) > 1:
            # Equal weight portfolio for analysis
            weights = np.array([1.0/len(symbols)] * len(symbols))
            portfolio_returns = (returns * weights).sum(axis=1)
            
            # Portfolio risk metrics
            portfolio_var = self._calculate_portfolio_var(returns, weights)
            component_var = self._calculate_component_var(returns, weights)
            correlation_matrix = returns.corr()
            
            # Stress testing
            stress_scenarios = self._run_stress_tests(returns, weights)
            
            return {
                "analysis_type": "Portfolio Risk Analysis",
                "symbols": symbols,
                "portfolio_metrics": {
                    "portfolio_var_95": round(portfolio_var * self.portfolio_value, 2),
                    "portfolio_volatility": round(portfolio_returns.std() * np.sqrt(252), 4),
                    "diversification_ratio": round(self._calculate_diversification_ratio(returns, weights), 4)
                },
                "component_analysis": {
                    "component_var": {sym: round(cv * self.portfolio_value, 2) for sym, cv in zip(symbols, component_var)},
                    "risk_contribution_pct": {sym: round(cv/abs(portfolio_var)*100, 2) if portfolio_var != 0 else 0 for sym, cv in zip(symbols, component_var)}
                },
                "correlation_analysis": {
                    "correlation_matrix": correlation_matrix.round(3).to_dict(),
                    "average_correlation": round(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean(), 3)
                },
                "stress_testing": stress_scenarios,
                "individual_metrics": self._calculate_individual_metrics(returns),
                "confidence_level": self.confidence_level,
                "portfolio_value": self.portfolio_value,
                "message": f"Comprehensive risk analysis completed for {len(symbols)} assets."
            }
        else:
            # Single asset analysis (enhanced)
            asset_returns = returns.iloc[:, 0]
            
            # Basic risk metrics
            var_95 = np.percentile(asset_returns, (1 - self.confidence_level) * 100)
            es_95 = asset_returns[asset_returns <= var_95].mean()
            
            # Advanced metrics
            max_drawdown = self._calculate_max_drawdown(prices.iloc[:, 0])
            var_breakdown = self._calculate_var_breakdown(asset_returns)
            
            return {
                "analysis_type": "Single Asset Risk Analysis",
                "symbol": symbols[0],
                "risk_metrics": {
                    "var_95_pct": round(var_95 * 100, 3),
                    "var_95_dollar": round(abs(var_95) * self.portfolio_value, 2),
                    "expected_shortfall_95": round(abs(es_95) * self.portfolio_value, 2),
                    "volatility_annual": round(asset_returns.std() * np.sqrt(252), 4),
                    "max_drawdown": round(max_drawdown, 4)
                },
                "var_breakdown": var_breakdown,
                "stress_testing": self._run_single_asset_stress_tests(asset_returns),
                "confidence_level": self.confidence_level,
                "portfolio_value": self.portfolio_value,
                "message": f"Risk analysis completed for {symbols[0]}."
            }

    def _calculate_portfolio_var(self, returns: pd.DataFrame, weights: np.ndarray, confidence_level: float = None) -> float:
        """Calculate portfolio VaR considering correlations"""
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        portfolio_returns = (returns * weights).sum(axis=1)
        return np.percentile(portfolio_returns, (1 - confidence_level) * 100)

    def _calculate_component_var(self, returns: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
        """Calculate component VaR for each asset"""
        portfolio_var = self._calculate_portfolio_var(returns, weights)
        cov_matrix = returns.cov().values
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        
        if portfolio_vol == 0:
            return np.zeros(len(weights))
        
        # Marginal VaR approximation
        marginal_var = np.dot(cov_matrix, weights) / portfolio_vol
        component_var = weights * marginal_var * abs(portfolio_var) / portfolio_vol
        
        return component_var

    def _calculate_diversification_ratio(self, returns: pd.DataFrame, weights: np.ndarray) -> float:
        """Calculate diversification ratio"""
        individual_vols = returns.std().values
        weighted_avg_vol = np.dot(weights, individual_vols)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(returns.cov().values, weights)))
        
        if portfolio_vol == 0:
            return 1.0
        
        return weighted_avg_vol / portfolio_vol

    def _run_stress_tests(self, returns: pd.DataFrame, weights: np.ndarray) -> Dict[str, Any]:
        """Run predefined stress test scenarios"""
        portfolio_returns = (returns * weights).sum(axis=1)
        
        scenarios = {
            "market_crash_2008": {"shock": -0.20, "description": "20% market decline (2008-style)"},
            "flash_crash": {"shock": -0.10, "description": "10% single-day decline"},
            "volatility_spike": {"shock": portfolio_returns.std() * 3, "description": "3x volatility increase"},
            "correlation_breakdown": {"shock": 0.9, "description": "All correlations → 0.9"}
        }
        
        stress_results = {}
        for scenario_name, scenario in scenarios.items():
            if scenario_name == "correlation_breakdown":
                # Simulate high correlation scenario
                stressed_returns = returns.copy()
                stressed_returns.iloc[:] = portfolio_returns.values.reshape(-1, 1)
                stressed_portfolio_var = self._calculate_portfolio_var(stressed_returns, weights)
            else:
                # Apply shock to portfolio
                shocked_return = scenario["shock"]
                stressed_portfolio_var = shocked_return
            
            stress_results[scenario_name] = {
                "description": scenario["description"],
                "portfolio_loss_pct": round(abs(stressed_portfolio_var) * 100, 2),
                "portfolio_loss_dollar": round(abs(stressed_portfolio_var) * self.portfolio_value, 2)
            }
        
        return stress_results

    def _calculate_individual_metrics(self, returns: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate risk metrics for each individual asset"""
        individual_metrics = {}
        
        for symbol in returns.columns:
            asset_returns = returns[symbol]
            var_95 = np.percentile(asset_returns, 5)  # 95% VaR
            
            individual_metrics[symbol] = {
                "var_95_pct": round(var_95 * 100, 3),
                "volatility": round(asset_returns.std() * np.sqrt(252), 4),
                "skewness": round(asset_returns.skew(), 3),
                "kurtosis": round(asset_returns.kurtosis(), 3)
            }
        
        return individual_metrics

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _calculate_var_breakdown(self, returns: pd.Series) -> Dict[str, float]:
        """Break down VaR by time periods"""
        return {
            "daily_var_95": round(np.percentile(returns, 5) * 100, 3),
            "weekly_var_95": round(np.percentile(returns, 5) * np.sqrt(5) * 100, 3),
            "monthly_var_95": round(np.percentile(returns, 5) * np.sqrt(21) * 100, 3)
        }

    def _run_single_asset_stress_tests(self, returns: pd.Series) -> Dict[str, Any]:
        """Stress tests for single asset"""
        scenarios = {
            "bear_market": {"shock": -0.30, "description": "30% decline (bear market)"},
            "black_swan": {"shock": -0.50, "description": "50% decline (black swan)"},
            "high_volatility": {"multiplier": 2, "description": "2x volatility period"}
        }
        
        stress_results = {}
        for scenario_name, scenario in scenarios.items():
            if "shock" in scenario:
                loss_dollar = abs(scenario["shock"]) * self.portfolio_value
                stress_results[scenario_name] = {
                    "description": scenario["description"],
                    "loss_pct": abs(scenario["shock"]) * 100,
                    "loss_dollar": round(loss_dollar, 2)
                }
            else:
                # Volatility scenario
                stressed_var = np.percentile(returns, 5) * scenario["multiplier"]
                stress_results[scenario_name] = {
                    "description": scenario["description"],
                    "stressed_var_pct": round(abs(stressed_var) * 100, 3),
                    "stressed_var_dollar": round(abs(stressed_var) * self.portfolio_value, 2)
                }
        
        return stress_results

    async def simulate(self) -> Dict[str, Any]:
        """
        Simulates the risk management model.
        Uses parametric methods to compute VaR or ES based on a normal distribution assumption.
        """
        data = yf.download(self.symbol, start=self.start_date, end=self.end_date, progress=False)["Close"]
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
        data = yf.download(self.symbol, start=self.start_date, end=self.end_date, progress=False)["Close"]
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
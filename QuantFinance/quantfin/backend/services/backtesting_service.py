from typing import Dict, Any
from quantfin.backend.models.trading_model import TradingModel
from quantfin.backend.models.portfolio_model import PortfolioModel
from quantfin.backend.models.risk_model import RiskModel
import logging

logger = logging.getLogger(__name__)

async def run_trading_backtest(model: TradingModel, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Backtests a trading model with historical data.

    Args:
        model (TradingModel): The trading model to backtest.
        symbol (str): The stock symbol.
        start_date (str): The start date for historical data.
        end_date (str): The end date for historical data.
 
    Returns:
        Dict[str, Any]: The results of the backtest.
    """
    logger.info("Running trading backtest")
    return await model.backtest(symbol, start_date, end_date)

async def run_portfolio_backtest(model: PortfolioModel, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Backtests a portfolio model with historical data.

    Args:
        model (PortfolioModel): The portfolio model to backtest.
        symbol (str): The stock symbol.
        start_date (str): The start date for historical data.
        end_date (str): The end date for historical data.

    Returns:
        Dict[str, Any]: The results of the backtest.
    """
    logger.info("Running portfolio backtest")
    return await model.backtest(symbol, start_date, end_date)

async def run_risk_backtest(model: RiskModel, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Backtests a risk model with historical data.

    Args:
        model (RiskModel): The risk model to backtest.
        symbol (str): The stock symbol.
        start_date (str): The start date for historical data.
        end_date (str): The end date for historical data.

    Returns:
        Dict[str, Any]: The results of the backtest.
    """
    logger.info("Running risk backtest")
    return await model.backtest(symbol, start_date, end_date)
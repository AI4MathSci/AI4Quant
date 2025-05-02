from typing import Dict, Any
#from ..models.trading_model import TradingModel
#from ..models.portfolio_model import PortfolioModel
#from ..models.risk_model import RiskModel
from quantfin.backend.models.trading_model import TradingModel
from quantfin.backend.models.portfolio_model import PortfolioModel
from quantfin.backend.models.risk_model import RiskModel
import logging

logger = logging.getLogger(__name__)

def run_trading_backtest(model: TradingModel, historical_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backtests a trading model with historical data.

    Args:
        model (TradingModel): The trading model to backtest.
        historical_data (Dict[str, Any]): The historical data.

    Returns:
        Dict[str, Any]: The results of the backtest.
    """
    logger.info("Running trading backtest")
    return model.backtest(historical_data)

def run_portfolio_backtest(model: PortfolioModel, historical_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backtests a portfolio model with historical data.

    Args:
        model (PortfolioModel): The portfolio model to backtest.
        historical_data (Dict[str, Any]): The historical data.

    Returns:
        Dict[str, Any]: The results of the backtest.
    """
    logger.info("Running portfolio backtest")
    return model.backtest(historical_data)

def run_risk_backtest(model: RiskModel, historical_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backtests a risk model with historical data.

    Args:
        model (RiskModel): The risk model to backtest.
        historical_data (Dict[str, Any]): The historical data.

    Returns:
        Dict[str, Any]: The results of the backtest.
    """
    logger.info("Running risk backtest")
    return model.backtest(historical_data)
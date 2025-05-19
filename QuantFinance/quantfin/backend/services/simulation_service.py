from typing import Dict, Any
from quantfin.backend.models.trading_model import TradingModel
from quantfin.backend.models.portfolio_model import PortfolioModel
from quantfin.backend.models.risk_model import RiskModel
import logging

logger = logging.getLogger(__name__)

async def run_trading_simulation(model: TradingModel, symbol: str = "AAPL") -> Dict[str, Any]:
    """
    Runs a trading simulation using the provided model.

    Args:
        model (TradingModel): The trading model to simulate.
        symbol (str): The stock symbol to use for sentiment analysis. Defaults to "AAPL".

    Returns:
        Dict[str, Any]: The results of the simulation.
    """
    logger.info(f"Running trading simulation with symbol: {symbol}")
    return await model.simulate(symbol)

async def run_portfolio_simulation(model: PortfolioModel) -> Dict[str, Any]:
    """
    Runs a portfolio simulation using the provided model.

    Args:
        model (PortfolioModel): The portfolio model to simulate.

    Returns:
        Dict[str, Any]: The results of the simulation.
    """
    logger.info("Running portfolio simulation")
    return await model.simulate()

async def run_risk_simulation(model: RiskModel) -> Dict[str, Any]:
    """
    Runs a risk simulation using the provided model.

    Args:
        model (RiskModel): The risk model to simulate.

    Returns:
        Dict[str, Any]: The results of the simulation.
    """
    logger.info("Running risk simulation")
    return await model.simulate()
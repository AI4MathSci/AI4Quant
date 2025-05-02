from typing import Dict, Any
from quantfin.backend.models.trading_model import TradingModel
from quantfin.backend.models.portfolio_model import PortfolioModel
from quantfin.backend.models.risk_model import RiskModel
import logging

logger = logging.getLogger(__name__)

def run_trading_simulation(model: TradingModel) -> Dict[str, Any]:
    """
    Runs a trading simulation using the provided model.

    Args:
        model (TradingModel): The trading model to simulate.

    Returns:
        Dict[str, Any]: The results of the simulation.
    """
    logger.info("Running trading simulation")
    return model.simulate()

def run_portfolio_simulation(model: PortfolioModel) -> Dict[str, Any]:
    """
    Runs a portfolio simulation using the provided model.

    Args:
        model (PortfolioModel): The portfolio model to simulate.

    Returns:
        Dict[str, Any]: The results of the simulation.
    """
    logger.info("Running portfolio simulation")
    return model.simulate()

def run_risk_simulation(model: RiskModel) -> Dict[str, Any]:
    """
    Runs a risk simulation using the provided model.

    Args:
        model (RiskModel): The risk model to simulate.

    Returns:
        Dict[str, Any]: The results of the simulation.
    """
    logger.info("Running risk simulation")
    return model.simulate()
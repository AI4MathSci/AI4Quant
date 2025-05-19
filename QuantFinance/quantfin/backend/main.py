from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
#from .services import simulation_service, backtesting_service
#from .models import trading_model, portfolio_model, risk_model
from quantfin.backend.services import simulation_service, backtesting_service
from quantfin.backend.models import trading_model, portfolio_model, risk_model
from fastapi.middleware.cors import CORSMiddleware
import logging
from quantfin.backend.config import config  # Import the config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware to allow cross-origin requests (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Data validation models using Pydantic
class SimulationRequest(BaseModel):
    model_name: str
    parameters: Dict[str, Any]
    symbol: str = "AAPL"  # Default to AAPL if not provided

class BacktestRequest(BaseModel):
    model_name: str
    parameters: Dict[str, Any]
    symbol: str
    start_date: str
    end_date: str

class HealthCheckResponse(BaseModel):
    status: str = "OK"
    # version: str = config.app_version  # Access version from config   # comment this out since it raises error of "config does not have attrivute app_version"
    version: str = "Version 01"  # put this in to replace the above line during debugging to avoid the error

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    """
    return {"status": "OK", "version": "0.1.0"}

@app.post("/simulate")
async def run_simulation(request: SimulationRequest) -> Dict[str, Any]:
    """
    Endpoint for running quantitative finance simulations.

    Args:
        request (SimulationRequest): The simulation request containing the model name, parameters, and symbol.

    Returns:
        Dict[str, Any]: The results of the simulation.

    Raises:
        HTTPException: If the model is not found or an error occurs during simulation.
    """
    try:
        if request.model_name == "Algorithmic Trading Model":
            model = trading_model.TradingModel(**request.parameters)  # type: ignore
            results = await simulation_service.run_trading_simulation(model, request.symbol)
        elif request.model_name == "Portfolio Management Model":
            model = portfolio_model.PortfolioModel(**request.parameters) # type: ignore
            results = await simulation_service.run_portfolio_simulation(model)
        elif request.model_name == "Risk Management Model":
            model = risk_model.RiskModel(**request.parameters) # type: ignore
            results = await simulation_service.run_risk_simulation(model)
        else:
            raise HTTPException(status_code=400, detail="Model not found")
        return results
    except Exception as e:
        logger.error(f"Error running simulation: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@app.post("/backtest")
async def run_backtest(request: BacktestRequest) -> Dict[str, Any]:
    """
    Endpoint for backtesting quantitative finance models with historical data.

    Args:
        request (BacktestRequest): The backtest request containing the model name, parameters, and historical data.

    Returns:
        Dict[str, Any]: The results of the backtest.

    Raises:
        HTTPException: If the model is not found or an error occurs during backtesting.
    """
    try:
        if request.model_name == "Algorithmic Trading Model":
            model = trading_model.TradingModel(**request.parameters) # type: ignore
            results = await backtesting_service.run_trading_backtest(model, request.symbol, request.start_date, request.end_date)
        elif request.model_name == "Portfolio Management Model":
            model = portfolio_model.PortfolioModel(**request.parameters) # type: ignore
            results = await backtesting_service.run_portfolio_backtest(model, request.symbol, request.start_date, request.end_date)
        elif request.model_name == "Risk Management Model":
            model = risk_model.RiskModel(**request.parameters) # type: ignore
            results = await backtesting_service.run_risk_backtest(model, request.symbol, request.start_date, request.end_date)
        else:
            raise HTTPException(status_code=400, detail="Model not found")
        return results
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
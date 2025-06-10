from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import os
import logging

from quantfin.backend.models.trading_model import TradingModel
from quantfin.backend.models.portfolio_model import PortfolioModel
from quantfin.backend.models.risk_model import RiskModel
from quantfin.backend.config.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="QuantFin API", version="1.0.0")

# Add CORS middleware to allow cross-origin requests (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def root():
    return {"message": "QuantFin API is running"}

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    """
    return {"status": "OK", "version": "1.0.0"}

@app.get("/config/openai-status")
async def get_openai_status():
    """Check if OpenAI API key is configured for sentiment analysis"""
    return {"configured": config.has_openai_key}

# Trading endpoints
@app.post("/trading/simulate/{symbol}")
async def trading_simulate(symbol: str, request: dict):
    """Simulate trading strategy"""
    try:
        model = TradingModel(**request)
        return await model.simulate(symbol)
    except Exception as e:
        import traceback
        logger.error(f"Trading simulation error: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/trading/backtest/{symbol}")
async def trading_backtest(symbol: str, request: dict):
    """Backtest trading strategy"""
    try:
        model = TradingModel(**request)
        return await model.backtest(symbol)
    except Exception as e:
        logger.error(f"Trading backtest error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Portfolio endpoints
@app.post("/portfolio/optimize")
async def portfolio_optimize(request: dict):
    """Optimize portfolio"""
    try:
        model = PortfolioModel(**request)
        return await model.optimize()
    except Exception as e:
        logger.error(f"Portfolio optimization error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/portfolio/simulate")
async def portfolio_simulate(request: dict):
    """Simulate portfolio strategy"""
    try:
        model = PortfolioModel(**request)
        return await model.simulate()
    except Exception as e:
        logger.error(f"Portfolio simulation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/portfolio/backtest")
async def portfolio_backtest(request: dict):
    """Backtest portfolio strategy"""
    try:
        model = PortfolioModel(**request)
        return await model.backtest()
    except Exception as e:
        logger.error(f"Portfolio backtest error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Risk endpoints
@app.post("/risk/analyze")
async def risk_analyze(request: dict):
    """Analyze risk"""
    try:
        model = RiskModel(**request)
        return await model.analyze()
    except Exception as e:
        logger.error(f"Risk analysis error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/risk/simulate")
async def risk_simulate(request: dict):
    """Simulate risk analysis"""
    try:
        model = RiskModel(**request)
        return await model.simulate()
    except Exception as e:
        logger.error(f"Risk simulation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/risk/backtest")
async def risk_backtest(request: dict):
    """Backtest risk analysis"""
    try:
        model = RiskModel(**request)
        return await model.backtest()
    except Exception as e:
        logger.error(f"Risk backtest error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
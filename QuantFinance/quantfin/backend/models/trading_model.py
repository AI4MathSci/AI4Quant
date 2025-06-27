from pydantic import BaseModel
from typing import Dict, Any, Optional, List, Tuple
import backtrader as bt
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
import numpy as np
from itertools import product
import pandas_market_calendars as mcal

from quantfin.backend.strategies.trading_strategy_factory import get_trading_strategy
from quantfin.backend.config.config import config
from quantfin.backend.analysis.llamaindex_engine import QuantFinLlamaEngine


logger = logging.getLogger(__name__)

# Global sentiment cache shared across all instances
_sentiment_cache = {}

class TradingModel(BaseModel):
    """
    Represents a trading model with its parameters.
    """
    initial_capital: float = 100000.0
    strategy: str = "Simple Moving Average"
    sma_window: int = 20
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    bb_period: int = 20
    bb_dev: float = 2.0
    start_date: str
    end_date: str
    
    # Simplified Sentiment Parameters (LlamaIndex Only)
    use_sentiment: bool = False
    sentiment_weight: float = 0.3
    sentiment_threshold: float = 0.1
    
    # LlamaIndex Analysis Controls
    analysis_scope: str = "news"  # "news", "comprehensive", "filings"
    include_risk_analysis: bool = False
    include_catalyst_analysis: bool = False
    sentiment_confidence_threshold: float = 0.5
    
    # Sentiment API Call Controls
    sentiment_api_frequency: int = 5  # Call API every N trading days
    sentiment_decay_factor: float = 0.9  # Decay factor for sentiment over time
    sentiment_combo_weight: float = 0.5  # Weight for combining decayed and keyword-based sentiment (0.5 = equal weight)
    
    # New Optimization Controls
    enable_optimization: bool = True
    optimization_split_ratio: float = 0.7  # 70% training, 30% testing
    compare_sentiment_versions: bool = True
    optimization_speed: str = "fast"  # "fast", "balanced", "thorough"
    max_combinations: int = 400
    
    # Instance-level engine
    _llama_engine: Optional[QuantFinLlamaEngine] = None
    
    class Config:
        # Allow private attributes in Pydantic model
        arbitrary_types_allowed = True

    def _get_sentiment_cache_key(self, symbol: str, date: str = None) -> str:
        """Generate cache key for sentiment data"""
        if date:
            return f"sentiment_{symbol}_{date}_{self.analysis_scope}"
        else:
            return f"sentiment_{symbol}_{self.start_date}_{self.end_date}_{self.analysis_scope}"

    def _get_keyword_sentiment(self, text: str) -> Dict[str, Any]:
        """Get sentiment using keyword-based method"""
        text_lower = text.lower()
        bullish_indicators = ["bullish", "positive", "optimistic", "strong outlook", "growth potential", "buy", "uptrend"]
        bearish_indicators = ["bearish", "negative", "pessimistic", "weak outlook", "declining", "sell", "downtrend"]
        
        bullish_score = sum(1 for indicator in bullish_indicators if indicator in text_lower)
        bearish_score = sum(1 for indicator in bearish_indicators if indicator in text_lower)
        
        if bullish_score > bearish_score:
            sentiment = "bullish"
            score = min(0.8, 0.3 + (bullish_score - bearish_score) * 0.1)
        elif bearish_score > bullish_score:
            sentiment = "bearish" 
            score = max(-0.8, -0.3 - (bearish_score - bullish_score) * 0.1)
        else:
            sentiment = "neutral"
            score = 0.0
        
        confidence = min(0.9, 0.4 + len(text.split()) / 1000)
        
        return {
            "sentiment": sentiment,
            "score": score,
            "confidence": confidence
        }

    async def _build_sentiment_data(self, symbol: str, trading_days) -> Dict[str, Any]:
        """
        Build sentiment data dictionary for a given trading period with date-specific caching.
        
        Args:
            symbol: Stock symbol
            trading_days: List of trading days
            
        Returns:
            Dictionary indexed by trading date (YYYY-MM-DD) -> sentiment info dict or None
        """
        # Build new sentiment data
        logger.info(f"Building new sentiment data for {symbol}")
        
        # Pre-fetch sentiment data for all trading days with hybrid fallback
        # sentiment_data: dict indexed by trading date (YYYY-MM-DD) -> sentiment info dict or None
        sentiment_data = {}
        last_sentiment = None
        last_sentiment_date = None

        for i, trading_day in enumerate(trading_days):
            date_str = trading_day.strftime('%Y-%m-%d')
            
            if self.use_sentiment:
                if i % self.sentiment_api_frequency == 0:
                    # API call day - check cache for this specific date
                    cache_key = self._get_sentiment_cache_key(symbol, date_str)
                    
                    # Check global cache for this specific date
                    global _sentiment_cache
                    if cache_key in _sentiment_cache:
                        logger.info(f"Using cached sentiment data for {symbol} on {date_str}")
                        sentiment_data[date_str] = _sentiment_cache[cache_key]
                    else:
                        # API call day - no cache hit
                        sentiment_data[date_str] = await self.get_sentiment_signal(symbol, date=date_str)
                        # Cache the result for this specific date
                        _sentiment_cache[cache_key] = sentiment_data[date_str]
                    
                    last_sentiment = sentiment_data[date_str]
                    last_sentiment_date = trading_day
                else:
                    # Non-API day: use hybrid fallback
                    if last_sentiment is not None:
                        # Decayed/interpolated value
                        days_since = (trading_day - last_sentiment_date).days
                        decayed_score = last_sentiment['score'] * (self.sentiment_decay_factor ** days_since)
                    else:
                        decayed_score = 0.0

                    # Keyword-based fallback - use a simple text analysis
                    # For now, we'll use a placeholder text that could be enhanced with actual news data
                    placeholder_text = f"Market analysis for {symbol} on {date_str}"
                    keyword_result = self._get_keyword_sentiment(placeholder_text)
                    keyword_score = keyword_result.get('score', 0.0)

                    # Combine the two scores
                    combined_score = (self.sentiment_combo_weight * decayed_score + 
                                    (1 - self.sentiment_combo_weight) * keyword_score)

                    # Construct the fallback sentiment dictionary
                    fallback_sentiment = {
                        "sentiment": "bullish" if combined_score > 0.1 else "bearish" if combined_score < -0.1 else "neutral",
                        "score": combined_score,
                        "confidence": 0.5,
                        "enabled": True,
                        "source": "hybrid_fallback"
                    }
                    sentiment_data[date_str] = fallback_sentiment
            else:
                # Sentiment disabled
                sentiment_data[date_str] = None
        
        return sentiment_data

    def _slice_sentiment_data(self, sentiment_data: Dict[str, Any], start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Slice sentiment data to match the specified date range.
        
        Args:
            sentiment_data: Full sentiment data dictionary (date -> sentiment_info)
            start_date: Start date for slicing (YYYY-MM-DD)
            end_date: End date for slicing (YYYY-MM-DD)
        
        Returns:
            Sliced sentiment data for the specified period
        """
        if not sentiment_data:
            logger.warning(f"No sentiment data available for period {start_date} to {end_date}. Proceeding without sentiment.")
            return sentiment_data
        
        # Convert dates to datetime for comparison
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Filter sentiment data for the specified date range
        sliced_sentiment = {}
        for date_str, sentiment_info in sentiment_data.items():
            date_dt = pd.to_datetime(date_str)
            if start_dt <= date_dt <= end_dt:
                sliced_sentiment[date_str] = sentiment_info
        
        # Log if no data found in the specified range
        if not sliced_sentiment:
            available_dates = list(sentiment_data.keys())[:5]  # Show first 5 dates
            logger.warning(f"No sentiment data found in specified range {start_date} to {end_date}. Available dates: {available_dates}...")
        
        return sliced_sentiment

    def _get_optimization_params(self) -> Dict[str, List]:
        """Get parameter ranges for optimization based on strategy and speed setting"""
        
        # Speed-based configurations for base parameters
        speed_configs = {
            "lightning": {
                "sentiment_weight": [0.0, 0.2, 0.4],
                "sentiment_threshold": [0.1, 0.2]
            },
            "fast": {
                "sentiment_weight": [0.0, 0.1, 0.3, 0.5],
                "sentiment_threshold": [0.05, 0.15]
            },
            "medium": {
                "sentiment_weight": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                "sentiment_threshold": [0.05, 0.1, 0.15, 0.2]
            },
            "balanced": {
                "sentiment_weight": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                "sentiment_threshold": [0.05, 0.1, 0.15, 0.2, 0.25]
            },
            "thorough": {
                "sentiment_weight": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
                "sentiment_threshold": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
            },
            "exhaustive": {
                "sentiment_weight": [i/20.0 for i in range(0, 11)],  # 0.0 to 0.5 in 0.025 steps
                "sentiment_threshold": [i/40.0 for i in range(2, 13)]  # 0.05 to 0.3 in 0.025 steps
            }
        }
        
        base_params = speed_configs.get(self.optimization_speed, speed_configs["medium"])
        
        # Strategy-specific parameters based on speed
        if self.strategy == "Simple Moving Average":
            if self.optimization_speed == "lightning":
                strategy_params = {"sma_window": [15, 20, 25]}
            elif self.optimization_speed == "fast":
                strategy_params = {"sma_window": [10, 20, 30]}
            elif self.optimization_speed == "medium":
                strategy_params = {"sma_window": [10, 15, 20, 25, 30]}
            elif self.optimization_speed == "balanced":
                strategy_params = {"sma_window": list(range(10, 31, 3))}  # [10, 13, 16, 19, 22, 25, 28]
            elif self.optimization_speed == "thorough":
                strategy_params = {"sma_window": list(range(10, 31, 2))}  # [10, 12, 14, ..., 30]
            else:  # exhaustive
                strategy_params = {"sma_window": list(range(8, 33))}  # [8, 9, 10, ..., 32]
                
        elif self.strategy == "RSI":
            if self.optimization_speed == "lightning":
                strategy_params = {
                    "rsi_period": [14, 18],
                    "rsi_overbought": [70, 80],
                    "rsi_oversold": [20, 30]
                }
            elif self.optimization_speed in ["fast", "medium"]:
                strategy_params = {
                    "rsi_period": [10, 14, 18, 22],
                    "rsi_overbought": [65, 70, 75, 80],
                    "rsi_oversold": [20, 25, 30, 35]
                }
            elif self.optimization_speed == "balanced":
                strategy_params = {
                    "rsi_period": [8, 10, 12, 14, 16, 18, 20, 22],
                    "rsi_overbought": [65, 68, 70, 72, 75, 78, 80],
                    "rsi_oversold": [15, 20, 22, 25, 28, 30, 35]
                }
            elif self.optimization_speed == "thorough":
                strategy_params = {
                    "rsi_period": list(range(8, 25, 2)),
                    "rsi_overbought": list(range(65, 81, 2)),
                    "rsi_oversold": list(range(15, 36, 2))
                }
            else:  # exhaustive
                strategy_params = {
                    "rsi_period": list(range(6, 26)),
                    "rsi_overbought": list(range(65, 86)),
                    "rsi_oversold": list(range(10, 36))
                }
                
        elif self.strategy == "Bollinger Bands":
            if self.optimization_speed == "lightning":
                strategy_params = {
                    "bb_period": [15, 20, 25],
                    "bb_dev": [1.5, 2.0, 2.5]
                }
            elif self.optimization_speed in ["fast", "medium"]:
                strategy_params = {
                    "bb_period": [15, 20, 25],
                    "bb_dev": [1.5, 2.0, 2.5]
                }
            elif self.optimization_speed == "balanced":
                strategy_params = {
                    "bb_period": [12, 15, 18, 20, 22, 25, 28],
                    "bb_dev": [1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8]
                }
            elif self.optimization_speed == "thorough":
                strategy_params = {
                    "bb_period": list(range(12, 29, 2)),
                    "bb_dev": [round(x * 0.2, 1) for x in range(6, 16)]  # 1.2 to 3.0 in 0.2 steps
                }
            else:  # exhaustive
                strategy_params = {
                    "bb_period": list(range(10, 31)),
                    "bb_dev": [round(x * 0.1, 1) for x in range(10, 31)]  # 1.0 to 3.0 in 0.1 steps
                }
        else:
            strategy_params = {"sma_window": [20]}  # Default fallback
        
        # Combine base and strategy params
        all_params = {**base_params, **strategy_params}
        
        # Limit combinations if needed
        total_combinations = np.prod([len(v) for v in all_params.values()])
        if total_combinations > self.max_combinations:
            logger.warning(f"Parameter combinations ({total_combinations}) exceed limit ({self.max_combinations}). Reducing parameter ranges.")
            # Reduce parameter ranges
            for key in all_params:
                if len(all_params[key]) > 3:
                    all_params[key] = all_params[key][::2]  # Take every 2nd value
        
        return all_params

    def _split_data_period(self) -> Tuple[str, str, str, str]:
        """Split date range into training and testing periods"""
         # Backend validation to prevent 100% training ratio
        if self.optimization_split_ratio >= 1.0:
            raise ValueError("Training ratio must be less than 100%. At least some data must be reserved for testing.")

        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')
        
        total_days = (end - start).days
        train_days = int(total_days * self.optimization_split_ratio)
        
        train_end = start + timedelta(days=train_days)
        
        return (
            self.start_date,
            train_end.strftime('%Y-%m-%d'), 
            train_end.strftime('%Y-%m-%d'),
            self.end_date
        )

    async def _run_single_backtest(self, symbol: str, start_date: str, end_date: str, 
                                  params: Dict[str, Any], use_sentiment: bool) -> Dict[str, Any]:
        """Run a single backtest with given parameters"""
        try:
            # Download data for the specified period
            # Suppress yfinance progress bar
            data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
            
            if data.empty:
                return {"error": "No data available", "final_value": self.initial_capital}
            
            # Fix: Flatten MultiIndex columns if they exist
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            cerebro = bt.Cerebro()
            cerebro.broker.set_cash(self.initial_capital)
            datafeed = bt.feeds.PandasData(dataname=data)
            cerebro.adddata(datafeed)
            
            # Prepare strategy parameters
            strategy_params = params.copy()
            # CRITICAL FIX: Always set use_sentiment parameter
            strategy_params['use_sentiment'] = use_sentiment
            
            # Handle sentiment data
            if use_sentiment:
                # ensure sentiment_data exists in strategy_params and it's a dict
                strategy_params['sentiment_data'] = strategy_params.get('sentiment_data', {})
                if not isinstance(strategy_params['sentiment_data'], dict):
                    logger.warning(f"Sentiment data is not a dict: {type(strategy_params['sentiment_data'])}")
                    strategy_params['sentiment_data'] = {}
            else:
                # Technical-only
                strategy_params['sentiment_data'] = {}

            # Calculate trading days for the sliced period using NYSE calendar
            market_calendar = mcal.get_calendar('NYSE')
            trading_days = market_calendar.valid_days(
            start_date=start_date,
            end_date=end_date
            )

            strategy_params['trading_days'] = trading_days
            strategy_class, final_params = get_trading_strategy(self.strategy, strategy_params)
            cerebro.addstrategy(strategy_class, **final_params)
            
            result = cerebro.run()
            final_value = cerebro.broker.getvalue()
            
            return {
                "final_value": final_value,
                "return_pct": ((final_value - self.initial_capital) / self.initial_capital) * 100,
                "params": params
            }
            
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            return {"error": str(e), "final_value": self.initial_capital, "return_pct": 0.0, "params": params}

    def _format_parameters_display(self, params: Dict[str, Any]) -> str:
        """Format parameters for user-friendly display"""
        if not params:
            return "N/A"
        
        display_parts = []
        
        # Strategy-specific parameters
        if self.strategy == "Simple Moving Average" and "sma_window" in params:
            display_parts.append(f"SMA Window: {params['sma_window']} days")
        elif self.strategy == "RSI":
            if "rsi_period" in params:
                display_parts.append(f"RSI Period: {params['rsi_period']} days")
            if "rsi_overbought" in params:
                display_parts.append(f"Overbought: {params['rsi_overbought']}")
            if "rsi_oversold" in params:
                display_parts.append(f"Oversold: {params['rsi_oversold']}")
        elif self.strategy == "Bollinger Bands":
            if "bb_period" in params:
                display_parts.append(f"BB Period: {params['bb_period']} days")
            if "bb_dev" in params:
                display_parts.append(f"BB Deviation: {params['bb_dev']}")
        
        # Sentiment parameters
        if "sentiment_weight" in params:
            display_parts.append(f"Sentiment Weight: {params['sentiment_weight']*100:.0f}%")
        if "sentiment_threshold" in params:
            display_parts.append(f"Sentiment Threshold: {params['sentiment_threshold']}")
        
        return " | ".join(display_parts)

    async def get_sentiment_signal(self, symbol: str, date: str = None) -> Dict[str, Any]:
        """LlamaIndex-powered sentiment analysis"""
        logger.info(f"get_sentiment_signal called with use_sentiment={self.use_sentiment}, date={date}")
        
        if not self.use_sentiment:
            logger.info("Sentiment analysis disabled, returning neutral")
            return {
                "sentiment": "neutral", 
                "score": 0.0, 
                "confidence": 0.0,
                "enabled": False,
                "source": "disabled"
            }
        
        if not config.has_openai_key:
            logger.warning("Sentiment analysis requested but OpenAI API key validation failed")
            return {
                "sentiment": "neutral", 
                "score": 0.0, 
                "confidence": 0.0,
                "enabled": False,
                "error": "OpenAI API key not valid or connection failed",
                "source": "error"
            }
        
        try:            
            # Build knowledge base with date-specific context
            if date:
                logger.info(f"Building LlamaIndex knowledge base for {symbol} as of {date} (scope: {self.analysis_scope})")
                await self._llama_engine.build_financial_knowledge_base(
                    symbol=symbol, 
                    scope=self.analysis_scope,
                    days_back=30,
                    analysis_date=date
                )
            else:
                logger.error("analysis_date is required for temporal sentiment analysis")
                raise ValueError("analysis_date parameter is required for temporal sentiment analysis")
            
            # Get sentiment data
            sentiment_data = await self._llama_engine.get_sentiment_data(
                symbol=symbol,
                confidence_threshold=self.sentiment_confidence_threshold
            )
            
            # Add risk analysis if requested
            if self.include_risk_analysis and sentiment_data.get("enabled", False):
                logger.info("Including risk factor analysis")
                risk_analysis = await self._llama_engine.get_risk_analysis(symbol)
                sentiment_data["risk_analysis"] = risk_analysis
            
            # Add catalyst analysis if requested
            if self.include_catalyst_analysis and sentiment_data.get("enabled", False):
                logger.info("Including catalyst analysis")
                catalyst_analysis = await self._llama_engine.get_catalyst_analysis(symbol)
                sentiment_data["catalyst_analysis"] = catalyst_analysis
            
            logger.info(f"LlamaIndex sentiment analysis completed: {sentiment_data.get('sentiment')} (score: {sentiment_data.get('score')})")
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"LlamaIndex sentiment analysis error: {e}")
            return {
                "sentiment": "neutral", 
                "score": 0.0, 
                "confidence": 0.0,
                "enabled": False,
                "error": str(e),
                "source": "error"
            }

    async def simulate(self, symbol: str) -> Dict[str, Any]:
        logger.info(f"Starting simulation for {symbol}")
        logger.info(f"Model parameters: use_sentiment={self.use_sentiment}, strategy={self.strategy}, analysis_scope={self.analysis_scope}")
        
        # Initialize LlamaEngine if sentiment analysis is enabled
        if self.use_sentiment and self._llama_engine is None:
            logger.info("Initializing LlamaEngine for sentiment analysis")
            self._llama_engine = QuantFinLlamaEngine()
       
        # Calculate all trading days using NYSE calendar
        market_calendar = mcal.get_calendar('NYSE')
        trading_days = market_calendar.valid_days(
            start_date=self.start_date,
            end_date=self.end_date
        )

        # Build sentiment data for the trading period
        sentiment_data = await self._build_sentiment_data(symbol, trading_days)

        # Suppress yfinance progress bar
        data = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)
        
        # Fix: Flatten MultiIndex columns if they exist
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        cerebro = bt.Cerebro()
        cerebro.broker.set_cash(self.initial_capital)
        datafeed = bt.feeds.PandasData(dataname=data)
        cerebro.adddata(datafeed)
        
        # Pass this dictionary to the strategy
        strategy_params = self.dict()
        strategy_params['sentiment_data'] = sentiment_data  # Indexed by trading date
        strategy_params['trading_days'] = trading_days

        # Instantiate and run the strategy as usual
        strategy_class, final_params = get_trading_strategy(self.strategy, strategy_params)
        cerebro.addstrategy(strategy_class, **final_params)
        
        result = cerebro.run()
        final_value = cerebro.broker.getvalue()
        
        return {
            "strategy": self.strategy,
            "initial_capital": self.initial_capital,
            "final_portfolio_value": round(final_value, 2),
            "sentiment_analysis": sentiment_data,
            "message": f"Simulation successful ({self.strategy}) using LlamaIndex"
        }

    async def backtest(self, symbol: str) -> Dict[str, Any]:
        """Enhanced backtesting with split-sample optimization and sentiment comparison"""
        logger.info(f"Starting {('optimized' if self.enable_optimization else 'simple')} backtest for {symbol}")
        
        # Initialize LlamaEngine if sentiment analysis is enabled
        if self.use_sentiment and self._llama_engine is None:
            logger.info("Initializing LlamaEngine for sentiment analysis")
            self._llama_engine = QuantFinLlamaEngine()

        # Calculate all trading days using NYSE calendar
        market_calendar = mcal.get_calendar('NYSE')
        trading_days = market_calendar.valid_days(
            start_date=self.start_date,
            end_date=self.end_date
        )

        # Build sentiment data for the trading period
        sentiment_data = await self._build_sentiment_data(symbol, trading_days)

        if not self.enable_optimization:
            # Simple backtest - use current parameters as-is
            logger.info("Running simple backtest with current parameters")
            
            # Suppress yfinance progress bar
            data = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            cerebro = bt.Cerebro()
            cerebro.broker.set_cash(self.initial_capital)
            datafeed = bt.feeds.PandasData(dataname=data)
            cerebro.adddata(datafeed)
            
            strategy_params = self.dict()
            strategy_params['sentiment_data'] = sentiment_data
            strategy_params['trading_days'] = trading_days
            
            strategy_class, final_params = get_trading_strategy(self.strategy, strategy_params)
            cerebro.addstrategy(strategy_class, **final_params)
            
            result = cerebro.run()
            final_value = cerebro.broker.getvalue()
            
            return {
                "methodology": "Simple Backtest",
                "strategy": self.strategy,
                "initial_capital": self.initial_capital,
                "final_portfolio_value": round(final_value, 2),
                "total_return_pct": round(((final_value - self.initial_capital) / self.initial_capital) * 100, 2),
                "sentiment_analysis": sentiment_data,
                "parameters_used": {k: v for k, v in self.dict().items() if k in ['sma_window', 'rsi_period', 'rsi_overbought', 'rsi_oversold', 'bb_period', 'bb_dev', 'sentiment_weight', 'sentiment_threshold']},
                "message": f"Simple backtesting completed ({self.strategy})"
            }
        
        # Enhanced backtest with optimization
        logger.info("Running enhanced backtest with split-sample optimization")
        
        # Step 2: Split data into training and testing periods
        train_start, train_end, test_start, test_end = self._split_data_period()
        logger.info(f"Data split - Training: {train_start} to {train_end}, Testing: {test_start} to {test_end}")
        
        # Step 3: Get parameter ranges for optimization
        param_ranges = self._get_optimization_params()
        param_combinations = list(product(*param_ranges.values()))
        param_keys = list(param_ranges.keys())
        
        logger.info(f"Optimizing {len(param_combinations)} parameter combinations")
        
        optimization_results = {
            "sentiment_version": {"best_params": None, "best_return": -float('inf'), "all_results": []},
            "technical_only": {"best_params": None, "best_return": -float('inf'), "all_results": []}
        }
        
        # Step 4: Training phase optimization
        logger.info("Starting training phase optimization...")
        
        for i, combination in enumerate(param_combinations):
            params = dict(zip(param_keys, combination))
            
            if i % 20 == 0:
                logger.info(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
            
            # Test sentiment version
            if self.compare_sentiment_versions:
                params['sentiment_data'] = sentiment_data  # Add sentiment data
                result = await self._run_single_backtest(symbol, train_start, train_end, params, True)
                if "error" not in result:
                    optimization_results["sentiment_version"]["all_results"].append(result)
                    if result["return_pct"] > optimization_results["sentiment_version"]["best_return"]:
                        optimization_results["sentiment_version"]["best_return"] = result["return_pct"]
                        optimization_results["sentiment_version"]["best_params"] = params.copy()

            # Test technical-only version
            params['sentiment_data'] = {}  # Add empty sentiment data
            result = await self._run_single_backtest(symbol, train_start, train_end, params, False)
            if "error" not in result:
                optimization_results["technical_only"]["all_results"].append(result)
                if result["return_pct"] > optimization_results["technical_only"]["best_return"]:
                    optimization_results["technical_only"]["best_return"] = result["return_pct"]
                    optimization_results["technical_only"]["best_params"] = params.copy()
        
        logger.info("Training phase completed. Starting out-of-sample testing...")
        
        # Step 5: Out-of-sample testing with best parameters
        test_results = {}
        
        if self.compare_sentiment_versions and optimization_results["sentiment_version"]["best_params"]:
            logger.info("Testing best sentiment parameters on out-of-sample data")
            test_results["sentiment_version"] = await self._run_single_backtest(
                symbol, test_start, test_end, 
                optimization_results["sentiment_version"]["best_params"], True
            )
        
        if optimization_results["technical_only"]["best_params"]:
            logger.info("Testing best technical parameters on out-of-sample data")
            test_results["technical_only"] = await self._run_single_backtest(
                symbol, test_start, test_end, 
                optimization_results["technical_only"]["best_params"], False
            )
        
        # Step 6: Format user-friendly parameter displays
        sentiment_params_display = self._format_parameters_display(optimization_results["sentiment_version"]["best_params"])
        technical_params_display = self._format_parameters_display(optimization_results["technical_only"]["best_params"])
        
        # Determine recommendation
        sentiment_better = (
            len(test_results) == 2 and 
            test_results.get("sentiment_version", {}).get("return_pct", 0) > 
            test_results.get("technical_only", {}).get("return_pct", 0)
        )
        
        recommended_approach = "Sentiment-enhanced" if sentiment_better else "Technical-only"
        recommended_params = sentiment_params_display if sentiment_better else technical_params_display
        
        # Step 7: Compile comprehensive results with flattened structure
        return {
            "methodology": "Split-Sample Optimization",
            "strategy": self.strategy,
            "optimization_speed": self.optimization_speed,
            
            # User-friendly summary (flattened for easy display)
            "recommended_approach": recommended_approach,
            "recommended_parameters": recommended_params,
            "confidence_level": f"High (tested {len(param_combinations)} combinations)",
            
            # Performance summary
            "sentiment_training_return": f"{optimization_results['sentiment_version']['best_return']:.2f}%" if optimization_results["sentiment_version"]["best_params"] else "N/A",
            "technical_training_return": f"{optimization_results['technical_only']['best_return']:.2f}%",
            "sentiment_test_return": f"{test_results.get('sentiment_version', {}).get('return_pct', 0):.2f}%" if "sentiment_version" in test_results else "N/A",
            "technical_test_return": f"{test_results.get('technical_only', {}).get('return_pct', 0):.2f}%" if "technical_only" in test_results else "N/A",
            
            # Best parameter displays (flattened)
            "best_sentiment_parameters": sentiment_params_display,
            "best_technical_parameters": technical_params_display,
            
            # Data split info
            "training_period": f"{train_start} to {train_end}",
            "test_period": f"{test_start} to {test_end}",
            "split_ratio": f"{int(self.optimization_split_ratio*100)}% training, {int((1-self.optimization_split_ratio)*100)}% testing",
            
            # Results summary
            "parameters_tested": len(param_combinations),
            "sentiment_adds_value": sentiment_better if len(test_results) == 2 else "Cannot determine",
            
            # Detailed results (for advanced users)
            "detailed_results": {
                "data_split": {
                    "training_period": f"{train_start} to {train_end}",
                    "test_period": f"{test_start} to {test_end}",
                    "split_ratio": f"{int(self.optimization_split_ratio*100)}% training, {int((1-self.optimization_split_ratio)*100)}% testing"
                },
                "optimization_results": {
                    "parameters_tested": len(param_combinations),
                    "optimization_duration": "Completed",
                    "best_parameters": {
                        "sentiment_version": optimization_results["sentiment_version"]["best_params"],
                        "technical_only": optimization_results["technical_only"]["best_params"]
                    },
                    "training_performance": {
                        "sentiment_version": f"{optimization_results['sentiment_version']['best_return']:.2f}%" if optimization_results["sentiment_version"]["best_params"] else "N/A",
                        "technical_only": f"{optimization_results['technical_only']['best_return']:.2f}%"
                    }
                },
                "out_of_sample_results": {
                    "sentiment_version": {
                        "final_value": test_results.get("sentiment_version", {}).get("final_value", "N/A"),
                        "return_pct": f"{test_results.get('sentiment_version', {}).get('return_pct', 0):.2f}%"
                    } if "sentiment_version" in test_results else "N/A",
                    "technical_only": {
                        "final_value": test_results.get("technical_only", {}).get("final_value", "N/A"),
                        "return_pct": f"{test_results.get('technical_only', {}).get('return_pct', 0):.2f}%"
                    } if "technical_only" in test_results else "N/A"
                },
                "sentiment_analysis": sentiment_data
            },
            
            "message": f"Enhanced backtesting completed with {len(param_combinations)} parameter combinations tested"
        }
import backtrader as bt
from typing import Dict, Any, Tuple, Type
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class BaseStrategy(bt.Strategy):
    """Base strategy class with common sentiment and market date handling"""
    
    params = (
        ('sma_window', 20),
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
        ('bb_period', 20),
        ('bb_dev', 2.0),
        ('use_sentiment', False),
        ('sentiment_weight', 0.3),
        ('sentiment_threshold', 0.1),
        ('sentiment_data', None),
        ('symbol', None),
        ('trading_days', None),
    )

    def __init__(self):
        super().__init__()
        self.current_trading_day_index = 0

    def _get_current_market_date(self):
        if self.current_trading_day_index < len(self.params.trading_days):
            return self.params.trading_days[self.current_trading_day_index].strftime('%Y-%m-%d')
        return None

class SMAStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.sma = bt.indicators.SimpleMovingAverage(period=self.params.sma_window)
        
    def next(self):
        current_market_date = self._get_current_market_date()
                
        # sentiment_data: dict indexed by trading date (YYYY-MM-DD) or None
        sentiment_data = self.params.sentiment_data.get(current_market_date, None) or {
            "sentiment": "neutral",
            "score": 0.0,
            "confidence": 0.0,
            "enabled": False
        }
        sentiment_score = sentiment_data.get('score', 0.0)

        if not self.position:
            # Basic SMA signal
            sma_signal = 1 if self.data.close[0] > self.sma[0] else -1
            
            # Apply sentiment if enabled
            if self.params.use_sentiment and abs(sentiment_score) > self.params.sentiment_threshold:
                combined_signal = (1 - self.params.sentiment_weight) * sma_signal + \
                                  self.params.sentiment_weight * sentiment_score
                if combined_signal > 0.1:  # Buy threshold
                    self.buy()
            else:
                # Standard SMA logic
                if sma_signal > 0:
                    self.buy()
        else:
            # Exit logic (simplified)
            if self.data.close[0] < self.sma[0]:
                self.sell()
        
        # Increment trading day index
        self.current_trading_day_index += 1

class RSIStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.rsi = bt.indicators.RelativeStrengthIndex(period=self.params.rsi_period)
        
    def next(self):
        current_market_date = self._get_current_market_date()
                
        # sentiment_data: dict indexed by trading date (YYYY-MM-DD) or None
        sentiment_data = self.params.sentiment_data.get(current_market_date, None) or {
            "sentiment": "neutral",
            "score": 0.0,
            "confidence": 0.0,
            "enabled": False
        }
        sentiment_score = sentiment_data.get('score', 0.0)

        if not self.position:
            # Basic RSI signal
            rsi_oversold = self.rsi[0] < self.params.rsi_oversold
            
            # Apply sentiment if enabled
            if self.params.use_sentiment and abs(sentiment_score) > self.params.sentiment_threshold:
                # Sentiment can override RSI or strengthen signal (keep current logic)
                if sentiment_score > 0.3:  # Strong bullish sentiment
                    self.buy()
                elif rsi_oversold and sentiment_score > -0.3:  # Not bearish
                    self.buy()
            else:
                # Standard RSI logic
                if rsi_oversold:
                    self.buy()
        else:
            # Exit logic
            if self.rsi[0] > self.params.rsi_overbought:
                self.sell()
        
        # Increment trading day index
        self.current_trading_day_index += 1

class BollingerBandsStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.bb = bt.indicators.BollingerBands(period=self.params.bb_period, devfactor=self.params.bb_dev)
        
    def next(self):
        current_market_date = self._get_current_market_date()
                
        # sentiment_data: dict indexed by trading date (YYYY-MM-DD) or None
        sentiment_data = self.params.sentiment_data.get(current_market_date, None) or {
            "sentiment": "neutral",
            "score": 0.0,
            "confidence": 0.0,
            "enabled": False
        }
        sentiment_score = sentiment_data.get('score', 0.0)

        if not self.position:
            # Basic Bollinger Bands signal
            near_lower_band = self.data.close[0] <= self.bb.lines.bot[0] * 1.02
            
            # Apply sentiment if enabled
            if self.params.use_sentiment and abs(sentiment_score) > self.params.sentiment_threshold:
                # Sentiment can trigger entries even without touching bands (keep current logic)
                if sentiment_score > 0.5:  # Very bullish
                    self.buy()
                elif near_lower_band and sentiment_score > -0.3:  # Not bearish
                    self.buy()
            else:
                # Standard BB logic
                if near_lower_band:
                    self.buy()
        else:
            # Exit logic
            if self.data.close[0] >= self.bb.lines.top[0] * 0.98:
                self.sell()
        
        # Increment trading day index
        self.current_trading_day_index += 1

def get_trading_strategy(strategy_name: str, params: Dict[str, Any]) -> Tuple[Type[bt.Strategy], Dict[str, Any]]:
    """
    Returns the appropriate trading strategy class and cleaned parameters.
    
    Args:
        strategy_name: Name of the strategy to use
        params: Dictionary of parameters for the strategy
        
    Returns:
        Tuple of (Strategy class, cleaned parameters dictionary)
    """
    
    # Define strategy mapping and their expected parameters
    strategy_configs = {
        "Simple Moving Average": {
            "class": SMAStrategy,
            "params": ['sma_window', 'use_sentiment', 'sentiment_weight', 'sentiment_threshold', 'sentiment_data', 'symbol', 'trading_days']
        },
        "RSI": {
            "class": RSIStrategy, 
            "params": ['rsi_period', 'rsi_overbought', 'rsi_oversold', 'use_sentiment', 'sentiment_weight', 'sentiment_threshold', 'sentiment_data', 'symbol', 'trading_days']
        },
        "Bollinger Bands": {
            "class": BollingerBandsStrategy,
            "params": ['bb_period', 'bb_dev', 'use_sentiment', 'sentiment_weight', 'sentiment_threshold', 'sentiment_data', 'symbol', 'trading_days']
        },
    }
    
    # Get strategy configuration
    strategy_config = strategy_configs.get(strategy_name)
    if not strategy_config:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    strategy_class = strategy_config["class"]
    expected_params = strategy_config["params"]
    
    # Clean parameters - remove non-strategy parameters and filter by expected params
    excluded_params = {
        'initial_capital', 'strategy',
        # Exclude LlamaIndex-specific parameters that don't go to BackTrader
        'analysis_scope', 'include_risk_analysis', 'include_catalyst_analysis', 
        'sentiment_confidence_threshold'
    }
    
    strategy_params = {
        k: v for k, v in params.items() 
        if k not in excluded_params and k in expected_params
    }
    
    return strategy_class, strategy_params 
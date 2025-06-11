import backtrader as bt
from typing import Dict, Any, Tuple, Type
import logging

class SMAStrategy(bt.Strategy):
    params = (
        ('sma_window', 20),
        ('use_sentiment', False),
        ('sentiment_weight', 0.3),
        ('sentiment_threshold', 0.1),
        ('sentiment_data', None),
    )

    def __init__(self):
        logger = logging.getLogger(__name__)
        
        logger.info(f"SMAStrategy.__init__ called")
        logger.info(f"params.sentiment_data: {self.params.sentiment_data}")
        logger.info(f"params.use_sentiment: {self.params.use_sentiment}")
        
        self.sma = bt.indicators.SimpleMovingAverage(period=self.params.sma_window)
        
        # Initialize sentiment signal if available
        self.sentiment_signal = 0.0
        if self.params.sentiment_data and self.params.use_sentiment:
            self.sentiment_signal = self.params.sentiment_data.get('score', 0.0)
        
        logger.info(f"Set sentiment_signal: {self.sentiment_signal}")

    def next(self):
        if not self.position:
            # Basic SMA signal
            sma_signal = 1 if self.data.close[0] > self.sma[0] else -1
            
            # Apply sentiment if enabled
            if self.params.use_sentiment and abs(self.sentiment_signal) > self.params.sentiment_threshold:
                # Combine SMA signal with sentiment
                combined_signal = (1 - self.params.sentiment_weight) * sma_signal + \
                                  self.params.sentiment_weight * self.sentiment_signal
                
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

class RSIStrategy(bt.Strategy):
    params = (
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
        ('use_sentiment', False),
        ('sentiment_weight', 0.3),
        ('sentiment_threshold', 0.1),
        ('sentiment_data', None),
    )

    def __init__(self):
        self.rsi = bt.indicators.RelativeStrengthIndex(period=self.params.rsi_period)
        
        # Initialize sentiment signal if available
        self.sentiment_signal = 0.0
        if self.params.sentiment_data and self.params.use_sentiment:
            self.sentiment_signal = self.params.sentiment_data.get('score', 0.0)

    def next(self):
        if not self.position:
            # Basic RSI signal
            rsi_oversold = self.rsi[0] < self.params.rsi_oversold
            
            # Apply sentiment if enabled
            if self.params.use_sentiment and abs(self.sentiment_signal) > self.params.sentiment_threshold:
                # Sentiment can override RSI or strengthen signal
                if self.sentiment_signal > 0.3:  # Strong bullish sentiment
                    self.buy()
                elif rsi_oversold and self.sentiment_signal > -0.3:  # Not bearish
                    self.buy()
            else:
                # Standard RSI logic
                if rsi_oversold:
                    self.buy()
        else:
            # Exit logic
            if self.rsi[0] > self.params.rsi_overbought:
                self.sell()

class BollingerBandsStrategy(bt.Strategy):
    params = (
        ('bb_period', 20),
        ('bb_dev', 2.0),
        ('use_sentiment', False),
        ('sentiment_weight', 0.3),
        ('sentiment_threshold', 0.1),
        ('sentiment_data', None),
    )

    def __init__(self):
        self.bb = bt.indicators.BollingerBands(period=self.params.bb_period, devfactor=self.params.bb_dev)
        
        # Initialize sentiment signal if available
        self.sentiment_signal = 0.0
        if self.params.sentiment_data and self.params.use_sentiment:
            self.sentiment_signal = self.params.sentiment_data.get('score', 0.0)

    def next(self):
        if not self.position:
            # Basic Bollinger Bands signal
            near_lower_band = self.data.close[0] <= self.bb.lines.bot[0] * 1.02
            
            # Apply sentiment if enabled
            if self.params.use_sentiment and abs(self.sentiment_signal) > self.params.sentiment_threshold:
                # Sentiment can trigger entries even without touching bands
                if self.sentiment_signal > 0.5:  # Very bullish
                    self.buy()
                elif near_lower_band and self.sentiment_signal > -0.3:  # Not bearish
                    self.buy()
            else:
                # Standard Bollinger Bands logic
                if near_lower_band:
                    self.buy()
        else:
            # Exit logic
            if self.data.close[0] >= self.bb.lines.top[0] * 0.98:
                self.sell()

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
            "params": ['sma_window', 'use_sentiment', 'sentiment_weight', 'sentiment_threshold', 'sentiment_data']
        },
        "RSI": {
            "class": RSIStrategy, 
            "params": ['rsi_period', 'rsi_overbought', 'rsi_oversold', 'use_sentiment', 'sentiment_weight', 'sentiment_threshold', 'sentiment_data']
        },
        "Bollinger Bands": {
            "class": BollingerBandsStrategy,
            "params": ['bb_period', 'bb_dev', 'use_sentiment', 'sentiment_weight', 'sentiment_threshold', 'sentiment_data']
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
        'initial_capital', 'strategy', 'start_date', 'end_date',
        # Exclude LlamaIndex-specific parameters that don't go to BackTrader
        'analysis_scope', 'include_risk_analysis', 'include_catalyst_analysis', 
        'sentiment_confidence_threshold'
    }
    
    strategy_params = {
        k: v for k, v in params.items() 
        if k not in excluded_params and k in expected_params
    }
    
    return strategy_class, strategy_params 
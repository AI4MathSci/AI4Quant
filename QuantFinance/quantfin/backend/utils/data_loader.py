import pandas as pd
import yfinance as yf
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

def load_historical_data(symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Loads historical stock data from Yahoo Finance.

    Args:
        symbol (str): The stock symbol (e.g., "AAPL").
        start_date (Optional[str], optional): The start date for the data (YYYY-MM-DD). Defaults to None.
        end_date (Optional[str], optional): The end date for the data (YYYY-MM-DD). Defaults to None.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary containing the historical data,
                                  or an empty dictionary if an error occurs.  The key is the symbol.
    """
    try:
        logger.info(f"Loading historical data for {symbol} from {start_date} to {end_date}")
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            logger.warning(f"No data found for {symbol} in the specified date range.")
            return {}  # Return an empty dictionary
        return {symbol: data}  # Return data as a dictionary with symbol as key.
    except Exception as e:
        logger.error(f"Error loading historical data for {symbol}: {e}")
        return {}  # Return an empty dictionary in case of error
    # Note:  Consider adding error handling to raise exceptions, or return a specific error object.
    #        The current implementation returns an empty dict on error, which might not be the best approach
    #        for all use cases.
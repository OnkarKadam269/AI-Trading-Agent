import pandas as pd
from typing import Dict, Optional
from .mt5_connector import mt5_conn
import logging

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self):
        self.cache: Dict[str, Dict[int, pd.DataFrame]] = {}
        
    def fetch_and_cache(self, symbol: str, timeframes: list, count: int = 1000):
        """Fetches data for multiple timeframes and caches them."""
        if symbol not in self.cache:
            self.cache[symbol] = {}
            
        for tf in timeframes:
            df = mt5_conn.get_historical_data(symbol, tf, count)
            if df is not None:
                self.cache[symbol][tf] = df
            else:
                logger.warning(f"Failed to fetch data for {symbol} at TF {tf}")

    def get_data(self, symbol: str, timeframe: int) -> Optional[pd.DataFrame]:
        return self.cache.get(symbol, {}).get(timeframe)

data_manager = DataManager()

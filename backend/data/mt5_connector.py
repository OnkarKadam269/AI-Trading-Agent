import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import logging
from typing import Optional
from core.config import settings

logger = logging.getLogger(__name__)

class MT5Connector:
    def __init__(self):
        self.connected = False

    def initialize(self) -> bool:
        """Initializes the MT5 terminal and logs in."""
        if not mt5.initialize(path=settings.MT5_PATH):
            logger.error(f"MT5 initialization failed. Error: {mt5.last_error()}")
            return False
        
        # If account details are provided, try logging in
        if settings.MT5_ACCOUNT and settings.MT5_PASSWORD and settings.MT5_SERVER:
            authorized = mt5.login(
                settings.MT5_ACCOUNT, 
                password=settings.MT5_PASSWORD, 
                server=settings.MT5_SERVER
            )
            if not authorized:
                logger.error(f"MT5 login failed. Error: {mt5.last_error()}")
                return False
            logger.info(f"Connected to MT5 Account: {settings.MT5_ACCOUNT}")
        else:
            logger.warning("MT5 started but no account credentials provided in .env.")
            
        self.connected = True
        return True

    def get_historical_data(self, symbol: str, timeframe: int, count: int) -> Optional[pd.DataFrame]:
        """Fetches historical OHLCV data."""
        if not self.connected:
            logger.error("MT5 not connected.")
            return None
            
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            logger.error(f"Failed to fetch data for {symbol}. Error: {mt5.last_error()}")
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df

    def get_current_tick(self, symbol: str):
        """Fetches the latest tick data."""
        if not self.connected:
            return None
        tick = mt5.symbol_info_tick(symbol)
        return tick

    def shutdown(self):
        """Shutdown MT5 connection."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 connection closed.")

# Global instance
mt5_conn = MT5Connector()

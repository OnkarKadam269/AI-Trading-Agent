from typing import Dict, Any
from core.config import settings
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Handles global risk checks before any trade is executed.
    """

    @staticmethod
    def check_risk(account_equity: float, current_drawdown: float, active_trades_count: int, signal: Dict[str, Any]) -> bool:
        """
        Runs safety checks before allowing execution.
        """
        logger.info(f"Running risk checks for {signal.get('symbol')}...")
        
        if active_trades_count >= settings.MAX_CONCURRENT_TRADES:
            logger.warning(f"Risk Check Failed: Max concurrent trades ({settings.MAX_CONCURRENT_TRADES}) reached.")
            return False
            
        if current_drawdown >= settings.MAX_DAILY_DRAWDOWN_PCT:
            logger.warning(f"Risk Check Failed: Max daily drawdown ({settings.MAX_DAILY_DRAWDOWN_PCT}%) exceeded.")
            return False
            
        # Simulated spread and margin checks
        logger.info("All risk checks passed.")
        return True

risk_manager = RiskManager()

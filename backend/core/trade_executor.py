import MetaTrader5 as mt5
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class TradeExecutor:
    """
    Executes trades on MetaTrader 5 based on validated AI signals.
    """

    @staticmethod
    def execute_order(symbol: str, order_type: int, lot_size: float, price: float, sl: float, tp: float) -> Dict[str, Any]:
        """
        Sends an order to MT5.
        """
        logger.info(f"Preparing to execute {order_type} on {symbol}...")
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot_size),
            "type": order_type,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 20,
            "magic": 123456,
            "comment": "AI Agent Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Real MT5 Execution
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed. Retcode: {result.retcode}")
            return {"status": "failed", "retcode": result.retcode}
            
        logger.info(f"Live execution successful: Ticket #{result.order}")
        return {"status": "success", "ticket": result.order, "request": request}

trade_executor = TradeExecutor()

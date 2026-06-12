import pandas as pd
import ta
import logging
from core.smc_analysis import SMCAnalyzer

logger = logging.getLogger(__name__)

class DynamicStrategyRouter:
    @staticmethod
    def get_setup(df: pd.DataFrame, regime: str) -> dict:
        """
        Routes the market data to the appropriate strategy based on the regime.
        If a mathematical setup is found, queries Kronos AI for validation.
        """
        if df.empty or len(df) < 50:
            return None
            
        setup = None
        if regime == "TRENDING" or regime == "HIGH_VOLATILITY":
            setup = TrendingStrategy.analyze(df)
        elif regime == "RANGING":
            setup = RangingStrategy.analyze(df)
        elif regime == "CHOPPY" or regime == "LOW_VOLATILITY":
            setup = ScalpingStrategy.analyze(df)
            
        if setup:
            from core.kronos_client import kronos_client
            logger.info(f"Quant setup found ({setup['type']}). Requesting Kronos Deep Learning Forecast...")
            
            prediction = kronos_client.predict(df)
            if prediction:
                setup['kronos_forecast'] = prediction
                
                # Validation: Kronos must agree with the Quant Signal (>60% confidence)
                if setup['signal'] == 'BUY' and prediction.get('probability_up', 0) < 60:
                    logger.info("Kronos rejected BUY setup. Probability too low.")
                    return None
                if setup['signal'] == 'SELL' and prediction.get('probability_down', 0) < 60:
                    logger.info("Kronos rejected SELL setup. Probability too low.")
                    return None
                
                logger.info(f"Kronos confirmed setup with {max(prediction['probability_up'], prediction['probability_down'])}% probability.")
        
        return setup


class TrendingStrategy:
    @staticmethod
    def analyze(df: pd.DataFrame) -> dict:
        """
        Looks for SMC structures (FVG/OB) in a trending market.
        """
        smc_data = SMCAnalyzer.analyze(df)
        fvgs = smc_data.get('fvgs', [])
        
        if len(fvgs) > 0:
            current_fvg = fvgs[-1]
            entry = df.iloc[-1]['close']
            atr = df['high'].rolling(14).max().iloc[-1] - df['low'].rolling(14).min().iloc[-1]
            
            return {
                "type": "Momentum/FVG Pullback",
                "signal": "BUY" if entry > current_fvg['top'] else "SELL", # Simplified logic
                "entry": entry,
                "sl": entry - (atr * 0.5) if entry > current_fvg['top'] else entry + (atr * 0.5),
                "tp": entry + (atr * 1.5) if entry > current_fvg['top'] else entry - (atr * 1.5),
                "rr": "1:3"
            }
        return None


class RangingStrategy:
    @staticmethod
    def analyze(df: pd.DataFrame) -> dict:
        """
        Mean Reversion strategy using Bollinger Bands and RSI.
        """
        # Calculate BB
        bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        bb_upper = bb_indicator.bollinger_hband().iloc[-1]
        bb_lower = bb_indicator.bollinger_lband().iloc[-1]
        
        # Calculate RSI
        rsi_indicator = ta.momentum.RSIIndicator(df['close'], window=14)
        rsi = rsi_indicator.rsi().iloc[-1]
        
        entry = df.iloc[-1]['close']
        
        if entry >= bb_upper and rsi > 70:
            # Overbought condition at upper band
            return {
                "type": "Mean Reversion (Ranging)",
                "signal": "SELL",
                "entry": entry,
                "sl": entry + (entry * 0.002), # 0.2% stop
                "tp": bb_indicator.bollinger_mavg().iloc[-1], # Target the middle band
                "rr": "Dynamic"
            }
        elif entry <= bb_lower and rsi < 30:
            # Oversold condition at lower band
            return {
                "type": "Mean Reversion (Ranging)",
                "signal": "BUY",
                "entry": entry,
                "sl": entry - (entry * 0.002),
                "tp": bb_indicator.bollinger_mavg().iloc[-1],
                "rr": "Dynamic"
            }
            
        return None


class ScalpingStrategy:
    @staticmethod
    def analyze(df: pd.DataFrame) -> dict:
        """
        Looks for micro-breakouts in low volatility conditions.
        """
        # Calculate recent tight range
        recent_high = df['high'].tail(5).max()
        recent_low = df['low'].tail(5).min()
        entry = df.iloc[-1]['close']
        
        # If price just broke the tight 5-candle range
        if entry > recent_high:
            return {
                "type": "Micro Breakout (Scalping)",
                "signal": "BUY",
                "entry": entry,
                "sl": recent_low,
                "tp": entry + (entry - recent_low), # 1:1 RR for scalping
                "rr": "1:1"
            }
        elif entry < recent_low:
            return {
                "type": "Micro Breakout (Scalping)",
                "signal": "SELL",
                "entry": entry,
                "sl": recent_high,
                "tp": entry - (recent_high - entry),
                "rr": "1:1"
            }
            
        return None

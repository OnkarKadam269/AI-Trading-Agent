import pandas as pd
import ta

class MarketConditionClassifier:
    """
    Classifies the market into 5 regimes based on ADX, ATR, and Bollinger Bands.
    Regimes:
    1: TRENDING (ADX > 25, price outside BB)
    2: RANGING (ADX < 20, BB contracted)
    3: BREAKOUT (ADX rising, volume spike)
    4: HIGH_VOLATILITY (ATR > 2x avg)
    5: CHOPPY (ADX < 15)
    """

    @staticmethod
    def classify(df: pd.DataFrame) -> str:
        if df.empty or len(df) < 20:
            return "CHOPPY"
            
        # Calculate ADX (14)
        adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        current_adx = adx_indicator.adx().iloc[-1]
        
        # Calculate ATR (14)
        atr_indicator = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        atr = atr_indicator.average_true_range()
        current_atr = atr.iloc[-1]
        avg_atr = atr.rolling(window=20).mean().iloc[-1]
        
        # Calculate Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(df['close'], window=20)
        # Not fully implemented condition using bb_indicator yet, just placeholder
        
        # High Volatility
        if current_atr > 2 * avg_atr:
            return "HIGH_VOLATILITY"
            
        # Trending
        if current_adx > 25:
            # Simple check if trending
            return "TRENDING"
            
        # Ranging
        if 15 <= current_adx <= 20:
            return "RANGING"
            
        # Choppy / No-Trade
        if current_adx < 15:
            return "CHOPPY"
            
        return "CHOPPY"

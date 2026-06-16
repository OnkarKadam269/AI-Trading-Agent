import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import ta
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - KRONOS TRAINER - %(message)s')
logger = logging.getLogger(__name__)

PAIRS = ["GBPUSD", "XAUUSD", "USDJPY", "AUDUSD"]
TIMEFRAME = mt5.TIMEFRAME_M5
CANDLES = 50000 # Download last 10,000 candles for training

def get_real_symbol(base_symbol):
    """Finds the actual symbol name in Exness (handles suffixes like EURUSDm)."""
    symbols = mt5.symbols_get()
    if symbols is None: return base_symbol
    for s in symbols:
        if base_symbol in s.name:
            return s.name
    return base_symbol

def get_data(symbol):
    real_symbol = get_real_symbol(symbol)
    logger.info(f"Detected real symbol: {real_symbol}")
    rates = mt5.copy_rates_from_pos(real_symbol, TIMEFRAME, 0, CANDLES)
    if rates is None or len(rates) == 0:
        logger.error(f"Failed to fetch data for {real_symbol}")
        return pd.DataFrame()
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def add_features(df):
    """Adds technical indicators as Machine Learning features."""
    if df.empty: return df
    
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_width'] = df['bb_high'] - df['bb_low']
    # Moving Averages
    df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
    df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
    
    # Price distance from SMA
    df['dist_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    
    df.dropna(inplace=True)
    return df

def label_data(df):
    """Labels the data: 1 if price goes UP in the next 5 candles, 0 if DOWN."""
    if df.empty: return df
    # Look ahead 5 candles (approx 1 hour on M15)
    df['future_close'] = df['close'].shift(-5)
    df.dropna(inplace=True)
    
    # Target: 1 if future close > current close, else 0
    df['target'] = (df['future_close'] > df['close']).astype(int)
    return df

def train_model():
    logger.info("Initializing MT5...")
    if not mt5.initialize():
        logger.error("MT5 initialization failed!")
        return
        
    all_data = pd.DataFrame()
    
    for symbol in PAIRS:
        logger.info(f"Processing {symbol}...")
        df = get_data(symbol)
        if df.empty:
            continue
        df = add_features(df)
        df = label_data(df)
        
        if not df.empty:
            all_data = pd.concat([all_data, df])
        
    mt5.shutdown()
    
    logger.info(f"Total training rows collected: {len(all_data)}")
    
    # Define features
    features = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'bb_width', 'dist_sma20']
    X = all_data[features]
    y = all_data['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    logger.info("Training Kronos AI (XGBoost Classifier)... This may take a minute.")
    model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.05, 
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # Test accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    logger.info(f"Training Complete! Model Accuracy: {accuracy * 100:.2f}%")
    
    # Save the model
    with open('kronos_model_m5.pkl', 'wb') as f:
        pickle.dump(model, f)
    logger.info("Model saved successfully as 'kronos_model_m5.pkl'. Ready for RunPod!")

if __name__ == "__main__":
    train_model()

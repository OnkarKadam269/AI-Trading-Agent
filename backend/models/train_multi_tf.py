import pandas as pd
import numpy as np
import ta
import pickle
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - MULTI-TF TRAINER - %(message)s')
logger = logging.getLogger(__name__)

PAIRS = ['GBPUSDm', 'USDJPYm', 'XAUUSDm']
TIMEFRAMES = ['M5', 'M15', 'M30', 'H1', 'H4']

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
    
    # Price distance from SMA
    df['dist_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    
    df.dropna(inplace=True)
    return df

def label_data(df):
    """Labels the data: 1 if price goes UP in the next 5 candles, 0 if DOWN."""
    if df.empty: return df
    # Look ahead 5 candles
    df['future_close'] = df['close'].shift(-5)
    df.dropna(inplace=True)
    
    # Target: 1 if future close > current close, else 0
    df['target'] = (df['future_close'] > df['close']).astype(int)
    return df

def train_models():
    raw_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw')
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    for tf in TIMEFRAMES:
        logger.info(f"==== TRAINING MODEL FOR TIMEFRAME: {tf} ====")
        all_data = pd.DataFrame()
        
        for pair in PAIRS:
            filepath = os.path.join(raw_dir, f"{pair}_{tf}_7M.csv")
            if not os.path.exists(filepath):
                logger.error(f"Missing data file: {filepath}")
                continue
                
            df = pd.read_csv(filepath)
            if df.empty:
                continue
            
            # Ensure time is index (though not strictly necessary for XGBoost if we drop it)
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
            
            df = add_features(df)
            df = label_data(df)
            
            if not df.empty:
                all_data = pd.concat([all_data, df])
                
        if all_data.empty:
            logger.warning(f"No valid data found for timeframe {tf}. Skipping.")
            continue
            
        logger.info(f"Total training rows collected for {tf}: {len(all_data)}")
        
        # Define features
        features = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'bb_width', 'dist_sma20']
        X = all_data[features]
        y = all_data['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        
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
        logger.info(f"[{tf}] Training Complete! Model Accuracy: {accuracy * 100:.2f}%")
        
        model_path = os.path.join(models_dir, f'kronos_model_{tf}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved specialized model to {model_path}\n")

if __name__ == "__main__":
    train_models()

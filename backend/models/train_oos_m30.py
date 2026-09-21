import pandas as pd
import numpy as np
import ta
import pickle
import logging
import os
from sklearn.metrics import accuracy_score
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(message)s')

PAIRS = ['GBPUSDm', 'USDJPYm', 'XAUUSDm']
CUTOFF_DATE = '2026-05-01'

def add_features(df):
    if df.empty: return df
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_width'] = df['bb_high'] - df['bb_low']
    df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
    df['dist_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    df.dropna(inplace=True)
    return df

def label_data(df):
    if df.empty: return df
    df['future_close'] = df['close'].shift(-5)
    df.dropna(inplace=True)
    df['target'] = (df['future_close'] > df['close']).astype(int)
    return df

def train_oos_model():
    raw_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw')
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    all_train = pd.DataFrame()
    all_test = pd.DataFrame()
    
    for pair in PAIRS:
        filepath = os.path.join(raw_dir, f"{pair}_M30_7M.csv")
        if not os.path.exists(filepath): continue
            
        df = pd.read_csv(filepath)
        df['time'] = pd.to_datetime(df['time'])
        df = add_features(df)
        df = label_data(df)
        
        # Split chronologically
        train_df = df[df['time'] < CUTOFF_DATE]
        test_df = df[df['time'] >= CUTOFF_DATE]
        
        all_train = pd.concat([all_train, train_df])
        all_test = pd.concat([all_test, test_df])
        
    if all_train.empty:
        print("No training data found.")
        return
        
    print(f"Training on Nov 2025 - Apr 2026 (Rows: {len(all_train)})")
    print(f"Testing on May 2026 - Jun 2026 (Rows: {len(all_test)})")
    
    features = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'bb_width', 'dist_sma20']
    
    X_train = all_train[features]
    y_train = all_train['target']
    
    X_test = all_test[features]
    y_test = all_test['target']
    
    print("Training OOS XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.05, 
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Out-Of-Sample Accuracy: {acc*100:.2f}%")
    
    model_path = os.path.join(models_dir, 'kronos_model_M30_OOS.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved true OOS model to {model_path}")

if __name__ == "__main__":
    train_oos_model()

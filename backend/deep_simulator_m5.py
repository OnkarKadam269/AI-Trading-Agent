import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pickle
import ta
import logging
import warnings
import json
import os
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - MFE/MAE SIMULATOR - %(message)s')

PAIRS = ["GBPUSD", "USDJPY", "XAUUSD", "AUDUSD"]
TIMEFRAME_M5 = mt5.TIMEFRAME_M5
TIMEFRAME_M1 = mt5.TIMEFRAME_M1
CANDLES_M5 = 75000  # Approx 1 year of data

SL_PCT = 0.002
TP_PCT = 0.003

RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'deep_backtest_results_m5.json')

def get_real_symbol(base_symbol):
    symbols = mt5.symbols_get()
    if symbols is None: return base_symbol
    for s in symbols:
        if base_symbol in s.name:
            return s.name
    return base_symbol

def get_m15_data(symbol):
    real_symbol = get_real_symbol(symbol)
    rates = mt5.copy_rates_from_pos(real_symbol, TIMEFRAME_M5, 0, CANDLES_M5)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def get_m1_data_forward(symbol, start_time, max_candles=1440): # max 24 hours
    real_symbol = get_real_symbol(symbol)
    # Convert start_time to UTC timestamp
    start_ts = int(start_time.timestamp())
    rates = mt5.copy_rates_from(real_symbol, TIMEFRAME_M1, start_ts, max_candles)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def add_features(df):
    if df.empty: return df
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_width'] = bb.bollinger_wband()
    df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
    df['dist_sma20'] = (df['close'] - df['sma_20']) / df['sma_20'] * 100
    df.dropna(inplace=True)
    return df

def run_deep_simulation():
    logging.info("Initializing MT5 Institutional Simulation Engine...")
    if not mt5.initialize():
        logging.error("MT5 initialization failed!")
        return
        
    try:
        model_path = os.path.join(os.path.dirname(__file__), "kronos_model_m5.pkl")
        with open(model_path, "rb") as f:
            xgb_model = pickle.load(f)
    except FileNotFoundError:
        logging.error("Model not found! Please run train_kronos.py first.")
        return

    all_simulated_trades = []

    for symbol in PAIRS:
        logging.info(f"Downloading M15 master data for {symbol}...")
        df = get_m15_data(symbol)
        if df.empty: continue
            
        df = add_features(df)
        if df.empty: continue
        
        features = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'bb_width', 'dist_sma20']
        X = df[features]
        predictions = xgb_model.predict(X)
        probabilities = xgb_model.predict_proba(X)[:, 1]
        
        df['prediction'] = predictions
        df['probability'] = probabilities
        
        # Filter for actual trade triggers
        buy_signals = df[df['probability'] > 0.60]
        sell_signals = df[df['probability'] < 0.40]
        
        trade_signals = pd.concat([buy_signals, sell_signals]).sort_index()
        logging.info(f"Found {len(trade_signals)} historical trade triggers for {symbol}. Dropping into M1 simulation...")
        
        for index, row in trade_signals.iterrows():
            entry_price = float(row['close'])
            entry_time = index
            probability = float(row['probability'])
            direction = "BUY" if probability > 0.60 else "SELL"
            
            # SL and TP
            if direction == "BUY":
                sl = entry_price * (1 - SL_PCT)
                tp = entry_price * (1 + TP_PCT)
            else:
                sl = entry_price * (1 + SL_PCT)
                tp = entry_price * (1 - TP_PCT)
                
            # Drop down to M1 to track tick-by-tick
            m1_df = get_m1_data_forward(symbol, entry_time)
            if m1_df.empty: continue
                
            mfe_price = entry_price
            mae_price = entry_price
            exit_price = None
            exit_time = None
            result = ""
            
            for _, m1_row in m1_df.iterrows():
                high = m1_row['high']
                low = m1_row['low']
                
                # Track Max Favorable / Adverse Excursions
                if direction == "BUY":
                    if high > mfe_price: mfe_price = high
                    if low < mae_price: mae_price = low
                    
                    if low <= sl:
                        exit_price = sl
                        exit_time = m1_row['time']
                        result = "LOSS"
                        break
                    if high >= tp:
                        exit_price = tp
                        exit_time = m1_row['time']
                        result = "WIN"
                        break
                else: # SELL
                    if low < mfe_price: mfe_price = low  # for sell, lower is better
                    if high > mae_price: mae_price = high # for sell, higher is worse
                    
                    if high >= sl:
                        exit_price = sl
                        exit_time = m1_row['time']
                        result = "LOSS"
                        break
                    if low <= tp:
                        exit_price = tp
                        exit_time = m1_row['time']
                        result = "WIN"
                        break
            
            if exit_price is None:
                # Timed out (didn't hit TP/SL in 24 hours)
                continue
                
            # Calculate MFE/MAE in Pips/Points absolute difference
            if direction == "BUY":
                mfe_diff = (mfe_price - entry_price) / entry_price * 100
                mae_diff = (entry_price - mae_price) / entry_price * 100
            else:
                mfe_diff = (entry_price - mfe_price) / entry_price * 100
                mae_diff = (mae_price - entry_price) / entry_price * 100
                
            all_simulated_trades.append({
                "pair": symbol,
                "direction": direction,
                "entry_time": str(entry_time),
                "exit_time": str(exit_time),
                "duration_minutes": (exit_time - entry_time).total_seconds() / 60.0,
                "probability": probability,
                "result": result,
                "mfe_pct": float(mfe_diff),  # Max profit % before reversing
                "mae_pct": float(mae_diff),  # Max drawdown % before winning
                "rsi": float(row['rsi']),
                "bb_width": float(row['bb_width']),
                "dist_sma20": float(row['dist_sma20'])
            })

    mt5.shutdown()
    
    logging.info(f"Simulation Complete! Processed {len(all_simulated_trades)} deep trades with MFE/MAE.")
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_simulated_trades, f, indent=4)
        
    logging.info(f"Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    run_deep_simulation()

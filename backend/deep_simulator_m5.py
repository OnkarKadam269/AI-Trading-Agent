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
RRR_TARGETS = {
    "1:1": 0.002,
    "1:1.5": 0.003,
    "1:2": 0.004,
    "1:3": 0.006
}

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
            
            # SL is constant
            if direction == "BUY":
                sl = entry_price * (1 - SL_PCT)
            else:
                sl = entry_price * (1 + SL_PCT)
                
            targets = {}
            for rrr, tp_pct in RRR_TARGETS.items():
                if direction == "BUY":
                    tp = entry_price * (1 + tp_pct)
                else:
                    tp = entry_price * (1 - tp_pct)
                targets[rrr] = {
                    "tp": tp,
                    "result": "",
                    "exit_time": None,
                    "active": True
                }
                
            # Drop down to M1 to track tick-by-tick
            m1_df = get_m1_data_forward(symbol, entry_time)
            if m1_df.empty: continue
                
            mfe_price = entry_price
            mae_price = entry_price
            
            for _, m1_row in m1_df.iterrows():
                high = m1_row['high']
                low = m1_row['low']
                
                if direction == "BUY":
                    if high > mfe_price: mfe_price = high
                    if low < mae_price: mae_price = low
                    
                    for rrr, target in targets.items():
                        if not target["active"]: continue
                        if low <= sl:
                            target["exit_time"] = str(m1_row['time'])
                            target["result"] = "LOSS"
                            target["active"] = False
                        elif high >= target["tp"]:
                            target["exit_time"] = str(m1_row['time'])
                            target["result"] = "WIN"
                            target["active"] = False
                else: # SELL
                    if low < mfe_price: mfe_price = low
                    if high > mae_price: mae_price = high
                    
                    for rrr, target in targets.items():
                        if not target["active"]: continue
                        if high >= sl:
                            target["exit_time"] = str(m1_row['time'])
                            target["result"] = "LOSS"
                            target["active"] = False
                        elif low <= target["tp"]:
                            target["exit_time"] = str(m1_row['time'])
                            target["result"] = "WIN"
                            target["active"] = False
                            
                if all(not t["active"] for t in targets.values()):
                    break
            
            # Format targets for JSON
            json_targets = {}
            for rrr, target in targets.items():
                if target["result"]:
                    json_targets[rrr] = {
                        "result": target["result"],
                        "exit_time": target["exit_time"]
                    }
            
            if not json_targets:
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
                "targets": json_targets,
                "probability": probability,
                "mfe_pct": float(mfe_diff),
                "mae_pct": float(mae_diff),
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

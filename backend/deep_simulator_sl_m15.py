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
TIMEFRAME_M15 = mt5.TIMEFRAME_M15
TIMEFRAME_M1 = mt5.TIMEFRAME_M1
CANDLES_M15 = 25000  # Approx 1 year of data

RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'deep_backtest_results_sl_m15.json')

def get_real_symbol(base_symbol):
    symbols = mt5.symbols_get()
    if symbols is None: return base_symbol
    for s in symbols:
        if base_symbol in s.name:
            return s.name
    return base_symbol

def get_m15_data(symbol):
    real_symbol = get_real_symbol(symbol)
    rates = mt5.copy_rates_from_pos(real_symbol, TIMEFRAME_M15, 0, CANDLES_M15)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def get_m1_data_forward(symbol, start_time, max_candles=1440): # max 24 hours
    real_symbol = get_real_symbol(symbol)
    start_ts = int(start_time.timestamp())
    rates = mt5.copy_rates_range(real_symbol, TIMEFRAME_M1, start_ts, start_ts + (max_candles * 60))
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
    
    # Add ATR
    df['atr_14'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    
    # Add Swing Highs/Lows (rolling 3 periods, shifted by 1 so we don't include current closing candle)
    df['swing_high_3'] = df['high'].shift(1).rolling(window=3).max()
    df['swing_low_3'] = df['low'].shift(1).rolling(window=3).min()
    
    df.dropna(inplace=True)
    return df

def run_deep_simulation():
    logging.info("Initializing MT5 Institutional Simulation Engine...")
    if not mt5.initialize():
        logging.error("MT5 initialization failed!")
        return
        
    try:
        model_path = os.path.join(os.path.dirname(__file__), "kronos_model.pkl")
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
            
            atr_val = float(row['atr_14'])
            swing_high = float(row['swing_high_3'])
            swing_low = float(row['swing_low_3'])
            
            # SL targets mapping
            targets = {
                "Fixed_0.1%": {"sl_dist": entry_price * 0.001},
                "Fixed_0.2%": {"sl_dist": entry_price * 0.002},
                "ATR_1.5x": {"sl_dist": max(entry_price * 0.0005, atr_val * 1.5)},
                "Swing_3": {"sl_dist": max(entry_price * 0.0005, (entry_price - swing_low) if direction == "BUY" else (swing_high - entry_price))}
            }
            
            for key, t in targets.items():
                sl_dist = t["sl_dist"]
                tp_dist = sl_dist * 1.5 # Target is always 1:1.5 RRR
                
                if direction == "BUY":
                    t["sl"] = entry_price - sl_dist
                    t["tp"] = entry_price + tp_dist
                else:
                    t["sl"] = entry_price + sl_dist
                    t["tp"] = entry_price - tp_dist
                    
                t["result"] = ""
                t["exit_time"] = None
                t["active"] = True
                
            # Drop down to M1 to track tick-by-tick
            m1_df = get_m1_data_forward(symbol, entry_time)
            if m1_df.empty: continue
            
            for _, m1_row in m1_df.iterrows():
                high = m1_row['high']
                low = m1_row['low']
                
                if direction == "BUY":
                    for key, target in targets.items():
                        if not target["active"]: continue
                        if low <= target["sl"]:
                            target["exit_time"] = str(m1_row['time'])
                            target["result"] = "LOSS"
                            target["active"] = False
                        elif high >= target["tp"]:
                            target["exit_time"] = str(m1_row['time'])
                            target["result"] = "WIN"
                            target["active"] = False
                else: # SELL
                    for key, target in targets.items():
                        if not target["active"]: continue
                        if high >= target["sl"]:
                            target["exit_time"] = str(m1_row['time'])
                            target["result"] = "LOSS"
                            target["active"] = False
                        elif low <= target["tp"]:
                            target["exit_time"] = str(m1_row['time'])
                            target["result"] = "WIN"
                            target["active"] = False
                            
                if all(not t["active"] for t in targets.values()):
                    break
            
            json_targets = {}
            for key, target in targets.items():
                if target["result"]:
                    json_targets[key] = {
                        "result": target["result"],
                        "exit_time": target["exit_time"]
                    }
            
            if not json_targets:
                continue
                
            all_simulated_trades.append({
                "pair": symbol,
                "direction": direction,
                "entry_time": str(entry_time),
                "targets": json_targets,
                "probability": probability
            })

    mt5.shutdown()
    
    logging.info(f"Simulation Complete! Processed {len(all_simulated_trades)} deep trades.")
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_simulated_trades, f, indent=4)
        
    logging.info(f"Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    run_deep_simulation()

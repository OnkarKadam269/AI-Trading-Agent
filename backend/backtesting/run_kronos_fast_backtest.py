import torch
import pandas as pd
import numpy as np
import time
import os
import sys
import pytz

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models", "kronos_real"))
from model import Kronos, KronosTokenizer, KronosPredictor

def get_symbol_properties(symbol):
    if "JPY" in symbol:
        return 0.01, 100000, True 
    elif "XAU" in symbol:
        return 0.10, 100, False 
    else: 
        return 0.0001, 100000, False

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def prepare_data(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if 'time' in df.columns:
        df = df.rename(columns={'time': 'timestamps'})
    elif 'date' in df.columns:
        df = df.rename(columns={'date': 'timestamps'})
        
    if 'tick_volume' in df.columns:
        df = df.rename(columns={'tick_volume': 'volume'})
        
    if 'amount' not in df.columns and 'volume' in df.columns and 'close' in df.columns:
        df['amount'] = df['volume'] * df['close']
        
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df.sort_values('timestamps').reset_index(drop=True)
    
    # Calculate ATR
    df['tr0'] = df['high'] - df['low']
    df['tr1'] = abs(df['high'] - df['close'].shift())
    df['tr2'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()
    
    # Calculate RSI
    df = calculate_rsi(df, period=14)
    
    return df

def simulate_fixed_target(df, start_idx, signal, entry_price, initial_atr, spread):
    """
    Simulates a fixed 1:1 TP/SL exit with a 12-candle timeout.
    """
    tp_distance = 1.0 * initial_atr
    sl_distance = 1.0 * initial_atr
    max_candles = 12
    
    if signal == 1: # BUY
        tp_price = entry_price + tp_distance
        sl_price = entry_price - sl_distance
        
        for i in range(start_idx, min(len(df), start_idx + max_candles)):
            high = df.iloc[i]['high']
            low = df.iloc[i]['low']
            
            # Pessimistic: check SL first
            if low <= sl_price:
                return sl_price - spread, i
            if high >= tp_price:
                return tp_price - spread, i
                
        # Timeout exit
        timeout_idx = min(len(df)-1, start_idx + max_candles - 1)
        return df.iloc[timeout_idx]['close'] - spread, timeout_idx
        
    elif signal == -1: # SELL
        tp_price = entry_price - tp_distance
        sl_price = entry_price + sl_distance
        
        for i in range(start_idx, min(len(df), start_idx + max_candles)):
            high = df.iloc[i]['high']
            low = df.iloc[i]['low']
            
            # Pessimistic: check SL first
            if high >= sl_price:
                return sl_price + spread, i
            if low <= tp_price:
                return tp_price + spread, i
                
        # Timeout exit
        timeout_idx = min(len(df)-1, start_idx + max_candles - 1)
        return df.iloc[timeout_idx]['close'] + spread, timeout_idx

def run_fast_backtest(timeframe, filename, symbol):
    csv_path = os.path.join(os.path.dirname(__file__), "data", "raw", filename)
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found. Skipping.")
        return
        
    pip_size, contract_size, is_jpy_quote = get_symbol_properties(symbol)
    spread = 0.5 * pip_size # 0.5 pip spread penalty
    
    df = prepare_data(csv_path)
    
    print("Loading Kronos Tokenizer and Model from HuggingFace...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    tokenizer = tokenizer.to(device)
    
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    
    lookback_window = 64
    pred_len = 1
    sample_count = 1  
    batch_size = 1    
    
    test_steps = len(df) - lookback_window - pred_len
    start_idx = lookback_window
    
    print(f"Starting BATCH inference on {device.upper()} for {test_steps} steps (Batch size: {batch_size})...")
    start_time = time.time()
    
    trade_logs = []
    account_balance = 5000.0
    risk_per_trade = 0.01  # 1% risk
    
    dfs = []
    xtsp = []
    ytsp = []
    indices = []
    
    ist = pytz.timezone('Asia/Kolkata')
    
    # To prevent multiple trades from opening concurrently, we track if we are in a trade
    in_trade_until_idx = -1
    
    for i in range(start_idx, start_idx + test_steps):
        # Skip if we are currently holding a trade from a previous signal
        if i <= in_trade_until_idx:
            continue
            
        context_start = i - lookback_window + 1
        x_df = df.iloc[context_start:i+1][['open', 'high', 'low', 'close', 'volume', 'amount']].reset_index(drop=True)
        x_timestamp = df.iloc[context_start:i+1]['timestamps'].reset_index(drop=True)
        y_timestamp = df.iloc[i+1:i+1+pred_len]['timestamps'].reset_index(drop=True)
        
        dfs.append(x_df)
        xtsp.append(x_timestamp)
        ytsp.append(y_timestamp)
        indices.append(i)
        
        if len(dfs) == batch_size or i == start_idx + test_steps - 1:
            try:
                pred_df_list = predictor.predict_batch(
                    df_list=dfs,
                    x_timestamp_list=xtsp,
                    y_timestamp_list=ytsp,
                    pred_len=pred_len,
                    T=1.0,
                    top_p=0.9,
                    sample_count=sample_count,
                    verbose=False
                )
                
                for j, pred_df in enumerate(pred_df_list):
                    idx = indices[j]
                    
                    pred_close = pred_df['close'].iloc[-1]
                    
                    current_close = df.iloc[idx]['close']
                    atr = df.iloc[idx]['atr']
                    rsi = df.iloc[idx]['rsi']
                    timestamp = df.iloc[idx+1]['timestamps']
                    
                    if pd.isna(atr) or pd.isna(rsi): 
                        continue
                        
                    # 2. Timezone Filter (No trading from 2 AM to 5 AM IST)
                    ts_utc = timestamp
                    if ts_utc.tzinfo is None:
                        ts_utc = pytz.utc.localize(ts_utc)
                    ts_ist = ts_utc.astimezone(ist)
                    
                    if 2 <= ts_ist.hour < 5:
                        continue # Block entries during this window
                        
                    signal = 0
                    gap = pred_close - current_close
                    
                    # MEAN REVERSION LOGIC: Oracle Trigger + RSI Overbought/Oversold Filter
                    if gap > (0.1 * atr) and rsi < 40: 
                        signal = 1 # BUY (Predicted up + Technically Oversold)
                    elif gap < -(0.1 * atr) and rsi > 60:
                        signal = -1 # SELL (Predicted down + Technically Overbought)
                        
                    if signal != 0:
                        entry_price = current_close
                        initial_risk = 1.0 * atr # 1:1 RRR
                        
                        # Institutional Lot Sizing
                        if is_jpy_quote:
                            risk_usd_per_lot = (initial_risk * contract_size) / current_close
                        else:
                            risk_usd_per_lot = initial_risk * contract_size
                            
                        lot_size = round((account_balance * risk_per_trade) / risk_usd_per_lot, 2)
                        if lot_size < 0.01: lot_size = 0.01
                        
                        # Simulate Fixed 1:1 TP/SL
                        exit_price, exit_idx = simulate_fixed_target(df, idx+1, signal, entry_price, atr, spread)
                        
                        # Calculate Profit
                        if signal == 1:
                            profit_points = exit_price - entry_price
                        else:
                            profit_points = entry_price - exit_price
                            
                        profit_pips = profit_points / pip_size
                        
                        if is_jpy_quote:
                            profit_usd = (profit_points * contract_size / exit_price) * lot_size
                        else:
                            profit_usd = profit_points * contract_size * lot_size
                            
                        account_balance += profit_usd
                        
                        trade_logs.append({
                            'Symbol': symbol,
                            'Entry_Time': timestamp,
                            'Exit_Time': df.iloc[exit_idx]['timestamps'],
                            'Type': 'BUY' if signal == 1 else 'SELL',
                            'Entry': entry_price,
                            'Exit': exit_price,
                            'Lot_Size': lot_size,
                            'Profit_Pips': round(profit_pips, 2),
                            'Profit_USD': round(profit_usd, 2),
                            'Balance': round(account_balance, 2)
                        })
                        
                        # Block new trades until this one closes
                        in_trade_until_idx = exit_idx
                    
            except Exception as e:
                print(f"Error processing batch: {e}")
            
            elapsed = time.time() - start_time
            if i % 100 == 0:
                print(f"Completed {i - start_idx + 1}/{test_steps} steps. Elapsed time: {elapsed:.2f}s")
            
            dfs = []
            xtsp = []
            ytsp = []
            indices = []
            
    total_time = time.time() - start_time
    print(f"Inference complete for {symbol} {timeframe}! Total time: {total_time:.2f}s")
    
    trades_df = pd.DataFrame(trade_logs)
    if not trades_df.empty:
        out_path = os.path.join(os.path.dirname(__file__), f"kronos_forex_trades_{symbol}_{timeframe}.csv")
        trades_df.to_csv(out_path, index=False)
        print(f"Saved trades for {symbol} {timeframe} to {out_path}")
        print(f"Final Balance: ${account_balance:.2f} (Total Trades: {len(trades_df)})")
    else:
        print(f"No trades executed for {symbol} {timeframe}.")

if __name__ == "__main__":
    tasks = [
        ("GBPUSD", "H1", "GBPUSDm_H1_7M.csv"),
        ("GBPUSD", "M30", "GBPUSDm_M30_7M.csv"),
        ("GBPUSD", "M15", "GBPUSDm_M15_7M.csv"),
        ("USDJPY", "H1", "USDJPYm_H1_7M.csv"),
        ("USDJPY", "M30", "USDJPYm_M30_7M.csv"),
        ("USDJPY", "M15", "USDJPYm_M15_7M.csv")
    ]
    for symbol, tf, fname in tasks:
        print(f"=== Starting {symbol} {tf} ===")
        run_fast_backtest(tf, fname, symbol)
        print("-" * 50)

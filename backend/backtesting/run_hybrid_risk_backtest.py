import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
torch.set_num_threads(os.cpu_count() or 4)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models", "kronos_real"))
from model import Kronos, KronosTokenizer, KronosPredictor

def get_bollinger_bands(df, period=20, std_dev=2):
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    df['bb_upper'] = sma + (std_dev * std)
    df['bb_lower'] = sma - (std_dev * std)
    df['sma20'] = sma
    return df

def calculate_ta(df, add_ema=False):
    df = get_bollinger_bands(df, 20, 2)
    tr0 = df['high'] - df['low']
    tr1 = (df['high'] - df['close'].shift()).abs()
    tr2 = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat((tr0, tr1, tr2), axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    df['+dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['-dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    df['+di'] = 100 * (df['+dm'].rolling(14).mean() / df['atr'])
    df['-di'] = 100 * (df['-dm'].rolling(14).mean() / df['atr'])
    df['dx'] = 100 * abs(df['+di'] - df['-di']) / (df['+di'] + df['-di'])
    df['adx'] = df['dx'].rolling(14).mean()
    
    if add_ema:
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        
    return df

def run_backtest():
    start_wall_time = time.time()
    
    print("Loading PyTorch Kronos Models...")
    device = "cpu"
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").to(device)
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small").to(device)
    model.eval()
    predictor = KronosPredictor(model, tokenizer, max_context=64)
    
    base_file = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "XAUUSD_M15_3Y.csv")
    df_m15 = pd.read_csv(base_file, parse_dates=['time'])
    df_m15 = df_m15.sort_values('time').set_index('time')
    
    end_date = df_m15.index[-1]
    start_date = end_date - pd.DateOffset(months=3) 
    df_m15 = df_m15.loc[start_date:]
    
    print(f"Data loaded from {start_date} to {end_date} (3 months).")
    
    # 1. Resample to H4 (Macro Bias)
    df_h4 = df_m15.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'tick_volume': 'sum'
    }).dropna()
    df_h4 = df_h4.rename(columns={'tick_volume': 'volume'})
    df_h4['amount'] = df_h4['volume'] * df_h4['close']
    df_h4 = calculate_ta(df_h4)
    
    # 2. Resample to M30 (ATR Shield)
    df_m30 = df_m15.resample('30min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'tick_volume': 'sum'
    }).dropna()
    df_m30 = df_m30.rename(columns={'tick_volume': 'volume'})
    df_m30['amount'] = df_m30['volume'] * df_m30['close']
    df_m30 = calculate_ta(df_m30)
    
    # 3. M15 native (Execution & 200 EMA)
    df_m15_ta = df_m15.copy()
    df_m15_ta = df_m15_ta.rename(columns={'tick_volume': 'volume'})
    df_m15_ta['amount'] = df_m15_ta['volume'] * df_m15_ta['close']
    df_m15_ta = calculate_ta(df_m15_ta, add_ema=True)
    
    # PyTorch Inference on H4
    print("\n--- Phase 1: PyTorch Inference on H4 ---")
    df_h4 = df_h4.dropna().reset_index()
    batch_size = 128
    lookback = 64
    pred_close_list = [np.nan] * len(df_h4)
    df_lists, x_times, y_times, indices = [], [], [], []
    
    print(f"Starting parallel batched inference on {len(df_h4)} H4 candles...")
    for i in range(lookback, len(df_h4) - 1):
        x_df = df_h4.iloc[i-lookback+1:i+1][['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
        x_timestamp = pd.Series(df_h4.iloc[i-lookback+1:i+1]['time'].values)
        y_timestamp = pd.Series([df_h4.iloc[i+1]['time']])
        
        df_lists.append(x_df)
        x_times.append(x_timestamp)
        y_times.append(y_timestamp)
        indices.append(i)
        
        if len(df_lists) == batch_size or i == len(df_h4) - 2:
            with torch.inference_mode():
                preds = predictor.predict_batch(
                    df_list=df_lists, x_timestamp_list=x_times, y_timestamp_list=y_times,
                    pred_len=1, T=1.0, top_p=0.9, sample_count=1, verbose=False
                )
            for idx_in_batch, p_df in enumerate(preds):
                pred_close_list[indices[idx_in_batch]] = p_df['close'].iloc[-1]
            df_lists, x_times, y_times, indices = [], [], [], []
            
    df_h4['pred_close'] = pred_close_list
    df_h4['gap'] = df_h4['pred_close'] - df_h4['close']
    
    # Hybrid Dual Timeframe Execution
    print("\n--- Phase 2: Hybrid Risk Execution (H4 -> M30 ATR -> M15 EMA) ---")
    
    df_h4 = df_h4.set_index('time')
    # df_m30 already has datetime index from resample
    df_m15_ta = df_m15_ta.dropna().reset_index()
    df_m15_ta['time_ist'] = df_m15_ta['time'] + pd.Timedelta(hours=3.5)
    
    SPREAD_POINTS = 0.60
    RISK_PCT = 0.01  # 1.0%
    
    balance = 5000.0
    trades, wins, losses = 0, 0, 0
    all_trades_log = []
    open_trade = None
    
    print(f"Simulating M15 execution on {len(df_m15_ta)} candles...")
    for i in range(1, len(df_m15_ta) - 1):
        curr = df_m15_ta.iloc[i]
        nxt = df_m15_ta.iloc[i+1]
        
        try:
            htf_state = df_h4.loc[:curr['time']].iloc[-1]
            m30_state = df_m30.loc[:curr['time']].iloc[-1]
        except IndexError:
            continue
        
        if open_trade is None:
            if pd.isna(htf_state['gap']) or pd.isna(m30_state['atr']):
                continue
                
            hour_ist = nxt['time_ist'].hour
            if 2 <= hour_ist < 5:
                continue
            
            htf_gap = htf_state['gap']
            htf_adx = htf_state['adx']
            
            m30_atr = m30_state['atr'] # USING M30 ATR FOR STOP LOSS
            
            # The 200 EMA Filter on M15
            m15_ema200 = curr['ema200']
            
            if htf_adx > 20:
                # BUY CONDITION
                if htf_gap > 0.1 * htf_state['atr']:
                    if curr['close'] > m15_ema200 and curr['close'] <= curr['bb_lower']:
                        sl_dist_points = 1.5 * m30_atr
                        risk_amount = balance * RISK_PCT
                        raw_lot = risk_amount / (sl_dist_points * 100)
                        lot_size = max(0.01, min(round(raw_lot, 2), 100.0))
                        
                        open_trade = {
                            'type': 'BUY', 'entry': nxt['open'], 'sl': nxt['open'] - sl_dist_points,
                            'risk_usd': risk_amount, 'lot_size': lot_size, 'entry_time_ist': nxt['time_ist'],
                            'balance_before': balance, 'base_atr': m30_atr
                        }
                        
                # SELL CONDITION
                elif htf_gap < -0.1 * htf_state['atr']:
                    if curr['close'] < m15_ema200 and curr['close'] >= curr['bb_upper']:
                        sl_dist_points = 1.5 * m30_atr
                        risk_amount = balance * RISK_PCT
                        raw_lot = risk_amount / (sl_dist_points * 100)
                        lot_size = max(0.01, min(round(raw_lot, 2), 100.0))
                        
                        open_trade = {
                            'type': 'SELL', 'entry': nxt['open'], 'sl': nxt['open'] + sl_dist_points,
                            'risk_usd': risk_amount, 'lot_size': lot_size, 'entry_time_ist': nxt['time_ist'],
                            'balance_before': balance, 'base_atr': m30_atr
                        }
        else:
            # Trailing stop using M30 ATR
            try:
                current_m30_atr = df_m30.loc[:curr['time']].iloc[-1]['atr']
            except IndexError:
                current_m30_atr = open_trade['base_atr']
                
            if open_trade['type'] == 'BUY':
                if curr['low'] <= open_trade['sl']:
                    points_diff = open_trade['sl'] - open_trade['entry']
                    points_diff -= SPREAD_POINTS 
                    
                    pnl_usd = points_diff * 100 * open_trade['lot_size']
                    balance += pnl_usd
                    trades += 1
                    if pnl_usd > 0: wins += 1; 
                    else: losses += 1
                    rrr = (pnl_usd / open_trade['risk_usd']) if open_trade['risk_usd'] > 0 else 0
                    
                    all_trades_log.append({
                        'Type': 'BUY', 'Entry Time (IST)': open_trade['entry_time_ist'].strftime('%Y-%m-%d %H:%M:%S'), 
                        'Exit Time (IST)': curr['time_ist'].strftime('%Y-%m-%d %H:%M:%S'), 'Entry Price': round(open_trade['entry'], 2), 
                        'Exit Price': round(open_trade['sl'], 2), 'Pip Capture': round(points_diff * 10, 1), 'Slippage/Spread': '6 pips',
                        'Lot Size': open_trade['lot_size'], 'Risk-Reward Ratio': round(rrr, 2), 'Profit/Loss ($)': round(pnl_usd, 2), 'Balance ($)': round(balance, 2)
                    })
                    open_trade = None
                else:
                    new_sl = curr['close'] - (1.5 * current_m30_atr)
                    if new_sl > open_trade['sl']: open_trade['sl'] = new_sl
                        
            elif open_trade['type'] == 'SELL':
                if curr['high'] >= open_trade['sl']:
                    points_diff = open_trade['entry'] - open_trade['sl']
                    points_diff -= SPREAD_POINTS
                    
                    pnl_usd = points_diff * 100 * open_trade['lot_size']
                    balance += pnl_usd
                    trades += 1
                    if pnl_usd > 0: wins += 1; 
                    else: losses += 1
                    rrr = (pnl_usd / open_trade['risk_usd']) if open_trade['risk_usd'] > 0 else 0
                    
                    all_trades_log.append({
                        'Type': 'SELL', 'Entry Time (IST)': open_trade['entry_time_ist'].strftime('%Y-%m-%d %H:%M:%S'), 
                        'Exit Time (IST)': curr['time_ist'].strftime('%Y-%m-%d %H:%M:%S'), 'Entry Price': round(open_trade['entry'], 2), 
                        'Exit Price': round(open_trade['sl'], 2), 'Pip Capture': round(points_diff * 10, 1), 'Slippage/Spread': '6 pips',
                        'Lot Size': open_trade['lot_size'], 'Risk-Reward Ratio': round(rrr, 2), 'Profit/Loss ($)': round(pnl_usd, 2), 'Balance ($)': round(balance, 2)
                    })
                    open_trade = None
                else:
                    new_sl = curr['close'] + (1.5 * current_m30_atr)
                    if new_sl < open_trade['sl']: open_trade['sl'] = new_sl

    win_rate = (wins / trades * 100) if trades > 0 else 0
    
    print(f"\nHybrid Results: Trades {trades}, WinRate {win_rate:.1f}%, Final Balance ${balance:.2f}")
    
    df_log = pd.DataFrame(all_trades_log)
    df_log.to_csv("XAUUSD_Hybrid_Risk_Backtest.csv", index=False)
    
    print(f"\nAll tests completed successfully in {time.time() - start_wall_time:.2f} seconds!")

if __name__ == "__main__":
    run_backtest()

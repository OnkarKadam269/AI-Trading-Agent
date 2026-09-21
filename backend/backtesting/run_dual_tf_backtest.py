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

def calculate_ta(df):
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
    return df

def predict_htf(predictor, df_htf):
    df_htf = df_htf.dropna().reset_index()
    batch_size = 128
    lookback = 64
    pred_close_list = [np.nan] * len(df_htf)
    
    df_lists, x_times, y_times, indices = [], [], [], []
    
    print(f"Starting parallel batched inference on {len(df_htf)} HTF candles...")
    for i in range(lookback, len(df_htf) - 1):
        x_df = df_htf.iloc[i-lookback+1:i+1][['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
        x_timestamp = pd.Series(df_htf.iloc[i-lookback+1:i+1]['time'].values)
        y_timestamp = pd.Series([df_htf.iloc[i+1]['time']])
        
        df_lists.append(x_df)
        x_times.append(x_timestamp)
        y_times.append(y_timestamp)
        indices.append(i)
        
        if len(df_lists) == batch_size or i == len(df_htf) - 2:
            with torch.inference_mode():
                preds = predictor.predict_batch(
                    df_list=df_lists, x_timestamp_list=x_times, y_timestamp_list=y_times,
                    pred_len=1, T=1.0, top_p=0.9, sample_count=1, verbose=False
                )
            for idx_in_batch, p_df in enumerate(preds):
                pred_close_list[indices[idx_in_batch]] = p_df['close'].iloc[-1]
            
            df_lists, x_times, y_times, indices = [], [], [], []
            
    df_htf['pred_close'] = pred_close_list
    df_htf['gap'] = df_htf['pred_close'] - df_htf['close']
    return df_htf

def simulate_dual_tf(df_htf, df_ltf, tf_name):
    # Prepare Data
    df_htf = df_htf.set_index('time')
    df_ltf = df_ltf.dropna().reset_index()
    df_ltf['time_ist'] = df_ltf['time'] + pd.Timedelta(hours=3.5)
    
    SPREAD_POINTS = 0.60
    RISK_PCT = 0.002
    
    balance = 5000.0
    trades, wins, losses = 0, 0, 0
    all_trades_log = []
    open_trade = None
    
    print(f"Simulating LTF execution on {len(df_ltf)} candles...")
    for i in range(1, len(df_ltf) - 1):
        curr = df_ltf.iloc[i]
        nxt = df_ltf.iloc[i+1]
        
        # Get HTF Bias valid for this LTF time
        # The HTF candle that is closest to curr.time but <= curr.time
        # In Pandas, asof gets the latest index value <= the given value
        try:
            htf_state = df_htf.loc[:curr['time']].iloc[-1]
        except IndexError:
            continue
        
        if pd.isna(htf_state['gap']):
            continue
            
        if open_trade is None:
            hour_ist = nxt['time_ist'].hour
            if 2 <= hour_ist < 5:
                continue
            
            htf_gap = htf_state['gap']
            htf_atr = htf_state['atr']
            htf_adx = htf_state['adx']
            
            if htf_adx > 20:
                # HTF Buy Bias + LTF Lower BB Sweep
                if htf_gap > 0.1 * htf_atr and curr['close'] <= curr['bb_lower']:
                    sl_dist_points = 1.5 * curr['atr']
                    risk_amount = balance * RISK_PCT
                    raw_lot = risk_amount / (sl_dist_points * 100)
                    lot_size = max(0.01, min(round(raw_lot, 2), 100.0))
                    
                    open_trade = {
                        'type': 'BUY', 'entry': nxt['open'], 'sl': nxt['open'] - sl_dist_points,
                        'risk_usd': risk_amount, 'lot_size': lot_size, 'entry_time_ist': nxt['time_ist'],
                        'balance_before': balance
                    }
                    
                # HTF Sell Bias + LTF Upper BB Sweep
                elif htf_gap < -0.1 * htf_atr and curr['close'] >= curr['bb_upper']:
                    sl_dist_points = 1.5 * curr['atr']
                    risk_amount = balance * RISK_PCT
                    raw_lot = risk_amount / (sl_dist_points * 100)
                    lot_size = max(0.01, min(round(raw_lot, 2), 100.0))
                    
                    open_trade = {
                        'type': 'SELL', 'entry': nxt['open'], 'sl': nxt['open'] + sl_dist_points,
                        'risk_usd': risk_amount, 'lot_size': lot_size, 'entry_time_ist': nxt['time_ist'],
                        'balance_before': balance
                    }
        else:
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
                        'Timeframe Config': tf_name, 'Type': 'BUY', 'Entry Time (IST)': open_trade['entry_time_ist'].strftime('%Y-%m-%d %H:%M:%S'), 
                        'Exit Time (IST)': curr['time_ist'].strftime('%Y-%m-%d %H:%M:%S'), 'Entry Price': round(open_trade['entry'], 2), 
                        'Exit Price': round(open_trade['sl'], 2), 'Pip Capture': round(points_diff * 10, 1), 'Slippage/Spread': '6 pips',
                        'Lot Size': open_trade['lot_size'], 'Risk-Reward Ratio': round(rrr, 2), 'Profit/Loss ($)': round(pnl_usd, 2), 'Balance ($)': round(balance, 2)
                    })
                    open_trade = None
                else:
                    new_sl = curr['close'] - (1.5 * curr['atr'])
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
                        'Timeframe Config': tf_name, 'Type': 'SELL', 'Entry Time (IST)': open_trade['entry_time_ist'].strftime('%Y-%m-%d %H:%M:%S'), 
                        'Exit Time (IST)': curr['time_ist'].strftime('%Y-%m-%d %H:%M:%S'), 'Entry Price': round(open_trade['entry'], 2), 
                        'Exit Price': round(open_trade['sl'], 2), 'Pip Capture': round(points_diff * 10, 1), 'Slippage/Spread': '6 pips',
                        'Lot Size': open_trade['lot_size'], 'Risk-Reward Ratio': round(rrr, 2), 'Profit/Loss ($)': round(pnl_usd, 2), 'Balance ($)': round(balance, 2)
                    })
                    open_trade = None
                else:
                    new_sl = curr['close'] + (1.5 * curr['atr'])
                    if new_sl < open_trade['sl']: open_trade['sl'] = new_sl

    win_rate = (wins / trades * 100) if trades > 0 else 0
    return balance, trades, win_rate, all_trades_log

def run_backtest():
    start_wall_time = time.time()
    
    print("Loading PyTorch Kronos Models...")
    device = "cpu"
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").to(device)
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small").to(device)
    model.eval()
    predictor = KronosPredictor(model, tokenizer, max_context=64)
    
    base_file = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "XAUUSD_M1_3Y.csv")
    df_m1 = pd.read_csv(base_file, parse_dates=['time'])
    df_m1 = df_m1.sort_values('time').set_index('time')
    
    end_date = df_m1.index[-1]
    start_date = end_date - pd.DateOffset(months=2) # Last 2 months
    df_m1 = df_m1.loc[start_date:]
    
    print(f"Data loaded from {start_date} to {end_date} (2 months).")
    
    def resample_df(rule):
        df_rs = df_m1.resample(rule).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'tick_volume': 'sum'
        }).dropna()
        df_rs = df_rs.rename(columns={'tick_volume': 'volume'})
        df_rs['amount'] = df_rs['volume'] * df_rs['close']
        return calculate_ta(df_rs)
    
    df_h1 = resample_df('1h')
    df_m30 = resample_df('30min')
    df_m5 = resample_df('5min')
    df_m3 = resample_df('3min')
    
    print("\n--- Phase 1: PyTorch Inference on HTF ---")
    df_h1 = predict_htf(predictor, df_h1)
    df_m30 = predict_htf(predictor, df_m30)
    
    print("\n--- Phase 2: Dual Timeframe Execution ---")
    
    # 1. H1 -> M5
    print("\nTesting: H1 -> M5")
    bal_h1, tr_h1, wr_h1, log_h1 = simulate_dual_tf(df_h1, df_m5, "H1->M5")
    print(f"H1->M5 Results: Trades {tr_h1}, WinRate {wr_h1:.1f}%, Balance ${bal_h1:.2f}")
    df_log_h1 = pd.DataFrame(log_h1)
    df_log_h1.to_csv("XAUUSD_H1_M5_Dual_Backtest.csv", index=False)
    
    # 2. M30 -> M3
    print("\nTesting: M30 -> M3")
    bal_m30, tr_m30, wr_m30, log_m30 = simulate_dual_tf(df_m30, df_m3, "M30->M3")
    print(f"M30->M3 Results: Trades {tr_m30}, WinRate {wr_m30:.1f}%, Balance ${bal_m30:.2f}")
    df_log_m30 = pd.DataFrame(log_m30)
    df_log_m30.to_csv("XAUUSD_M30_M3_Dual_Backtest.csv", index=False)
    
    print(f"\nAll tests completed successfully in {time.time() - start_wall_time:.2f} seconds!")

if __name__ == "__main__":
    run_backtest()

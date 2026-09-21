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
    start_date = end_date - pd.DateOffset(months=7)
    df_m15 = df_m15.loc[start_date:]
    
    print(f"Data loaded from {start_date} to {end_date} (7 months).")
    
    resample_rules = {
        'H1': '1h',
        'M30': '30min',
        'M15': '15min'
    }
    
    results_summary = []
    all_trades_log = []
    
    SPREAD_POINTS = 0.06 # 0.6 pips on standard XAUUSD broker
    RISK_PCT = 0.002 # 0.2%
    
    for tf_name, rule in resample_rules.items():
        print(f"\n--- Processing Timeframe: {tf_name} ---")
        if tf_name == 'M15':
            df_tf = df_m15.copy()
        else:
            df_tf = df_m15.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'tick_volume': 'sum'
            }).dropna()
            
        df_tf = df_tf.rename(columns={'tick_volume': 'volume'})
        df_tf['amount'] = df_tf['volume'] * df_tf['close']
        
        df_tf = get_bollinger_bands(df_tf, 20, 2)
        
        tr0 = df_tf['high'] - df_tf['low']
        tr1 = (df_tf['high'] - df_tf['close'].shift()).abs()
        tr2 = (df_tf['low'] - df_tf['close'].shift()).abs()
        tr = pd.concat((tr0, tr1, tr2), axis=1).max(axis=1)
        df_tf['atr'] = tr.rolling(14).mean()
        
        df_tf['up_move'] = df_tf['high'] - df_tf['high'].shift(1)
        df_tf['down_move'] = df_tf['low'].shift(1) - df_tf['low']
        df_tf['+dm'] = np.where((df_tf['up_move'] > df_tf['down_move']) & (df_tf['up_move'] > 0), df_tf['up_move'], 0)
        df_tf['-dm'] = np.where((df_tf['down_move'] > df_tf['up_move']) & (df_tf['down_move'] > 0), df_tf['down_move'], 0)
        df_tf['+di'] = 100 * (df_tf['+dm'].rolling(14).mean() / df_tf['atr'])
        df_tf['-di'] = 100 * (df_tf['-dm'].rolling(14).mean() / df_tf['atr'])
        df_tf['dx'] = 100 * abs(df_tf['+di'] - df_tf['-di']) / (df_tf['+di'] + df_tf['-di'])
        df_tf['adx'] = df_tf['dx'].rolling(14).mean()
        
        df_tf = df_tf.dropna().reset_index()
        
        # Convert MT5 server time (assumed UTC+2) to IST (UTC+5:30) for filtering
        # Time difference is 3.5 hours
        df_tf['time_ist'] = df_tf['time'] + pd.Timedelta(hours=3.5)
        
        batch_size = 128
        lookback = 64
        pred_close_list = [np.nan] * len(df_tf)
        
        df_lists, x_times, y_times, indices = [], [], [], []
        
        print(f"Starting parallel batched inference on {len(df_tf)} candles...")
        for i in range(lookback, len(df_tf) - 1):
            x_df = df_tf.iloc[i-lookback+1:i+1][['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
            x_timestamp = pd.Series(df_tf.iloc[i-lookback+1:i+1]['time'].values)
            y_timestamp = pd.Series([df_tf.iloc[i+1]['time']])
            
            df_lists.append(x_df)
            x_times.append(x_timestamp)
            y_times.append(y_timestamp)
            indices.append(i)
            
            if len(df_lists) == batch_size or i == len(df_tf) - 2:
                with torch.inference_mode():
                    preds = predictor.predict_batch(
                        df_list=df_lists,
                        x_timestamp_list=x_times,
                        y_timestamp_list=y_times,
                        pred_len=1, T=1.0, top_p=0.9, sample_count=1, verbose=False
                    )
                for idx_in_batch, p_df in enumerate(preds):
                    pred_close_list[indices[idx_in_batch]] = p_df['close'].iloc[-1]
                
                df_lists, x_times, y_times, indices = [], [], [], []
                
        df_tf['pred_close'] = pred_close_list
        df_tf['gap'] = df_tf['pred_close'] - df_tf['close']
        
        balance = 5000.0
        trades, wins, losses = 0, 0, 0
        open_trade = None
        
        for i in range(lookback, len(df_tf) - 1):
            curr = df_tf.iloc[i]
            nxt = df_tf.iloc[i+1]
            
            if open_trade is None:
                # Time filter: Do not enter between 2 AM and 5 AM IST
                hour_ist = nxt['time_ist'].hour
                if 2 <= hour_ist < 5:
                    continue
                    
                if pd.notna(curr['pred_close']) and curr['adx'] > 20:
                    # BUY
                    if curr['gap'] > 0.1 * curr['atr'] and curr['close'] < curr['bb_upper']:
                        sl_dist_points = 1.5 * curr['atr']
                        
                        # Live Risk Sizing Logic
                        risk_amount = balance * RISK_PCT
                        # (sl_dist_points / 0.01 tick size) * 1 tick value = loss per 1 lot. 
                        # Essentially sl_dist_points * 100 = loss per 1 standard lot.
                        raw_lot = risk_amount / (sl_dist_points * 100)
                        lot_size = round(raw_lot, 2)
                        lot_size = max(0.01, min(lot_size, 100.0))
                        
                        open_trade = {
                            'type': 'BUY', 
                            'entry': nxt['open'], 
                            'sl': nxt['open'] - sl_dist_points, 
                            'risk_usd': risk_amount,
                            'lot_size': lot_size,
                            'entry_time_ist': nxt['time_ist'], 
                            'balance_before': balance
                        }
                    
                    # SELL
                    elif curr['gap'] < -0.1 * curr['atr'] and curr['close'] > curr['bb_lower']:
                        sl_dist_points = 1.5 * curr['atr']
                        
                        risk_amount = balance * RISK_PCT
                        raw_lot = risk_amount / (sl_dist_points * 100)
                        lot_size = round(raw_lot, 2)
                        lot_size = max(0.01, min(lot_size, 100.0))
                        
                        open_trade = {
                            'type': 'SELL', 
                            'entry': nxt['open'], 
                            'sl': nxt['open'] + sl_dist_points, 
                            'risk_usd': risk_amount,
                            'lot_size': lot_size,
                            'entry_time_ist': nxt['time_ist'], 
                            'balance_before': balance
                        }
            else:
                if open_trade['type'] == 'BUY':
                    if curr['low'] <= open_trade['sl']:
                        points_diff = open_trade['sl'] - open_trade['entry']
                        # Apply Spread (Cost)
                        points_diff -= SPREAD_POINTS 
                        
                        pnl_usd = points_diff * 100 * open_trade['lot_size']
                        balance += pnl_usd
                        trades += 1
                        if pnl_usd > 0: wins += 1
                        else: losses += 1
                        rrr = (pnl_usd / open_trade['risk_usd']) if open_trade['risk_usd'] > 0 else 0
                        
                        all_trades_log.append({
                            'Timeframe': tf_name, 'Type': 'BUY', 'Entry Time (IST)': open_trade['entry_time_ist'].strftime('%Y-%m-%d %H:%M:%S'), 'Exit Time (IST)': curr['time_ist'].strftime('%Y-%m-%d %H:%M:%S'), 
                            'Entry Price': round(open_trade['entry'], 2), 'Exit Price': round(open_trade['sl'], 2), 'Lot Size': open_trade['lot_size'],
                            'Risk/Reward Ratio': round(rrr, 2), 'Profit/Loss ($)': round(pnl_usd, 2), 'Balance ($)': round(balance, 2)
                        })
                        open_trade = None
                    else:
                        new_sl = curr['close'] - (1.5 * curr['atr'])
                        if new_sl > open_trade['sl']: open_trade['sl'] = new_sl
                            
                elif open_trade['type'] == 'SELL':
                    if curr['high'] >= open_trade['sl']:
                        points_diff = open_trade['entry'] - open_trade['sl']
                        # Apply Spread (Cost)
                        points_diff -= SPREAD_POINTS
                        
                        pnl_usd = points_diff * 100 * open_trade['lot_size']
                        balance += pnl_usd
                        trades += 1
                        if pnl_usd > 0: wins += 1
                        else: losses += 1
                        rrr = (pnl_usd / open_trade['risk_usd']) if open_trade['risk_usd'] > 0 else 0
                        
                        all_trades_log.append({
                            'Timeframe': tf_name, 'Type': 'SELL', 'Entry Time (IST)': open_trade['entry_time_ist'].strftime('%Y-%m-%d %H:%M:%S'), 'Exit Time (IST)': curr['time_ist'].strftime('%Y-%m-%d %H:%M:%S'), 
                            'Entry Price': round(open_trade['entry'], 2), 'Exit Price': round(open_trade['sl'], 2), 'Lot Size': open_trade['lot_size'],
                            'Risk/Reward Ratio': round(rrr, 2), 'Profit/Loss ($)': round(pnl_usd, 2), 'Balance ($)': round(balance, 2)
                        })
                        open_trade = None
                    else:
                        new_sl = curr['close'] + (1.5 * curr['atr'])
                        if new_sl < open_trade['sl']: open_trade['sl'] = new_sl

        win_rate = (wins / trades * 100) if trades > 0 else 0
        total_pnl = balance - 5000.0
        results_summary.append({
            'Timeframe': tf_name,
            'Trades': trades,
            'Win Rate': f"{win_rate:.1f}%",
            'Final Balance': f"${balance:.2f}",
            'Net Profit': f"${total_pnl:.2f}"
        })
        print(f"Finished {tf_name} -> Trades: {trades}, WR: {win_rate:.1f}%, Balance: ${balance:.2f}")

    md_content = "# Hyper-Realistic AI Backtest Results (XAUUSD)\n\n"
    md_content += "**Parameters**: $5000 Starting Balance | 0.2% Dynamic Risk | 0.6 Pip Spread | No Trading 2AM-5AM IST\n\n"
    md_content += "| Timeframe | Trades | Win Rate | Net Profit | Final Balance |\n"
    md_content += "|---|---|---|---|---|\n"
    for r in results_summary:
        md_content += f"| {r['Timeframe']} | {r['Trades']} | {r['Win Rate']} | {r['Net Profit']} | {r['Final Balance']} |\n"
        
    out_path = r"C:\Users\Omkar\.gemini\antigravity\brain\8f5fff91-0345-40a5-8099-28a0241724f5\realistic_backtest_report.md"
    try:
        with open(out_path, 'w') as f:
            f.write(md_content)
    except:
        with open('realistic_backtest_report.md', 'w') as f:
            f.write(md_content)
            
    detailed_df = pd.DataFrame(all_trades_log)
    excel_path = os.path.join(os.path.dirname(__file__), "Hyper_Realistic_Backtest.xlsx")
    csv_path = os.path.join(os.path.dirname(__file__), "Hyper_Realistic_Backtest.csv")
    
    try:
        detailed_df.to_excel(excel_path, index=False)
        print(f"\nDetailed trade log exported to {excel_path}")
    except:
        detailed_df.to_csv(csv_path, index=False)
        print(f"\nopenpyxl not found, detailed trade log exported to {csv_path}")
            
    total_time = time.time() - start_wall_time
    print(f"\nAll tests completed successfully in {total_time:.2f} seconds!")

if __name__ == "__main__":
    run_backtest()

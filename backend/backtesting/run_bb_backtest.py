import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import warnings
from datetime import datetime

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
        max_win, max_loss, highest_rrr = 0.0, 0.0, 0.0
        open_trade = None
        
        for i in range(lookback, len(df_tf) - 1):
            curr = df_tf.iloc[i]
            nxt = df_tf.iloc[i+1]
            
            if open_trade is None:
                if pd.notna(curr['pred_close']) and curr['adx'] > 20:
                    # Bollinger Band Setup
                    # BUY: Gap > 0.1 ATR and Close < Upper BB (room to grow)
                    if curr['gap'] > 0.1 * curr['atr'] and curr['close'] < curr['bb_upper']:
                        sl_dist = 1.5 * curr['atr']
                        open_trade = {'type': 'BUY', 'entry': nxt['open'], 'sl': nxt['open'] - sl_dist, 'risk': sl_dist, 'entry_time': nxt['time'], 'balance_before': balance}
                    
                    # SELL: Gap < -0.1 ATR and Close > Lower BB
                    elif curr['gap'] < -0.1 * curr['atr'] and curr['close'] > curr['bb_lower']:
                        sl_dist = 1.5 * curr['atr']
                        open_trade = {'type': 'SELL', 'entry': nxt['open'], 'sl': nxt['open'] + sl_dist, 'risk': sl_dist, 'entry_time': nxt['time'], 'balance_before': balance}
            else:
                if open_trade['type'] == 'BUY':
                    if curr['low'] <= open_trade['sl']:
                        pnl_points = open_trade['sl'] - open_trade['entry']
                        pnl_usd = (pnl_points * 10) * 0.05
                        balance += pnl_usd
                        trades += 1
                        if pnl_usd > 0: wins += 1
                        else: losses += 1
                        rrr = pnl_points / open_trade['risk'] if open_trade['risk'] > 0 else 0
                        if rrr > highest_rrr: highest_rrr = rrr
                        
                        all_trades_log.append({
                            'Timeframe': tf_name, 'Type': 'BUY', 'Entry Time': open_trade['entry_time'], 'Exit Time': curr['time'], 
                            'Entry Price': open_trade['entry'], 'Exit Price': open_trade['sl'], 'Profit USD': pnl_usd, 'RRR': rrr
                        })
                        open_trade = None
                    else:
                        new_sl = curr['close'] - (1.5 * curr['atr'])
                        if new_sl > open_trade['sl']: open_trade['sl'] = new_sl
                            
                elif open_trade['type'] == 'SELL':
                    if curr['high'] >= open_trade['sl']:
                        pnl_points = open_trade['entry'] - open_trade['sl']
                        pnl_usd = (pnl_points * 10) * 0.05
                        balance += pnl_usd
                        trades += 1
                        if pnl_usd > 0: wins += 1
                        else: losses += 1
                        rrr = pnl_points / open_trade['risk'] if open_trade['risk'] > 0 else 0
                        if rrr > highest_rrr: highest_rrr = rrr
                        
                        all_trades_log.append({
                            'Timeframe': tf_name, 'Type': 'SELL', 'Entry Time': open_trade['entry_time'], 'Exit Time': curr['time'], 
                            'Entry Price': open_trade['entry'], 'Exit Price': open_trade['sl'], 'Profit USD': pnl_usd, 'RRR': rrr
                        })
                        open_trade = None
                    else:
                        new_sl = curr['close'] + (1.5 * curr['atr'])
                        if new_sl < open_trade['sl']: open_trade['sl'] = new_sl

        win_rate = (wins / trades * 100) if trades > 0 else 0
        results_summary.append({
            'Timeframe': tf_name,
            'Filter': 'Bollinger Band + SMA 20',
            'Trades': trades,
            'Win Rate': f"{win_rate:.1f}%",
            'Net Profit': f"${balance - 5000.0:.2f}",
            'Highest RRR': f"{highest_rrr:.2f}R"
        })
        print(f"Finished {tf_name} BB -> Trades: {trades}, WR: {win_rate:.1f}%, Profit: ${balance - 5000.0:.2f}")

    md_content = "# True PyTorch Bollinger Band Backtest (XAUUSD - 7 Months)\n\n"
    md_content += "| Timeframe | Filter | Trades | Win Rate | Net Profit | Highest RRR |\n"
    md_content += "|---|---|---|---|---|---|\n"
    for r in results_summary:
        md_content += f"| {r['Timeframe']} | {r['Filter']} | {r['Trades']} | {r['Win Rate']} | {r['Net Profit']} | {r['Highest RRR']} |\n"
        
    out_path = r"C:\Users\Omkar\.gemini\antigravity\brain\8f5fff91-0345-40a5-8099-28a0241724f5\bb_backtest_report.md"
    try:
        with open(out_path, 'w') as f:
            f.write(md_content)
    except:
        with open('bb_backtest_report.md', 'w') as f:
            f.write(md_content)
            
    # Export Detailed Trades to Excel/CSV
    detailed_df = pd.DataFrame(all_trades_log)
    excel_path = os.path.join(os.path.dirname(__file__), "Kronos_BB_Backtest_Detailed.xlsx")
    csv_path = os.path.join(os.path.dirname(__file__), "Kronos_BB_Backtest_Detailed.csv")
    
    detailed_df.to_csv(csv_path, index=False)
    try:
        detailed_df.to_excel(excel_path, index=False)
        print(f"\nDetailed trade log exported to {excel_path}")
    except:
        print(f"\nopenpyxl not found, detailed trade log exported to {csv_path}")
            
    total_time = time.time() - start_wall_time
    print(f"\nAll tests completed successfully in {total_time:.2f} seconds!")

if __name__ == "__main__":
    run_backtest()

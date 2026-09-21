import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import warnings
from datetime import timedelta

warnings.filterwarnings('ignore')
torch.set_num_threads(os.cpu_count() or 4)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models", "kronos_real"))
from model import Kronos, KronosTokenizer, KronosPredictor

def get_bollinger_bands(df, period=20, std_dev=2):
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    df['bb_upper'] = sma + (std_dev * std)
    df['bb_lower'] = sma - (std_dev * std)
    return df

def calculate_atr(df, period=14):
    tr0 = df['high'] - df['low']
    tr1 = (df['high'] - df['close'].shift()).abs()
    tr2 = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat((tr0, tr1, tr2), axis=1).max(axis=1)
    return tr.rolling(period).mean()

def run_hybrid_backtest():
    start_wall_time = time.time()
    
    print("Loading PyTorch Kronos Models...")
    device = "cpu"
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").to(device)
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small").to(device)
    model.eval()
    predictor = KronosPredictor(model, tokenizer, max_context=64)
    
    # 1. Load Data
    print("Loading data...")
    base_file = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "XAUUSD_M15_3Y.csv")
    df_m15 = pd.read_csv(base_file, parse_dates=['time'])
    df_m15 = df_m15.sort_values('time').set_index('time')
    
    # Filter Last 1 Year
    end_date = df_m15.index[-1]
    start_date = end_date - pd.DateOffset(years=1)
    df_m15 = df_m15.loc[start_date:]
    
    print(f"Data loaded from {start_date} to {end_date} (1 Year).")
    
    # 2. Resample to H4 (Macro Bias)
    df_h4 = df_m15.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'tick_volume': 'sum'
    }).dropna()
    df_h4 = df_h4.rename(columns={'tick_volume': 'volume'})
    df_h4['amount'] = df_h4['volume'] * df_h4['close']
    df_h4['atr'] = calculate_atr(df_h4)
    
    # 3. Resample to H1 (QML Execution)
    df_h1 = df_m15.resample('1h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'tick_volume': 'sum'
    }).dropna()
    df_h1['atr'] = calculate_atr(df_h1)
    
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
    
    # Pivot Highs / Lows on H1 (Window = 4)
    df_h1 = df_h1.dropna().reset_index()
    order = 4
    df_h1['pivot_high'] = False
    df_h1['pivot_low'] = False
    
    for i in range(order, len(df_h1) - order):
        if df_h1['high'].iloc[i] == max(df_h1['high'].iloc[i-order : i+order+1]):
            df_h1.loc[i, 'pivot_high'] = True
        if df_h1['low'].iloc[i] == min(df_h1['low'].iloc[i-order : i+order+1]):
            df_h1.loc[i, 'pivot_low'] = True

    # 4. Simulation
    print("\n--- Phase 2: Hybrid QML Execution (H4 Bias -> H1 QML) ---")
    
    df_h4 = df_h4.set_index('time')
    
    SPREAD_POINTS = 0.60
    RISK_PCT = 0.01
    BALANCE = 5000.0
    
    balance = BALANCE
    trades_log = []
    
    pending_orders = []
    open_trade = None
    pivots = []
    
    print(f"Simulating H1 execution on {len(df_h1)} candles...")
    for i in range(order, len(df_h1) - 1):
        curr = df_h1.iloc[i]
        
        try:
            htf_state = df_h4.loc[:curr['time']].iloc[-1]
        except IndexError:
            continue
            
        # 1. Check open trades
        if open_trade is not None:
            if open_trade['type'] == 'BUY':
                if curr['low'] <= open_trade['sl']:
                    loss = open_trade['sl'] - open_trade['entry'] - SPREAD_POINTS
                    pnl_usd = loss * 100 * open_trade['lot']
                    balance += pnl_usd
                    trades_log.append(build_trade_log(open_trade, curr, open_trade['sl'], pnl_usd, balance))
                    open_trade = None
                elif curr['high'] >= open_trade['tp']:
                    win = open_trade['tp'] - open_trade['entry'] - SPREAD_POINTS
                    pnl_usd = win * 100 * open_trade['lot']
                    balance += pnl_usd
                    trades_log.append(build_trade_log(open_trade, curr, open_trade['tp'], pnl_usd, balance))
                    open_trade = None
                    
            elif open_trade['type'] == 'SELL':
                if curr['high'] >= open_trade['sl']:
                    loss = open_trade['entry'] - open_trade['sl'] - SPREAD_POINTS
                    pnl_usd = loss * 100 * open_trade['lot']
                    balance += pnl_usd
                    trades_log.append(build_trade_log(open_trade, curr, open_trade['sl'], pnl_usd, balance))
                    open_trade = None
                elif curr['low'] <= open_trade['tp']:
                    win = open_trade['entry'] - open_trade['tp'] - SPREAD_POINTS
                    pnl_usd = win * 100 * open_trade['lot']
                    balance += pnl_usd
                    trades_log.append(build_trade_log(open_trade, curr, open_trade['tp'], pnl_usd, balance))
                    open_trade = None
            continue
            
        # 2. Check pending orders (expire after 50 H1 candles ~ 2 days)
        pending_orders = [o for o in pending_orders if (i - o['idx']) <= 50]
        
        order_triggered = False
        for o in pending_orders:
            if o['type'] == 'BUY' and curr['low'] <= o['entry'] and curr['high'] >= o['entry']:
                open_trade = o
                open_trade['entry_time'] = curr['time']
                order_triggered = True
                break
            elif o['type'] == 'SELL' and curr['high'] >= o['entry'] and curr['low'] <= o['entry']:
                open_trade = o
                open_trade['entry_time'] = curr['time']
                order_triggered = True
                break
                
        if order_triggered:
            pending_orders = [] 
            continue
            
        # 3. Add to pivots and check for QML + H4 Bias
        if curr['pivot_high']:
            pivots.append({'type': 'H', 'price': curr['high'], 'idx': i})
        if curr['pivot_low']:
            pivots.append({'type': 'L', 'price': curr['low'], 'idx': i})
            
        pivots = pivots[-10:]
        
        if len(pivots) >= 4:
            p1, p2, p3, p4 = pivots[-4], pivots[-3], pivots[-2], pivots[-1]
            htf_gap = htf_state['gap'] if not pd.isna(htf_state['gap']) else 0
            
            # Buy QML: L, H, LL, HH
            if p1['type'] == 'L' and p2['type'] == 'H' and p3['type'] == 'L' and p4['type'] == 'H':
                if p3['price'] < p1['price'] and p4['price'] > p2['price']:
                    if p4['idx'] == i:
                        # PYTORCH HYBRID FILTER: MACRO GAP MUST BE BULLISH
                        if htf_gap > 0:
                            entry = p1['price']
                            sl = p3['price'] - (curr['atr'] * 0.5) 
                            dist = entry - sl
                            if dist > 0:
                                tp = entry + (dist * 2.0) # 1:2 RRR
                                risk_usd = balance * RISK_PCT
                                lot = max(0.01, min(100.0, risk_usd / (dist * 100)))
                                
                                pending_orders.append({
                                    'type': 'BUY', 'entry': entry, 'sl': sl, 'tp': tp,
                                    'lot': round(lot, 2), 'idx': i, 'risk': risk_usd
                                })
                            
            # Sell QML: H, L, HH, LL
            elif p1['type'] == 'H' and p2['type'] == 'L' and p3['type'] == 'H' and p4['type'] == 'L':
                if p3['price'] > p1['price'] and p4['price'] < p2['price']:
                    if p4['idx'] == i:
                        # PYTORCH HYBRID FILTER: MACRO GAP MUST BE BEARISH
                        if htf_gap < 0:
                            entry = p1['price']
                            sl = p3['price'] + (curr['atr'] * 0.5)
                            dist = sl - entry
                            if dist > 0:
                                tp = entry - (dist * 2.0) # 1:2 RRR
                                risk_usd = balance * RISK_PCT
                                lot = max(0.01, min(100.0, risk_usd / (dist * 100)))
                                
                                pending_orders.append({
                                    'type': 'SELL', 'entry': entry, 'sl': sl, 'tp': tp,
                                    'lot': round(lot, 2), 'idx': i, 'risk': risk_usd
                                })

    # 5. Advanced Metrics
    df_log = pd.DataFrame(trades_log)
    if len(df_log) > 0:
        total_pnl = df_log['pnl'].sum()
        wins = df_log[df_log['pnl'] > 0]
        losses = df_log[df_log['pnl'] <= 0]
        win_rate = len(wins) / len(df_log) * 100
        
        gross_profit = wins['pnl'].sum()
        gross_loss = abs(losses['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        
        df_log['cum_bal'] = df_log['pnl'].cumsum() + BALANCE
        df_log['peak'] = df_log['cum_bal'].cummax()
        df_log['drawdown'] = (df_log['cum_bal'] - df_log['peak']) / df_log['peak'] * 100
        max_dd = df_log['drawdown'].min()
        
        current_loss_streak, max_loss_streak = 0, 0
        for p in df_log['pnl']:
            if p <= 0:
                current_loss_streak += 1
                max_loss_streak = max(max_loss_streak, current_loss_streak)
            else:
                current_loss_streak = 0
                
        returns = df_log['pnl'] / BALANCE
        sharpe = (returns.mean() / returns.std()) * np.sqrt(len(df_log)) if returns.std() != 0 else 0
        
        print("\n" + "="*40)
        print("HYBRID QML STRATEGY - ADVANCED METRICS")
        print("="*40)
        print(f"Total Trades:      {len(df_log)}")
        print(f"Start Balance:     ${BALANCE:.2f}")
        print(f"End Balance:       ${(BALANCE + total_pnl):.2f}")
        print(f"Net PnL:           ${total_pnl:.2f} ({(total_pnl/BALANCE)*100:.2f}%)")
        print(f"Win Rate:          {win_rate:.2f}%")
        print(f"Profit Factor:     {profit_factor:.2f}")
        print(f"Sharpe Ratio:      {sharpe:.2f}")
        print(f"Max Drawdown:      {max_dd:.2f}%")
        print(f"Max Losing Streak: {max_loss_streak}")
        print(f"Avg Winning Trade: ${avg_win:.2f}")
        print(f"Avg Losing Trade:  ${avg_loss:.2f}")
        print(f"Risk Per Trade:    1.0%")
        print("="*40)
        
        df_log = df_log.rename(columns={
            'type': 'Type', 'entry_time': 'Entry Time (IST)', 'exit_time': 'Exit Time (IST)',
            'entry': 'Entry Price', 'exit': 'Exit Price', 'lot': 'Lot Size',
            'pnl': 'Profit/Loss ($)', 'bal': 'Balance ($)'
        })
        df_log.to_csv("XAUUSD_H4_H1_QML_Backtest.csv", index=False)
        print("\nDetailed log saved to XAUUSD_H4_H1_QML_Backtest.csv")
    else:
        print("No trades executed.")
        
    print(f"\nAll tests completed successfully in {time.time() - start_wall_time:.2f} seconds!")

def build_trade_log(trade, curr_row, exit_price, pnl_usd, bal):
    return {
        'type': trade['type'],
        'entry_time': trade['entry_time'],
        'exit_time': curr_row['time'],
        'entry': round(trade['entry'], 2),
        'exit': round(exit_price, 2),
        'lot': trade['lot'],
        'pnl': round(pnl_usd, 2),
        'bal': round(bal, 2)
    }

if __name__ == "__main__":
    run_hybrid_backtest()

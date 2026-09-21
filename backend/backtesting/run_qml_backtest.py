import os
import time
import pandas as pd
import numpy as np
from datetime import timedelta

def calculate_atr(df, period=14):
    tr0 = df['high'] - df['low']
    tr1 = (df['high'] - df['close'].shift()).abs()
    tr2 = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat((tr0, tr1, tr2), axis=1).max(axis=1)
    return tr.rolling(period).mean()

def run_qml_backtest():
    start_time = time.time()
    
    # 1. Load Data
    print("Loading data...")
    base_file = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "XAUUSD_M15_3Y.csv")
    df = pd.read_csv(base_file, parse_dates=['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    # Filter Last 1 Year
    end_date = df['time'].iloc[-1]
    start_date = end_date - pd.DateOffset(years=1)
    df = df[df['time'] >= start_date].reset_index(drop=True)
    
    print(f"Data loaded from {df['time'].iloc[0]} to {df['time'].iloc[-1]} ({len(df)} candles).")
    
    # 2. Technicals
    df['atr'] = calculate_atr(df, 14)
    
    # Pivot Highs / Lows (Window = 5)
    order = 5
    df['pivot_high'] = False
    df['pivot_low'] = False
    
    # A simple peak detector
    for i in range(order, len(df) - order):
        if df['high'].iloc[i] == max(df['high'].iloc[i-order : i+order+1]):
            df.loc[i, 'pivot_high'] = True
        if df['low'].iloc[i] == min(df['low'].iloc[i-order : i+order+1]):
            df.loc[i, 'pivot_low'] = True

    # 3. Pattern Recognition & Simulation
    SPREAD_POINTS = 0.00
    RISK_PCT = 0.01
    BALANCE = 5000.0
    
    balance = BALANCE
    trades_log = []
    
    pending_orders = []
    open_trade = None
    
    # Collect pivots dynamically
    pivots = []
    
    print("Simulating QML structures...")
    
    for i in range(order, len(df) - 1):
        curr = df.iloc[i]
        
        # 1. Check open trades
        if open_trade is not None:
            if open_trade['type'] == 'BUY':
                if curr['low'] <= open_trade['sl']:
                    # SL Hit
                    loss = open_trade['sl'] - open_trade['entry'] - SPREAD_POINTS
                    pnl_usd = loss * 100 * open_trade['lot']
                    balance += pnl_usd
                    trades_log.append(build_trade_log(open_trade, curr, open_trade['sl'], pnl_usd, balance))
                    open_trade = None
                elif curr['high'] >= open_trade['tp']:
                    # TP Hit
                    win = open_trade['tp'] - open_trade['entry'] - SPREAD_POINTS
                    pnl_usd = win * 100 * open_trade['lot']
                    balance += pnl_usd
                    trades_log.append(build_trade_log(open_trade, curr, open_trade['tp'], pnl_usd, balance))
                    open_trade = None
                    
            elif open_trade['type'] == 'SELL':
                if curr['high'] >= open_trade['sl']:
                    # SL Hit
                    loss = open_trade['entry'] - open_trade['sl'] - SPREAD_POINTS
                    pnl_usd = loss * 100 * open_trade['lot']
                    balance += pnl_usd
                    trades_log.append(build_trade_log(open_trade, curr, open_trade['sl'], pnl_usd, balance))
                    open_trade = None
                elif curr['low'] <= open_trade['tp']:
                    # TP Hit
                    win = open_trade['entry'] - open_trade['tp'] - SPREAD_POINTS
                    pnl_usd = win * 100 * open_trade['lot']
                    balance += pnl_usd
                    trades_log.append(build_trade_log(open_trade, curr, open_trade['tp'], pnl_usd, balance))
                    open_trade = None
            continue
            
        # 2. Check pending orders
        # Remove expired
        pending_orders = [o for o in pending_orders if (i - o['idx']) <= 100]
        
        order_triggered = False
        for o in pending_orders:
            if o['type'] == 'BUY' and curr['low'] <= o['entry'] and curr['high'] >= o['entry']:
                # Trigger Buy
                open_trade = o
                open_trade['entry_time'] = curr['time']
                order_triggered = True
                break
            elif o['type'] == 'SELL' and curr['high'] >= o['entry'] and curr['low'] <= o['entry']:
                # Trigger Sell
                open_trade = o
                open_trade['entry_time'] = curr['time']
                order_triggered = True
                break
                
        if order_triggered:
            pending_orders = [] # Clear other orders once in a trade
            continue
            
        # 3. Add to pivots and check for QML
        if curr['pivot_high']:
            pivots.append({'type': 'H', 'price': curr['high'], 'idx': i})
        if curr['pivot_low']:
            pivots.append({'type': 'L', 'price': curr['low'], 'idx': i})
            
        # Keep only recent pivots
        pivots = pivots[-10:]
        
        if len(pivots) >= 4:
            p1, p2, p3, p4 = pivots[-4], pivots[-3], pivots[-2], pivots[-1]
            
            # Buy QML: L, H, LL, HH
            if p1['type'] == 'L' and p2['type'] == 'H' and p3['type'] == 'L' and p4['type'] == 'H':
                if p3['price'] < p1['price'] and p4['price'] > p2['price']:
                    # Valid Buy QML
                    if p4['idx'] == i: # Just formed
                        entry = p1['price']
                        sl = p3['price'] - (curr['atr'] * 0.5) # Buffer
                        dist = entry - sl
                        if dist > 0:
                            tp = entry + (dist * 2.0)
                            risk_usd = balance * RISK_PCT
                            lot = max(0.01, min(100.0, risk_usd / (dist * 100)))
                            
                            pending_orders.append({
                                'type': 'BUY', 'entry': entry, 'sl': sl, 'tp': tp,
                                'lot': round(lot, 2), 'idx': i, 'risk': risk_usd
                            })
                            
            # Sell QML: H, L, HH, LL
            elif p1['type'] == 'H' and p2['type'] == 'L' and p3['type'] == 'H' and p4['type'] == 'L':
                if p3['price'] > p1['price'] and p4['price'] < p2['price']:
                    # Valid Sell QML
                    if p4['idx'] == i:
                        entry = p1['price']
                        sl = p3['price'] + (curr['atr'] * 0.5)
                        dist = sl - entry
                        if dist > 0:
                            tp = entry - (dist * 2.0)
                            risk_usd = balance * RISK_PCT
                            lot = max(0.01, min(100.0, risk_usd / (dist * 100)))
                            
                            pending_orders.append({
                                'type': 'SELL', 'entry': entry, 'sl': sl, 'tp': tp,
                                'lot': round(lot, 2), 'idx': i, 'risk': risk_usd
                            })

    # 4. Advanced Metrics
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
        
        # Drawdown
        df_log['cum_bal'] = df_log['pnl'].cumsum() + BALANCE
        df_log['peak'] = df_log['cum_bal'].cummax()
        df_log['drawdown'] = (df_log['cum_bal'] - df_log['peak']) / df_log['peak'] * 100
        max_dd = df_log['drawdown'].min()
        
        # Streaks
        current_loss_streak, max_loss_streak = 0, 0
        for p in df_log['pnl']:
            if p <= 0:
                current_loss_streak += 1
                max_loss_streak = max(max_loss_streak, current_loss_streak)
            else:
                current_loss_streak = 0
                
        # Sharpe Ratio (Rough approx: annualized return / annualized volatility)
        # Using daily returns approximation from trade sequence
        returns = df_log['pnl'] / BALANCE
        sharpe = (returns.mean() / returns.std()) * np.sqrt(len(df_log)) if returns.std() != 0 else 0
        
        print("\n" + "="*40)
        print("15M QML STRATEGY - ADVANCED METRICS")
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
        
        # Rename columns for Excel
        df_log = df_log.rename(columns={
            'type': 'Type', 'entry_time': 'Entry Time (IST)', 'exit_time': 'Exit Time (IST)',
            'entry': 'Entry Price', 'exit': 'Exit Price', 'lot': 'Lot Size',
            'pnl': 'Profit/Loss ($)', 'bal': 'Balance ($)'
        })
        df_log.to_csv("XAUUSD_15M_QML_Backtest.csv", index=False)
        print("\nDetailed log saved to XAUUSD_15M_QML_Backtest.csv")
    else:
        print("No trades executed.")

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
    run_qml_backtest()

import pandas as pd
import numpy as np
import ta
import pickle
import os
import logging
from datetime import datetime, time

logging.basicConfig(level=logging.INFO, format='%(message)s')

PAIRS = ['GBPUSDm', 'USDJPYm', 'XAUUSDm']
TIMEFRAMES = {
    'M5': [0.05, 0.1, 0.15, 0.2],
    'M15': [0.1, 0.15, 0.2, 0.3],
    'M30': [0.2, 0.3, 0.4, 0.5],
    'H1': [0.3, 0.5, 0.7, 1.0],
    'H4': [0.5, 1.0, 1.5, 2.0]
}

def add_features(df):
    if df.empty: return df
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_width'] = df['bb_high'] - df['bb_low']
    df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
    df['dist_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    return df

def run_optimization():
    raw_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw')
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    results = []

    for tf, sl_list in TIMEFRAMES.items():
        print(f"\n=============================================")
        print(f"OPTIMIZING TIMEFRAME: {tf}")
        print(f"=============================================")
        
        model_path = os.path.join(models_dir, f'kronos_model_{tf}.pkl')
        if not os.path.exists(model_path):
            print(f"Model {model_path} missing.")
            continue
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        # Combine all pairs for this timeframe
        all_signals = []
        full_dfs = {}
        
        for pair in PAIRS:
            filepath = os.path.join(raw_dir, f"{pair}_{tf}_7M.csv")
            if not os.path.exists(filepath): continue
            
            df = pd.read_csv(filepath)
            df['time'] = pd.to_datetime(df['time'])
            
            df = add_features(df)
            df.dropna(inplace=True)
            
            # Predict
            features = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'bb_width', 'dist_sma20']
            X = df[features]
            probs = model.predict_proba(X)[:, 1]
            
            df['prob'] = probs
            df['signal'] = np.where(df['prob'] > 0.60, 1, np.where(df['prob'] < 0.40, -1, 0))
            
            # Time Filter: No trades 20:30 to 23:30 UTC (2 AM to 5 AM IST)
            df['hour'] = df['time'].dt.hour
            df['minute'] = df['time'].dt.minute
            
            def is_allowed(h, m):
                t = h + m/60.0
                if 20.5 <= t <= 23.5: return False
                return True
                
            df['allowed'] = [is_allowed(h, m) for h, m in zip(df['hour'], df['minute'])]
            
            # Keep track of the full dataframe for checking future prices
            full_dfs[pair] = df.reset_index(drop=True)
            
            # Extract just the signals to speed up backtesting
            sig_df = full_dfs[pair][(full_dfs[pair]['signal'] != 0) & (full_dfs[pair]['allowed'] == True)].copy()
            sig_df['pair'] = pair
            all_signals.append(sig_df)
            
        if not all_signals: continue
        signals_df = pd.concat(all_signals)
        print(f"Generated {len(signals_df)} valid signals for {tf} after probability and time filters.")
        
        for sl_pct in sl_list:
            tp_pct = sl_pct * 1.5
            
            total_trades = 0
            wins = 0
            losses = 0
            net_pnl_pips = 0
            
            for idx, row in signals_df.iterrows():
                pair = row['pair']
                sig = row['signal']
                entry_idx = idx
                
                # SPREAD AND SLIPPAGE
                # MT5 spread is in points. 1 pip = 10 points usually. We will use a baseline.
                # Assuming 5-digit broker, spread / 10 = pips. 
                spread_pips = row['spread'] / 10.0 if row['spread'] > 0 else 1.0
                slippage_pips = 2.0
                commission_pips = 0.5 # $5 per lot
                total_entry_cost_pips = (spread_pips / 2.0) + slippage_pips + (commission_pips / 2.0)
                total_exit_cost_pips = (spread_pips / 2.0) + slippage_pips + (commission_pips / 2.0)
                
                entry_price = row['close']
                
                if pair == 'XAUUSDm':
                    pip_value = 0.1 # Example multiplier for gold
                    sl_dist_price = entry_price * (sl_pct / 100.0)
                    tp_dist_price = entry_price * (tp_pct / 100.0)
                    cost_price = (total_entry_cost_pips + total_exit_cost_pips) * pip_value
                elif pair == 'USDJPYm':
                    pip_value = 0.01
                    sl_dist_price = entry_price * (sl_pct / 100.0)
                    tp_dist_price = entry_price * (tp_pct / 100.0)
                    cost_price = (total_entry_cost_pips + total_exit_cost_pips) * pip_value
                else: # GBPUSDm
                    pip_value = 0.0001
                    sl_dist_price = entry_price * (sl_pct / 100.0)
                    tp_dist_price = entry_price * (tp_pct / 100.0)
                    cost_price = (total_entry_cost_pips + total_exit_cost_pips) * pip_value

                if sig == 1: # BUY
                    sl = entry_price - sl_dist_price
                    tp = entry_price + tp_dist_price
                else: # SELL
                    sl = entry_price + sl_dist_price
                    tp = entry_price - tp_dist_price
                    
                # Look forward to find exit
                future_df = full_dfs[pair].iloc[entry_idx+1 : entry_idx+200] # max hold 200 candles
                
                trade_result = None
                pnl = 0
                for _, f_row in future_df.iterrows():
                    high = f_row['high']
                    low = f_row['low']
                    
                    if sig == 1:
                        if low <= sl:
                            trade_result = 'LOSS'
                            pnl = -sl_dist_price - cost_price
                            break
                        elif high >= tp:
                            trade_result = 'WIN'
                            pnl = tp_dist_price - cost_price
                            break
                    else:
                        if high >= sl:
                            trade_result = 'LOSS'
                            pnl = -sl_dist_price - cost_price
                            break
                        elif low <= tp:
                            trade_result = 'WIN'
                            pnl = tp_dist_price - cost_price
                            break
                            
                if trade_result == 'WIN':
                    wins += 1
                    net_pnl_pips += (pnl / pip_value)
                elif trade_result == 'LOSS':
                    losses += 1
                    net_pnl_pips += (pnl / pip_value)
                total_trades += 1
                
            if total_trades > 0:
                win_rate = (wins / total_trades) * 100
                expectancy = net_pnl_pips / total_trades
                print(f"  [SL {sl_pct}% | TP {tp_pct}%] Trades: {total_trades} | Win Rate: {win_rate:.2f}% | Net Pips: {net_pnl_pips:.1f} | Expectancy: {expectancy:.2f} pips/trade")
                results.append({
                    'Timeframe': tf, 'SL_%': sl_pct, 'TP_%': tp_pct, 
                    'Trades': total_trades, 'WinRate': win_rate, 
                    'NetPips': net_pnl_pips, 'Expectancy': expectancy
                })

    # Save best results
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        print("\n--- BEST STOP LOSS PER TIMEFRAME ---")
        best = res_df.loc[res_df.groupby('Timeframe')['Expectancy'].idxmax()]
        print(best.to_string(index=False))
        best.to_csv(os.path.join(os.path.dirname(__file__), 'optimal_sl_tp.csv'), index=False)

if __name__ == "__main__":
    run_optimization()

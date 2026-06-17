import pandas as pd
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict

def run_export():
    RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'deep_backtest_results_sl_m5.json')
    
    if not os.path.exists(RESULTS_FILE):
        print("No simulation results found. Please run deep_simulator_sl_m5.py first.")
        return
        
    with open(RESULTS_FILE, 'r') as f:
        all_trades = json.load(f)
        
    all_trades.sort(key=lambda x: datetime.strptime(x['entry_time'], "%Y-%m-%d %H:%M:%S"))
    
    PAIRS = set(t.get('pair') for t in all_trades if t.get('pair'))
    SL_TIERS = ["Fixed_0.1%", "Fixed_0.2%", "ATR_1.5x", "Swing_3"]
    stats = {}
    
    for sl_type in SL_TIERS:
        last_exit_times = {pair: datetime(1970, 1, 1) for pair in PAIRS}
        filtered_trades = []
        
        for t in all_trades:
            pair = t.get('pair')
            if not pair: continue
            
            entry_time_str = t.get('entry_time')
            if not entry_time_str: continue
            entry_time = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
            
            target = t.get('targets', {}).get(sl_type)
            if not target: continue
            
            exit_time_str = target.get('exit_time')
            if exit_time_str and exit_time_str != "None":
                exit_time = datetime.strptime(exit_time_str, "%Y-%m-%d %H:%M:%S")
            else:
                exit_time = entry_time + timedelta(minutes=120)
                
            if entry_time >= last_exit_times[pair]:
                trade_obj = {
                    'result': target.get('result', 'LOSS')
                }
                filtered_trades.append(trade_obj)
                last_exit_times[pair] = exit_time
                
        starting_capital = 5000.0
        capital = starting_capital
        risk_per_trade = 0.002
        reward_mult = 1.5 # Target is always 1.5 RRR
        
        wins = 0
        losses = 0
        peak = starting_capital
        max_dd = 0.0
        
        for ft in filtered_trades:
            risk_amount = capital * risk_per_trade
            if ft['result'] == 'WIN':
                capital += (risk_amount * reward_mult)
                wins += 1
            else:
                capital -= risk_amount
                losses += 1
                
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak * 100
            if dd > max_dd:
                max_dd = dd
                
        win_rate = wins / max(1, (wins + losses))
        stats[sl_type] = {
            "trades": wins + losses,
            "win_rate": win_rate,
            "profit": capital - starting_capital,
            "max_dd": max_dd
        }
    
    print("\n" + "="*70)
    print(" KRONOS M5 ADVANCED STOP-LOSS MATRIX (5k Start | 0.2% Risk | 1.5 RRR)")
    print("="*70)
    for sl_type, s in stats.items():
        print(f"{sl_type:<12} | Trades: {s['trades']:<4} | Win Rate: {s['win_rate']*100:>4.1f}% | Net Profit: ${s['profit']:>8.2f} | Max DD: {s['max_dd']:>4.1f}%")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_export()

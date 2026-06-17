import json
import os
from datetime import datetime, timedelta

def run_metrics():
    RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'deep_backtest_results_sl_m15.json')
    if not os.path.exists(RESULTS_FILE):
        print("No simulation results found.")
        return
        
    with open(RESULTS_FILE, 'r') as f:
        all_trades = json.load(f)
        
    all_trades.sort(key=lambda x: datetime.strptime(x['entry_time'], "%Y-%m-%d %H:%M:%S"))
    PAIRS = ["GBPUSD", "USDJPY", "XAUUSD", "AUDUSD"]
    
    cutoff_date = datetime(2026, 1, 1) # ~Last 6 months
    
    def extract_trades(sl_type):
        unfiltered = []
        filtered = []
        last_exit_times = {pair: datetime(1970, 1, 1) for pair in PAIRS}
        
        for t in all_trades:
            pair = t.get('pair')
            if pair not in PAIRS: continue
            
            entry_time = datetime.strptime(t['entry_time'], "%Y-%m-%d %H:%M:%S")
            if entry_time < cutoff_date:
                continue
                
            target = t.get('targets', {}).get(sl_type)
            if not target: continue
            
            exit_time_str = target.get('exit_time')
            if exit_time_str and exit_time_str != "None":
                exit_time = datetime.strptime(exit_time_str, "%Y-%m-%d %H:%M:%S")
            else:
                continue # Skip trades that never closed
                
            duration_mins = (exit_time - entry_time).total_seconds() / 60.0
            
            # GAP FILTER: If duration > 24 hours (1440 mins), it's a broker data gap anomaly
            if duration_mins > 1440 or duration_mins < 0:
                continue
                
            trade_obj = {
                'pair': pair,
                'result': target.get('result', 'LOSS'),
                'entry_time': entry_time,
                'exit_time': exit_time,
                'duration_mins': duration_mins
            }
            
            # Unfiltered
            unfiltered.append(trade_obj)
            
            # Filtered (Non-Concurrent)
            if entry_time >= last_exit_times[pair]:
                filtered.append(trade_obj)
                last_exit_times[pair] = exit_time
                
        return unfiltered, filtered
        
    def calculate_stats(trade_list, name, sl_type):
        starting_capital = 5000.0
        capital = starting_capital
        risk_per_trade = 0.002
        reward_mult = 1.5
        
        wins = 0
        losses = 0
        peak = starting_capital
        max_dd = 0.0
        
        for ft in trade_list:
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
                
        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        net_profit = capital - starting_capital
        
        print(f"\n=======================================================")
        print(f" {name}")
        print(f" M15 | {sl_type} | $5k Start | 0.2% Risk | 1.5 RRR")
        print(f" (Last 6 Months | Data Gap Filter Applied)")
        print(f"=======================================================")
        print(f"Net Profit:            ${net_profit:,.2f}")
        print(f"Max Drawdown:          {max_dd:.2f}%")
        print(f"Win Rate:              {win_rate:.1f}%")
        print(f"Total Trades:          {total_trades}")
        print(f"=======================================================")

    unfiltered_01, filtered_01 = extract_trades("Fixed_0.1%")
    calculate_stats(filtered_01, "WITHOUT CONCURRENT EXECUTION (Filtered 1 Per Pair)", "Fixed_0.1%")
    calculate_stats(unfiltered_01, "WITH CONCURRENT EXECUTION (Unfiltered Signals)", "Fixed_0.1%")
    
    unfiltered_02, filtered_02 = extract_trades("Fixed_0.2%")
    calculate_stats(filtered_02, "WITHOUT CONCURRENT EXECUTION (Filtered 1 Per Pair)", "Fixed_0.2%")
    calculate_stats(unfiltered_02, "WITH CONCURRENT EXECUTION (Unfiltered Signals)", "Fixed_0.2%")
    
    # Pair-Wise Analysis
    print("\n\n=======================================================")
    print(" PAIR-WISE STOP LOSS OPTIMIZATION MATRIX")
    print(" (Without Concurrent Execution - Live Agent Settings)")
    print("=======================================================")
    
    def get_pair_stats(trade_list, pair_name):
        pair_trades = [t for t in trade_list if t['pair'] == pair_name]
        wins = sum(1 for t in pair_trades if t['result'] == 'WIN')
        total = len(pair_trades)
        wr = (wins / total * 100) if total > 0 else 0
        return total, wr
        
    print(f"{'Pair':<10} | {'0.1% Win Rate':<15} | {'0.2% Win Rate':<15}")
    print("-" * 45)
    for pair in PAIRS:
        t_01, wr_01 = get_pair_stats(filtered_01, pair)
        t_02, wr_02 = get_pair_stats(filtered_02, pair)
        print(f"{pair:<10} | {wr_01:>5.1f}% ({t_01:<4} tr) | {wr_02:>5.1f}% ({t_02:<4} tr)")
    print("=======================================================\n")

if __name__ == "__main__":
    run_metrics()

import json
import os
from datetime import datetime, timedelta
from collections import defaultdict

def run_metrics():
    RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'deep_backtest_results_sl_m15.json')
    
    if not os.path.exists(RESULTS_FILE):
        print("No simulation results found.")
        return
        
    with open(RESULTS_FILE, 'r') as f:
        all_trades = json.load(f)
        
    all_trades.sort(key=lambda x: datetime.strptime(x['entry_time'], "%Y-%m-%d %H:%M:%S"))
    
    PAIRS = set(t.get('pair') for t in all_trades if t.get('pair'))
    sl_type = "Fixed_0.1%"
    
    # --- DATA EXTRACTION ---
    unfiltered_trades = []
    filtered_trades = []
    
    last_exit_times = {pair: datetime(1970, 1, 1) for pair in PAIRS}
    
    for t in all_trades:
        pair = t.get('pair')
        if not pair: continue
        
        entry_time = datetime.strptime(t['entry_time'], "%Y-%m-%d %H:%M:%S")
        target = t.get('targets', {}).get(sl_type)
        if not target: continue
        
        exit_time_str = target.get('exit_time')
        if exit_time_str and exit_time_str != "None":
            exit_time = datetime.strptime(exit_time_str, "%Y-%m-%d %H:%M:%S")
        else:
            exit_time = entry_time + timedelta(minutes=120)
            
        trade_obj = {
            'pair': pair,
            'result': target.get('result', 'LOSS'),
            'entry_time': entry_time,
            'exit_time': exit_time,
            'duration_mins': (exit_time - entry_time).total_seconds() / 60.0
        }
        
        # 1. Unfiltered (Concurrent allowed)
        unfiltered_trades.append(trade_obj)
        
        # 2. Filtered (No Concurrency)
        if entry_time >= last_exit_times[pair]:
            filtered_trades.append(trade_obj)
            last_exit_times[pair] = exit_time
            
    # --- CALCULATION LOGIC ---
    def calculate_stats(trade_list, name):
        starting_capital = 5000.0
        capital = starting_capital
        risk_per_trade = 0.002
        reward_mult = 1.5
        
        wins = 0
        losses = 0
        peak = starting_capital
        max_dd = 0.0
        
        current_win_streak = 0
        max_win_streak = 0
        current_loss_streak = 0
        max_loss_streak = 0
        
        durations = []
        daily_counts = defaultdict(int)
        
        for ft in trade_list:
            risk_amount = capital * risk_per_trade
            if ft['result'] == 'WIN':
                capital += (risk_amount * reward_mult)
                wins += 1
                current_win_streak += 1
                current_loss_streak = 0
                if current_win_streak > max_win_streak:
                    max_win_streak = current_win_streak
            else:
                capital -= risk_amount
                losses += 1
                current_loss_streak += 1
                current_win_streak = 0
                if current_loss_streak > max_loss_streak:
                    max_loss_streak = current_loss_streak
                    
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak * 100
            if dd > max_dd:
                max_dd = dd
                
            durations.append(ft['duration_mins'])
            date_str = ft['entry_time'].strftime("%Y-%m-%d")
            daily_counts[date_str] += 1
            
        total_trades = wins + losses
        win_rate = wins / max(1, total_trades) * 100
        net_profit = capital - starting_capital
        
        if len(daily_counts) > 0:
            min_day = min(daily_counts, key=daily_counts.get)
            max_day = max(daily_counts, key=daily_counts.get)
            min_trades_day = daily_counts[min_day]
            max_trades_day = daily_counts[max_day]
            avg_trades_day = sum(daily_counts.values()) / len(daily_counts)
        else:
            min_day, max_day, min_trades_day, max_trades_day, avg_trades_day = "N/A", "N/A", 0, 0, 0
            
        if len(durations) > 0:
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            min_duration = min(durations)
        else:
            avg_duration, max_duration, min_duration = 0, 0, 0
            
        print(f"\n=======================================================")
        print(f" {name}")
        print(f" M15 | Fixed_0.1% | $5k Start | 0.2% Risk | 1.5 RRR")
        print(f"=======================================================")
        print(f"Net Profit:            ${net_profit:,.2f}")
        print(f"Max Drawdown:          {max_dd:.2f}%")
        print(f"Win Rate:              {win_rate:.1f}%")
        print(f"Total Trades:          {total_trades}")
        print(f"Longest Win Streak:    {max_win_streak}")
        print(f"Longest Loss Streak:   {max_loss_streak}")
        print(f"-------------------------------------------------------")
        print(f"Fewest Trades in Day:  {min_trades_day} trades (on {min_day})")
        print(f"Highest Trades in Day: {max_trades_day} trades (on {max_day})")
        print(f"Average Trades/Day:    {avg_trades_day:.1f}")
        print(f"-------------------------------------------------------")
        print(f"Lowest Trade Duration: {min_duration:.1f} mins")
        print(f"Highest Trade Duration:{max_duration:.1f} mins")
        print(f"Average Trade Duration:{avg_duration:.1f} mins")
        print(f"=======================================================\n")

    calculate_stats(filtered_trades, "WITHOUT CONCURRENT EXECUTION (Filtered 1 Per Pair)")
    calculate_stats(unfiltered_trades, "WITH CONCURRENT EXECUTION (Unfiltered Signals)")

if __name__ == "__main__":
    run_metrics()

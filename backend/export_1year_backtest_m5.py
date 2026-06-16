import os
import json
import csv
from collections import defaultdict
from datetime import datetime, timedelta

RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'deep_backtest_results_m5.json')
CSV_OUTPUT = os.path.join(os.path.dirname(__file__), '6_month_backtest_report_m5.csv')

def run_1_year_export():
    with open(RESULTS_FILE, 'r') as f:
        all_trades = json.load(f)
        
    # Sort all trades chronologically
    all_trades.sort(key=lambda x: datetime.strptime(x['entry_time'], "%Y-%m-%d %H:%M:%S"))
    
    filtered_trades = []
    
    # Track the exit time of the last trade for each pair to enforce no-overlap rule
    last_exit_times = defaultdict(lambda: datetime.min)
    
    for t in all_trades:
        pair = t.get('pair')
        if pair == 'EURUSD': 
            continue
            
        try:
            entry_time = datetime.strptime(t['entry_time'], "%Y-%m-%d %H:%M:%S")
            exit_time_str = t.get('exit_time')
            if exit_time_str and exit_time_str != "None":
                exit_time = datetime.strptime(exit_time_str, "%Y-%m-%d %H:%M:%S")
                # Fix for corrupted MT5 historical data gaps that set exit dates months in the future
                if (exit_time - entry_time).total_seconds() > 4 * 3600:
                    exit_time = entry_time + timedelta(hours=4)
            else:
                exit_time = entry_time + timedelta(minutes=t.get('duration_minutes', 120))
                
            # Only take the trade if the previous trade on this pair has closed
            if entry_time >= last_exit_times[pair]:
                filtered_trades.append({
                    'dt': entry_time,
                    'date': entry_time.strftime("%Y-%m-%d"),
                    'time': entry_time.strftime("%H:%M:%S"),
                    'exit_dt': exit_time,
                    'pair': pair,
                    'direction': t.get('direction', 'BUY'),
                    'result': t.get('result', 'LOSS')
                })
                last_exit_times[pair] = exit_time
                
        except Exception as e:
            continue
            
    print(f"Original 1-Year Trades (Excl EURUSD): {len([t for t in all_trades if t.get('pair') != 'EURUSD'])}")
    print(f"Filtered 1-Year Trades (No Overlaps): {len(filtered_trades)}")
    
    # Now run the quant simulation on the filtered trades
    starting_capital = 1000.0
    capital = starting_capital
    risk_per_trade = 0.001 # 0.1%
    reward_risk_ratio = 1.5
    
    daily_pnl = defaultdict(float)
    peak_equity = starting_capital
    max_drawdown_pct = 0.0
    max_drawdown_usd = 0.0
    
    gross_profit = 0.0
    gross_loss = 0.0
    
    current_win_streak = 0
    max_win_streak = 0
    current_loss_streak = 0
    max_loss_streak = 0
    
    with open(CSV_OUTPUT, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Trade Number', 'Date', 'Time', 'Pair', 'Direction', 'Result', 'Risk Amount ($)', 'Profit/Loss ($)', 'New Account Balance ($)'])
        
        for idx, t in enumerate(filtered_trades):
            risk_amount = capital * risk_per_trade
            
            if t['result'] == 'WIN':
                profit = risk_amount * reward_risk_ratio
                capital += profit
                daily_pnl[t['date']] += profit
                gross_profit += profit
                pnl_str = f"{profit:.2f}"
                
                current_win_streak += 1
                if current_win_streak > max_win_streak: max_win_streak = current_win_streak
                current_loss_streak = 0
            else:
                loss = risk_amount
                capital -= loss
                daily_pnl[t['date']] -= loss
                gross_loss += loss
                pnl_str = f"-{loss:.2f}"
                
                current_loss_streak += 1
                if current_loss_streak > max_loss_streak: max_loss_streak = current_loss_streak
                current_win_streak = 0
                
            if capital > peak_equity:
                peak_equity = capital
            else:
                dd_pct = (peak_equity - capital) / peak_equity
                dd_usd = peak_equity - capital
                if dd_pct > max_drawdown_pct: max_drawdown_pct = dd_pct
                if dd_usd > max_drawdown_usd: max_drawdown_usd = dd_usd
                
            writer.writerow([
                idx + 1,
                t['date'],
                t['time'],
                t['pair'],
                t['direction'],
                t['result'],
                f"{risk_amount:.2f}",
                pnl_str,
                f"{capital:.2f}"
            ])
            
    total_pnl = capital - starting_capital
    daily_returns_usd = list(daily_pnl.values())
    avg_daily_pnl = sum(daily_returns_usd) / len(daily_returns_usd) if daily_returns_usd else 0.0
    
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    win_rate = sum(1 for t in filtered_trades if t['result'] == 'WIN') / len(filtered_trades) if filtered_trades else 0
    expected_return_per_trade = (win_rate * reward_risk_ratio) - ((1 - win_rate) * 1.0)
    
    print("\n=== 1-YEAR INSTITUTIONAL BACKTEST REPORT ===")
    print(f"Total PnL ($): ${total_pnl:.2f}")
    print(f"Final Account Balance: ${capital:.2f}")
    print(f"Average daily profit/loss ($): ${avg_daily_pnl:.2f}")
    print(f"Highest profit day ($): ${max(daily_returns_usd) if daily_returns_usd else 0:.2f}")
    print(f"Highest loss day ($): ${min(daily_returns_usd) if daily_returns_usd else 0:.2f}")
    print(f"Maximum drawdown ($ and %): ${max_drawdown_usd:.2f} ({max_drawdown_pct*100:.2f}%)")
    print(f"Longest losing streak: {max_loss_streak}")
    print(f"Longest winning streak: {max_win_streak}")
    print(f"Profit factor: {profit_factor:.2f}")
    print(f"Expected return (Risk Units per trade): {expected_return_per_trade:.4f}R")
    print(f"Win Rate: {win_rate*100:.2f}%")
    print(f"Total Trades: {len(filtered_trades)}")

if __name__ == "__main__":
    run_1_year_export()

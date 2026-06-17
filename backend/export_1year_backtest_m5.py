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
    
    PAIRS = set(t.get('pair') for t in all_trades if t.get('pair'))
    RRR_TIERS = ["1:1", "1:1.5", "1:2", "1:3"]
    stats = {}
    
    # We will save the 1:1.5 trades specifically for the CSV generation
    csv_filtered_trades = []
    
    for rrr in RRR_TIERS:
        last_exit_times = {pair: datetime(1970, 1, 1) for pair in PAIRS}
        filtered_trades = []
        
        for t in all_trades:
            pair = t.get('pair')
            if not pair: continue
            
            entry_time_str = t.get('entry_time')
            if not entry_time_str: continue
            entry_time = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
            
            target = t.get('targets', {}).get(rrr)
            if not target: continue
            
            exit_time_str = target.get('exit_time')
            if exit_time_str and exit_time_str != "None":
                exit_time = datetime.strptime(exit_time_str, "%Y-%m-%d %H:%M:%S")
            else:
                exit_time = entry_time + timedelta(minutes=120)
                
            # Only take the trade if the previous trade on this pair has closed
            if entry_time >= last_exit_times[pair]:
                trade_obj = {
                    'result': target.get('result', 'LOSS'),
                    'date': entry_time.strftime("%Y-%m-%d"),
                    'time': entry_time.strftime("%H:%M:%S"),
                    'pair': pair,
                    'direction': t.get('direction', 'BUY')
                }
                filtered_trades.append(trade_obj)
                
                if rrr == "1:1.5":
                    csv_filtered_trades.append(trade_obj)
                    
                last_exit_times[pair] = exit_time
                
        # Calculate Profit
        starting_capital = 10000.0
        capital = starting_capital
        risk_per_trade = 0.002
        reward_mult = {"1:1": 1.0, "1:1.5": 1.5, "1:2": 2.0, "1:3": 3.0}[rrr]
        
        wins = 0
        losses = 0
        for ft in filtered_trades:
            risk_amount = capital * risk_per_trade
            if ft['result'] == 'WIN':
                capital += (risk_amount * reward_mult)
                wins += 1
            else:
                capital -= risk_amount
                losses += 1
                
        win_rate = wins / max(1, (wins + losses))
        stats[rrr] = {
            "trades": wins + losses,
            "win_rate": win_rate,
            "profit": capital - starting_capital
        }
    
    print("\n--- KRONOS M5 RRR ANALYSIS MATRIX ---")
    for rrr, s in stats.items():
        print(f"RRR {rrr} -> Trades: {s['trades']} | Win Rate: {s['win_rate']*100:.1f}% | Net Profit: ${s['profit']:.2f}")
    print("--------------------------------------\n")
    
    # Standard backtest continues below for specific strategy
    starting_capital = 10000.0
    capital = starting_capital
    risk_per_trade = 0.002 # 0.2%
    reward_risk_ratio = 1.5
    
    daily_pnl = defaultdict(float)
    trade_results = []
    
    wins = 0
    losses = 0
    
    peak_equity = starting_capital
    max_drawdown_pct = 0.0
    max_drawdown_usd = 0.0
    
    gross_profit = 0.0
    gross_loss = 0.0
    
    current_win_streak = 0
    max_win_streak = 0
    current_loss_streak = 0
    max_loss_streak = 0
    
    # We will output to CSV file
    CSV_OUTPUT = os.path.join(os.path.dirname(__file__), '6_month_backtest_report_m5.csv')
    with open(CSV_OUTPUT, 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(['Date', 'Time', 'Pair', 'Direction', 'Result', 'Risk Amount ($)', 'Profit/Loss ($)', 'New Account Balance ($)'])
        
        for idx, t in enumerate(csv_filtered_trades):
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

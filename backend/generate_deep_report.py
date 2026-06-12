import os
import json
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(message)s')

RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'deep_backtest_results.json')
REPORT_FILE = os.path.join(os.path.dirname(__file__), 'deep_analysis_report.md')

def generate_report():
    if not os.path.exists(RESULTS_FILE):
        logging.error("No results file found.")
        return
        
    with open(RESULTS_FILE, 'r') as f:
        trades = json.load(f)
        
    if len(trades) == 0:
        logging.error("No trades in results.")
        return
        
    pairs_data = defaultdict(list)
    for t in trades:
        pairs_data[t['pair']].append(t)
        
    report = "# KRONOS AI: Deep Institutional MFE/MAE Analysis Report\n\n"
    report += "This report mathematically analyzes the Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE) of thousands of historical simulated trades to determine the optimal Trailing Stop and Exit logic for each pair.\n\n"
    
    for pair, pair_trades in pairs_data.items():
        total = len(pair_trades)
        wins = [t for t in pair_trades if t['result'] == 'WIN']
        losses = [t for t in pair_trades if t['result'] == 'LOSS']
        
        win_rate = (len(wins) / total) * 100 if total > 0 else 0
        
        loss_mfes = [t['mfe_pct'] for t in losses if t['mfe_pct'] > 0]
        avg_loss_mfe = sum(loss_mfes) / len(loss_mfes) if loss_mfes else 0
        
        win_maes = [t['mae_pct'] for t in wins if t['mae_pct'] > 0]
        avg_win_mae = sum(win_maes) / len(win_maes) if win_maes else 0
        
        # Painful Reversals: Losing trades that were at least halfway to TP (0.15% profit) before reversing
        painful_reversals = [t for t in losses if t['mfe_pct'] >= 0.15]
        
        # Immediate Losers: Trades that never went into profit at all (MFE < 0.05%)
        immediate_losers = [t for t in losses if t['mfe_pct'] < 0.05]
        
        report += f"## {pair} Analysis\n"
        report += f"- **Total Trades Analyzed:** {total}\n"
        report += f"- **Base Win Rate (Fixed SL/TP):** {win_rate:.2f}%\n"
        report += f"- **Average Drawdown before Winning (MAE):** {avg_win_mae:.3f}%\n"
        report += f"  *(Insight: If this is very high, trades suffer a lot of heat before winning. Consider a wider SL.)*\n"
        report += f"- **Average Profit before Losing (MFE):** {avg_loss_mfe:.3f}%\n"
        report += f"- **Painful Reversals (>50% to TP before reversing to SL):** {len(painful_reversals)} trades\n"
        
        if len(painful_reversals) > 0:
            saved_rate = (len(wins) + len(painful_reversals)) / total * 100
            report += f"  *(Actionable Insight: If we implement a **Trailing Stop** that moves SL to Breakeven at 0.15% profit, {len(painful_reversals)} losses could be prevented! This would boost the Win/Breakeven rate to {saved_rate:.2f}%!)*\n"
        else:
            report += f"  *(Actionable Insight: Reversals are rare. A fixed SL/TP is optimal here.)*\n"
            
        report += f"- **Immediate Losers (Reversed instantly after entry):** {len(immediate_losers)} trades\n"
        report += f"  *(Actionable Insight: The AI Coach is analyzing the RSI/BB Width of these exact {len(immediate_losers)} bad entries to generate Veto Rules.)*\n"
            
        report += "\n---\n\n"
        
    with open(REPORT_FILE, 'w') as f:
        f.write(report)
        
    logging.info(f"Report generated successfully: {REPORT_FILE}")

if __name__ == "__main__":
    generate_report()

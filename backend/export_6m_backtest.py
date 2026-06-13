import os
import json
import logging
import csv
from datetime import datetime
from dateutil.relativedelta import relativedelta

logging.basicConfig(level=logging.INFO, format='%(message)s')

RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'deep_backtest_results.json')
CSV_EXPORT_FILE = os.path.join(os.path.dirname(__file__), 'backtest_6_months_organized.csv')

def export_organized_data():
    if not os.path.exists(RESULTS_FILE):
        logging.error("No deep_backtest_results.json file found. Please run the deep_simulator.py first.")
        return
        
    logging.info("Loading massive backtest dataset...")
    with open(RESULTS_FILE, 'r') as f:
        trades = json.load(f)
        
    if len(trades) == 0:
        logging.error("No trades in results.")
        return
        
    # Calculate the date 6 months ago from today
    # Using the current script execution time as the reference point
    six_months_ago = datetime.now() - relativedelta(months=6)
    
    filtered_trades = []
    
    for t in trades:
        try:
            # Assuming entry_time is formatted like "2023-01-01 10:00:00"
            trade_date = datetime.strptime(t['entry_time'], "%Y-%m-%d %H:%M:%S")
            if trade_date >= six_months_ago:
                filtered_trades.append(t)
        except Exception as e:
            pass # Skip if date parsing fails

    # Sort trades: First by Pair (alphabetically), then by Date (chronologically)
    filtered_trades.sort(key=lambda x: (x['pair'], datetime.strptime(x['entry_time'], "%Y-%m-%d %H:%M:%S")))
    
    if not filtered_trades:
        logging.error("No trades found in the last 6 months.")
        return

    logging.info(f"Filtered down to {len(filtered_trades)} trades over the last 6 months.")
    
    # Define CSV columns
    headers = [
        "Pair", "Date", "Time", "Direction", "AI Probability", 
        "Result", "Max Favorable Excursion (MFE %)", "Max Adverse Excursion (MAE %)", 
        "RSI", "Bollinger Band Width", "Distance from SMA20 (%)"
    ]
    
    with open(CSV_EXPORT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for t in filtered_trades:
            trade_dt = datetime.strptime(t['entry_time'], "%Y-%m-%d %H:%M:%S")
            date_str = trade_dt.strftime("%Y-%m-%d")
            time_str = trade_dt.strftime("%H:%M:%S")
            
            row = [
                t.get('pair', ''),
                date_str,
                time_str,
                t.get('direction', ''),
                f"{t.get('probability', 0):.2f}",
                t.get('result', ''),
                f"{t.get('mfe_pct', 0):.3f}%",
                f"{t.get('mae_pct', 0):.3f}%",
                f"{t.get('rsi', 0):.2f}",
                f"{t.get('bb_width', 0):.4f}",
                f"{t.get('dist_sma20', 0):.3f}%"
            ]
            writer.writerow(row)
            
    logging.info(f"Successfully exported and organized data to: {CSV_EXPORT_FILE}")

if __name__ == "__main__":
    export_organized_data()

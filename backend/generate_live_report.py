import sqlite3
import pandas as pd
import os

def generate_report():
    db_path = 'journal.db'
    
    if not os.path.exists(db_path):
        print("\n[ERROR] Could not find journal.db!")
        print("Please make sure you are running this script from the same directory where your agent is running.")
        return

    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM trades", conn)
        conn.close()
    except Exception as e:
        print(f"\n[ERROR] Failed to read database: {e}")
        return

    if df.empty:
        print("\n[INFO] Your Live Agent has not executed any trades yet! The database is empty.")
        print("This is completely normal if the market was too noisy or if the agent just started.")
        return

    print("=======================================================")
    print("           LIVE ACCOUNT PERFORMANCE REPORT")
    print("=======================================================")
    
    total_trades = len(df)
    
    # Calculate Winners and Losers based on PnL or direction/exit
    # If exit_price is populated, we can calculate actual outcome.
    # For now, let's look at raw trades.
    
    print(f"Total Live Trades Executed: {total_trades}")
    print("\n--- Trade History ---")
    
    for index, row in df.iterrows():
        open_time = row['open_time']
        symbol = row['symbol']
        direction = row['direction']
        entry = row['entry_price']
        sl = row['stop_loss']
        
        print(f"[{open_time}] {direction} {symbol} | Entry: {entry} | SL: {sl}")

    print("=======================================================")
    print("This data is pulled directly from your LIVE MT5 Exness Account.")

if __name__ == "__main__":
    generate_report()

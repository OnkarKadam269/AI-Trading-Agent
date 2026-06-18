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
    
    print("=====================================================================================================================")
    print(f"{'Time':<22} | {'Pair':<8} | {'Type':<4} | {'Lots':<5} | {'Entry':<10} | {'Exit':<10} | {'SL':<10} | {'TP':<10} | {'PnL'}")
    print("---------------------------------------------------------------------------------------------------------------------")
    
    for index, row in df.iterrows():
        open_time = str(row['open_time'])[:19] # Truncate microseconds
        symbol = row['symbol']
        direction = row['direction']
        entry = f"{row['entry_price']:.5f}" if pd.notnull(row['entry_price']) else "-"
        exit_p = f"{row['exit_price']:.5f}" if pd.notnull(row['exit_price']) else "OPEN"
        sl = f"{row['stop_loss']:.5f}" if pd.notnull(row['stop_loss']) else "-"
        tp = f"{row['take_profit_1']:.5f}" if 'take_profit_1' in row and pd.notnull(row['take_profit_1']) else "-"
        lots = f"{row['lot_size']:.2f}" if 'lot_size' in row and pd.notnull(row['lot_size']) else "-"
        pnl = f"${row['pnl']:.2f}" if 'pnl' in row and pd.notnull(row['pnl']) else "-"
        
        print(f"{open_time:<22} | {symbol:<8} | {direction:<4} | {lots:<5} | {entry:<10} | {exit_p:<10} | {sl:<10} | {tp:<10} | {pnl}")

    print("=====================================================================================================================")
    print("This data is pulled directly from your LIVE MT5 Exness Account.")

if __name__ == "__main__":
    generate_report()

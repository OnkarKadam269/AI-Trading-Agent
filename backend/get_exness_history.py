import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

def get_real_exness_history():
    print("Connecting to MetaTrader 5...")
    if not mt5.initialize():
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return

    # Get history for the last 14 days
    date_to = datetime.now() + timedelta(days=1)
    date_from = date_to - timedelta(days=14)
    
    deals = mt5.history_deals_get(date_from, date_to)
    if deals is None:
        print(f"Failed to get deals history: {mt5.last_error()}")
        mt5.shutdown()
        return

    # Convert to DataFrame
    df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
    
    # Format time
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # MAGIC NUMBER 234000 is Agent 1
    # MAGIC NUMBER 999000 is Kronos Hyper (Agent 2)
    # Exclude deposits/withdrawals (deal type 2, etc.)
    df_agent = df[(df['magic'] == 234000) & (df['type'] <= 1)].copy()
    
    if df_agent.empty:
        print("\nNo trades found in broker history for Agent 1 (Magic 234000).")
        mt5.shutdown()
        return

    print("\n=====================================================================================================================")
    print("                                      REAL EXNESS BROKER HISTORY (AGENT 1)")
    print("=====================================================================================================================")
    print(f"{'Time':<22} | {'Ticket':<10} | {'Pair':<8} | {'Type':<6} | {'Lots':<5} | {'Price':<10} | {'Commission':<10} | {'Swap':<8} | {'Profit'}")
    print("---------------------------------------------------------------------------------------------------------------------")
    
    total_profit = 0.0
    total_commission = 0.0
    total_swap = 0.0
    
    for index, row in df_agent.iterrows():
        deal_time = str(row['time'])
        ticket = str(row['ticket'])
        symbol = row['symbol']
        # type 0 is BUY, 1 is SELL
        deal_type = "BUY" if row['type'] == 0 else "SELL"
        # Entry (0) or Exit (1) deal
        entry_exit = "ENTRY" if row['entry'] == 0 else "EXIT"
        
        lots = f"{row['volume']:.2f}"
        price = f"{row['price']:.5f}"
        commission = row['commission']
        swap = row['swap']
        profit = row['profit']
        
        total_profit += profit
        total_commission += commission
        total_swap += swap
        
        # Only show the EXIT deals to see realized profit, or show all? Let's show all.
        print(f"{deal_time:<22} | {ticket:<10} | {symbol:<8} | {entry_exit:<6} | {lots:<5} | {price:<10} | ${commission:<9.2f} | ${swap:<7.2f} | ${profit:.2f}")

    print("=====================================================================================================================")
    
    net_pnl = total_profit + total_commission + total_swap
    
    print("\n[EXACT ACCOUNT METRICS - AGENT 1]")
    print(f"Gross Trade Profit: ${total_profit:.2f}")
    print(f"Total Commissions Paid: ${total_commission:.2f}")
    print(f"Total Swap/Overnight Fees: ${total_swap:.2f}")
    print("---------------------------------")
    print(f"NET REALIZED PNL: ${net_pnl:.2f}")
    print("=================================")
    print("If your local database PnL differs from this number, it is because the database")
    print("does not deduct Exness broker commissions and swaps! This report pulls directly")
    print("from the Exness server.")
    
    mt5.shutdown()

if __name__ == "__main__":
    get_real_exness_history()

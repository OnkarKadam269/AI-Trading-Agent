import pandas as pd
import os

def process_file(timeframe):
    csv_path = os.path.join(os.path.dirname(__file__), f"kronos_hybrid_trades_{timeframe}.csv")
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    if df.empty:
        return
        
    # We originally built in a 0.2 pip spread. 
    # To make it 0.6, we deduct an additional 0.4 points from Profit_Pips of every trade.
    df['Profit_Pips'] = df['Profit_Pips'] - 0.4
    
    # Adjust the Exit Price to reflect this 0.4 deduction
    df['Exit'] = df.apply(
        lambda row: row['Exit'] - 0.4 if row['Type'] == 'BUY' else row['Exit'] + 0.4,
        axis=1
    )
    
    # Recalculate Profit_USD
    df['Profit_USD'] = (df['Profit_Pips'] * 100 * df['Lot_Size']).round(2)
    
    # Recalculate Balance
    starting_balance = 5000.0
    df['Balance'] = starting_balance + df['Profit_USD'].cumsum()
    
    # Save the updated raw file
    df.to_csv(csv_path, index=False)
    
    # Calculate Max Drawdown %
    peak = df['Balance'].cummax()
    peak = peak.replace(to_replace=0, method='ffill')
    drawdown = (peak - df['Balance']) / peak * 100
    max_dd_percent = drawdown.max()
    
    # Calculate Streaks
    df['Win'] = df['Profit_USD'] > 0
    
    # Winning streak
    win_streaks = df['Win'].groupby((~df['Win']).cumsum()).sum()
    max_win_streak = int(win_streaks.max())
    
    # Losing streak
    loss_streaks = (~df['Win']).groupby(df['Win'].cumsum()).sum()
    max_loss_streak = int(loss_streaks.max())
    
    print(f"--- {timeframe} Timeframe (0.6 pts Spread) ---")
    print(f"Final Balance: ${df['Balance'].iloc[-1]:.2f}")
    print(f"Max Drawdown: {max_dd_percent:.2f}%")
    print(f"Highest Winning Streak: {max_win_streak}")
    print(f"Highest Losing Streak: {max_loss_streak}")
    print("-" * 40)

if __name__ == "__main__":
    for tf in ["H1", "M30", "M15"]:
        process_file(tf)

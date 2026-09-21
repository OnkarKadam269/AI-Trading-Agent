import os
import pandas as pd
import pickle

def run_backtest():
    model_path = os.path.join(os.path.dirname(__file__), 'kronos_model.pkl')
    try:
        with open(model_path, 'rb') as f:
            kronos_model = pickle.load(f)
    except Exception as e:
        print(f"Failed to load AI model: {e}")
        return

    processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    pairs = ['GBPUSD', 'USDJPY', 'XAUUSD']
    year_label = '3Y'
    
    features = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'bb_width', 'dist_sma20']
    
    total_balance = 5000.0
    total_trades = 0
    total_wins = 0
    total_losses = 0
    
    for pair in pairs:
        print(f"--- Running Method 2 (2 Pips Slippage) Backtest for {pair} ---")
        csv_m15 = os.path.join(processed_dir, f'{pair}_M15_{year_label}.csv')
        csv_m1 = os.path.join(processed_dir, f'{pair}_M1_{year_label}.csv')
        
        if not os.path.exists(csv_m15) or not os.path.exists(csv_m1):
            print(f"Files for {pair} not found.")
            continue
            
        print("Loading and Predicting M15...")
        df_m15 = pd.read_csv(csv_m15, parse_dates=['time'])
        df_m15.set_index('time', inplace=True)
        
        X = df_m15[features]
        df_m15['pred'] = kronos_model.predict(X)
        df_m15['prob'] = kronos_model.predict_proba(X)[:, 1]
        
        # Shift predictions to the next M1 candle
        df_m15['signal_time'] = df_m15.index + pd.Timedelta(minutes=15)
        signals = df_m15[['signal_time', 'pred', 'prob']].set_index('signal_time')
        
        print("Loading M1 and joining signals...")
        df_m1 = pd.read_csv(csv_m1, parse_dates=['time'])
        df_m1.set_index('time', inplace=True)
        
        df_m1 = df_m1.join(signals, how='left')
        
        # Risk settings
        if 'XAUUSD' in pair:
            stop_loss_pct = 0.002
            take_profit_pct = 0.003
            pip_size = 0.1  # 1 standard pip in gold
        elif 'JPY' in pair:
            stop_loss_pct = 0.001
            take_profit_pct = 0.0015
            pip_size = 0.01
        else:
            stop_loss_pct = 0.001
            take_profit_pct = 0.0015
            pip_size = 0.0001
            
        open_trades = []
        pair_balance = 5000.0
        wins = 0
        losses = 0
        brokerage_and_slippage_paid = 0.0
        
        print("Simulating Tick-by-Tick Execution...")
        df_m1_np = df_m1.reset_index().to_dict('records')
        
        for row in df_m1_np:
            current_low = row['low']
            current_high = row['high']
            
            # 1. Update Open Trades
            for trade in list(open_trades):
                hit_sl = False
                hit_tp = False
                
                if trade['type'] == 'BUY':
                    if current_low <= trade['sl']:
                        hit_sl = True
                    if current_high >= trade['tp']:
                        hit_tp = True
                        
                elif trade['type'] == 'SELL':
                    if current_high >= trade['sl']:
                        hit_sl = True
                    if current_low <= trade['tp']:
                        hit_tp = True
                        
                if hit_sl or hit_tp:
                    loss_amt_planned = pair_balance * 0.002
                    sl_distance_price_planned = abs(trade['entry'] - trade['sl'])
                    position_size_usd = loss_amt_planned / (sl_distance_price_planned / trade['entry'])
                    lot_size = position_size_usd / 100000.0
                    
                    # Commission: $5 per lot
                    commission = lot_size * 5.0
                    
                    # Slippage Penalty: 6 Pips Roundtrip Value in USD
                    # USD value of 1 pip for this position size
                    if 'JPY' in pair:
                        pip_value_usd = (pip_size / row['close']) * position_size_usd
                    else:
                        pip_value_usd = (pip_size / trade['entry']) * position_size_usd

                    slippage_penalty = pip_value_usd * 2.0
                    total_penalty = commission + slippage_penalty
                    
                    if hit_sl:
                        pair_balance -= (loss_amt_planned + total_penalty)
                        brokerage_and_slippage_paid += total_penalty
                        losses += 1
                        open_trades.remove(trade)
                        
                    elif hit_tp and not hit_sl:
                        win_amt_planned = pair_balance * 0.002 * 1.5
                        pair_balance += (win_amt_planned - total_penalty)
                        brokerage_and_slippage_paid += total_penalty
                        wins += 1
                        open_trades.remove(trade)

            # 2. Open New Trades
            if not pd.isna(row['pred']):
                if len(open_trades) < 5:
                    entry_price = row['open'] # Standard Entry
                    
                    if row['pred'] == 1 and row['prob'] > 0.55:
                        sl = entry_price * (1 - stop_loss_pct)
                        tp = entry_price * (1 + take_profit_pct)
                        open_trades.append({'type': 'BUY', 'entry': entry_price, 'sl': sl, 'tp': tp})
                    elif row['pred'] == 0 and (1 - row['prob']) > 0.55:
                        sl = entry_price * (1 + stop_loss_pct)
                        tp = entry_price * (1 - take_profit_pct)
                        open_trades.append({'type': 'SELL', 'entry': entry_price, 'sl': sl, 'tp': tp})
                        
        print(f"End Balance for {pair}: ${pair_balance:.2f}")
        print(f"Wins: {wins}, Losses: {losses}, Win Rate: {wins/(wins+losses)*100:.2f}%" if (wins+losses)>0 else "No trades.")
        print(f"Total Penalty Paid (Comm + Slip): ${brokerage_and_slippage_paid:.2f}")
        
        total_trades += (wins + losses)
        total_wins += wins
        total_losses += losses
        total_balance += (pair_balance - 5000.0)
        
    print("\n===============================")
    print("      OVERALL 3-YEAR BACKTEST    ")
    print("===============================")
    print(f"Total Trades: {total_trades}")
    print(f"Total Wins: {total_wins}")
    print(f"Total Losses: {total_losses}")
    print(f"Overall Win Rate: {total_wins/(total_trades)*100:.2f}%" if total_trades>0 else "0.00%")
    print(f"Estimated Final Balance: ${total_balance:.2f}")
    
if __name__ == "__main__":
    run_backtest()

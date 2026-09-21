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
    
    # Lot size variables (approximate mapping to risk)
    # The actual bot calculates dynamic lots. For the backtest, $5 per lot round trip is $5 per standard lot ($100,000).
    # If we risk 0.2% on $5000 = $10 risk per trade.
    # $10 risk with 0.1% SL means position size is $10,000 (0.1 lot).
    # So brokerage on 0.1 lot is $0.50 per trade.
    
    for pair in pairs:
        print(f"--- Running Pessimistic Backtest for {pair} ({year_label}) ---")
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
        else:
            stop_loss_pct = 0.001
            take_profit_pct = 0.0015
            
        open_trades = []
        pair_balance = 5000.0
        wins = 0
        losses = 0
        brokerage_paid = 0.0
        
        print("Simulating Tick-by-Tick Execution...")
        df_m1_np = df_m1.reset_index().to_dict('records')
        
        for row in df_m1_np:
            current_low = row['low']
            current_high = row['high']
            
            # 1. Update Open Trades
            for trade in list(open_trades):
                hit_sl = False
                hit_tp = False
                exit_price = 0.0
                
                if trade['type'] == 'BUY':
                    if current_low <= trade['sl']:
                        hit_sl = True
                        exit_price = current_low # Actual slippage to low
                    if current_high >= trade['tp']:
                        hit_tp = True
                        if not hit_sl:
                            exit_price = trade['tp']
                        
                elif trade['type'] == 'SELL':
                    if current_high >= trade['sl']:
                        hit_sl = True
                        exit_price = current_high # Actual slippage to high
                    if current_low <= trade['tp']:
                        hit_tp = True
                        if not hit_sl:
                            exit_price = trade['tp']
                        
                if hit_sl or hit_tp:
                    # Calculate original intended lot size
                    loss_amt_planned = pair_balance * 0.002
                    sl_distance_price_planned = abs(trade['entry'] - trade['sl'])
                    position_size_usd = loss_amt_planned / (sl_distance_price_planned / trade['entry'])
                    lot_size = position_size_usd / 100000.0
                    commission = lot_size * 5.0
                    
                    if hit_sl:
                        # Calculate actual loss with slippage
                        actual_loss_distance = abs(trade['entry'] - exit_price)
                        loss_amt_actual = position_size_usd * (actual_loss_distance / trade['entry'])
                        pair_balance -= (loss_amt_actual + commission)
                        brokerage_paid += commission
                        losses += 1
                        open_trades.remove(trade)
                        
                    elif hit_tp and not hit_sl:
                        # Calculate win based on exact TP price
                        actual_win_distance = abs(trade['entry'] - exit_price)
                        win_amt_actual = position_size_usd * (actual_win_distance / trade['entry'])
                        pair_balance += (win_amt_actual - commission)
                        brokerage_paid += commission
                        wins += 1
                        open_trades.remove(trade)

            # 2. Open New Trades
            if not pd.isna(row['pred']):
                if len(open_trades) < 5:
                    if row['pred'] == 1 and row['prob'] > 0.55:
                        entry_price = row['high'] # Entry High for Longs
                        sl = entry_price * (1 - stop_loss_pct)
                        tp = entry_price * (1 + take_profit_pct)
                        open_trades.append({'type': 'BUY', 'entry': entry_price, 'sl': sl, 'tp': tp})
                    elif row['pred'] == 0 and (1 - row['prob']) > 0.55:
                        entry_price = row['low'] # Entry Low for Shorts
                        sl = entry_price * (1 + stop_loss_pct)
                        tp = entry_price * (1 - take_profit_pct)
                        open_trades.append({'type': 'SELL', 'entry': entry_price, 'sl': sl, 'tp': tp})
                        
        print(f"End Balance for {pair}: ${pair_balance:.2f}")
        print(f"Wins: {wins}, Losses: {losses}, Win Rate: {wins/(wins+losses)*100:.2f}%" if (wins+losses)>0 else "No trades.")
        print(f"Brokerage Paid: ${brokerage_paid:.2f}")
        
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

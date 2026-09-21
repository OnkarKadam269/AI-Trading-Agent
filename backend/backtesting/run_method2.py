import os
import pandas as pd
import pickle

def run_backtest():
    model_path = os.path.join(os.path.dirname(__file__), 'kronos_model.pkl')
    try:
        with open(model_path, 'rb') as f:
            kronos_model = pickle.load(f)
    except: return

    processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    pairs = ['GBPUSD', 'USDJPY', 'XAUUSD']
    year_label = '3Y'
    features = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'bb_width', 'dist_sma20']
    
    total_balance = 5000.0
    total_trades = 0
    total_wins = 0
    total_losses = 0
    
    for pair in pairs:
        print(f"--- Running METHOD 2 (Standard Execution + $5 Comm) for {pair} ---")
        csv_m15 = os.path.join(processed_dir, f'{pair}_M15_{year_label}.csv')
        csv_m1 = os.path.join(processed_dir, f'{pair}_M1_{year_label}.csv')
        
        if not os.path.exists(csv_m15) or not os.path.exists(csv_m1): continue
            
        df_m15 = pd.read_csv(csv_m15, parse_dates=['time'])
        df_m15.set_index('time', inplace=True)
        X = df_m15[features]
        df_m15['pred'] = kronos_model.predict(X)
        df_m15['prob'] = kronos_model.predict_proba(X)[:, 1]
        df_m15['signal_time'] = df_m15.index + pd.Timedelta(minutes=15)
        signals = df_m15[['signal_time', 'pred', 'prob']].set_index('signal_time')
        
        df_m1 = pd.read_csv(csv_m1, parse_dates=['time'])
        df_m1.set_index('time', inplace=True)
        df_m1 = df_m1.join(signals, how='left')
        
        if 'XAUUSD' in pair:
            stop_loss_pct, take_profit_pct = 0.002, 0.003
        else:
            stop_loss_pct, take_profit_pct = 0.001, 0.0015
            
        open_trades = []
        pair_balance = 5000.0
        wins, losses = 0, 0
        total_commissions = 0.0
        
        df_m1_np = df_m1.reset_index().to_dict('records')
        for row in df_m1_np:
            current_low, current_high = row['low'], row['high']
            
            for trade in list(open_trades):
                hit_sl, hit_tp = False, False
                
                if trade['type'] == 'BUY':
                    if current_low <= trade['sl']: hit_sl = True
                    if current_high >= trade['tp']: hit_tp = True
                        
                elif trade['type'] == 'SELL':
                    if current_high >= trade['sl']: hit_sl = True
                    if current_low <= trade['tp']: hit_tp = True
                        
                if hit_sl or hit_tp:
                    loss_amt_planned = pair_balance * 0.002
                    sl_distance_price_planned = abs(trade['entry'] - trade['sl'])
                    position_size_usd = loss_amt_planned / (sl_distance_price_planned / trade['entry'])
                    lot_size = position_size_usd / 100000.0
                    commission = lot_size * 5.0
                    
                    if hit_sl:
                        pair_balance -= (loss_amt_planned + commission)
                        total_commissions += commission
                        losses += 1
                        open_trades.remove(trade)
                    elif hit_tp and not hit_sl:
                        win_amt_planned = pair_balance * 0.002 * 1.5
                        pair_balance += (win_amt_planned - commission)
                        total_commissions += commission
                        wins += 1
                        open_trades.remove(trade)

            if not pd.isna(row['pred']) and len(open_trades) < 5:
                entry_price = row['open'] # Standard Entry (No artificial slippage)
                if row['pred'] == 1 and row['prob'] > 0.55:
                    sl, tp = entry_price * (1 - stop_loss_pct), entry_price * (1 + take_profit_pct)
                    open_trades.append({'type': 'BUY', 'entry': entry_price, 'sl': sl, 'tp': tp})
                elif row['pred'] == 0 and (1 - row['prob']) > 0.55:
                    sl, tp = entry_price * (1 + stop_loss_pct), entry_price * (1 - take_profit_pct)
                    open_trades.append({'type': 'SELL', 'entry': entry_price, 'sl': sl, 'tp': tp})
                        
        print(f"End Balance for {pair}: ${pair_balance:.2f} | Wins: {wins}, Losses: {losses} | Comm: ${total_commissions:.2f}")
        total_trades += (wins + losses)
        total_wins += wins
        total_losses += losses
        total_balance += (pair_balance - 5000.0)
        
    print("===============================\n      OVERALL METHOD 2\n===============================")
    print(f"Total Trades: {total_trades} | Win Rate: {total_wins/total_trades*100:.2f}% | Final Balance: ${total_balance:.2f}\n")
    
if __name__ == "__main__":
    run_backtest()

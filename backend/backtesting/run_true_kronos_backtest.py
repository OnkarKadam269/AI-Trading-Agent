import torch
import pandas as pd
import numpy as np
import time
import os
import sys

# Ensure backend directory is in path so we can import kronos_real.model
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models", "kronos_real"))
from model import Kronos, KronosTokenizer, KronosPredictor

def prepare_data(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Rename columns to match Kronos requirements
    df = df.rename(columns={
        'time': 'timestamps',
        'tick_volume': 'volume'
    })
    
    # Kronos needs OHLCVA. Forex doesn't have amount (turnover), so we approximate it
    if 'amount' not in df.columns:
        df['amount'] = df['volume'] * df['close']
        
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df.sort_values('timestamps').reset_index(drop=True)
    return df

def run_backtest():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "XAUUSD_M15_2023.csv")
    df = prepare_data(csv_path)
    
    # Take a smaller subset for the CPU speed test (e.g. last 1000 candles)
    subset_size = 1000
    df = df.iloc[-subset_size:].reset_index(drop=True)
    print(f"Using a subset of {len(df)} candles for the backtest.")

    print("Loading Kronos Tokenizer and Model from HuggingFace...")
    # Load tokenizer and model
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    
    # Force CPU for stability if no GPU is available, though KronosPredictor handles this.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    model = model.to(device)
    tokenizer = tokenizer.to(device)
    
    # Initialize Predictor
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    
    lookback_window = 256
    pred_len = 1
    sample_count = 5 # Reduced from 20 to 5 for CPU speed
    
    predictions = []
    actuals = []
    dates = []
    
    # We will simulate trading on the last 100 candles of our subset
    test_steps = 100
    start_idx = len(df) - test_steps - pred_len
    
    print(f"Starting inference on {device.upper()} for {test_steps} steps with {sample_count} Monte Carlo samples...")
    start_time = time.time()
    
    for i in range(start_idx, start_idx + test_steps):
        # Prepare context window
        context_start = max(0, i - lookback_window + 1)
        x_df = df.iloc[context_start:i+1][['open', 'high', 'low', 'close', 'volume', 'amount']].reset_index(drop=True)
        x_timestamp = df.iloc[context_start:i+1]['timestamps'].reset_index(drop=True)
        
        # Target timestamp
        y_timestamp = df.iloc[i+1:i+1+pred_len]['timestamps'].reset_index(drop=True)
        
        # Predict next candle
        try:
            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=1.0,          # Temperature
                top_p=0.9,      # Nucleus sampling
                sample_count=sample_count, # Monte Carlo average
                verbose=False
            )
            
            pred_close = pred_df['close'].iloc[-1]
            actual_close = df.iloc[i+1]['close']
            
            predictions.append(pred_close)
            actuals.append(actual_close)
            dates.append(df.iloc[i+1]['timestamps'])
            
            if (i - start_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Completed {i - start_idx + 1}/{test_steps} steps. Elapsed time: {elapsed:.2f}s")
                
        except Exception as e:
            print(f"Error at index {i}: {e}")
            break
            
    total_time = time.time() - start_time
    print(f"Inference complete! Total time: {total_time:.2f}s (Average: {total_time/test_steps:.2f}s per step)")
    
    # Calculate simple trading metrics
    results_df = pd.DataFrame({
        'time': dates,
        'actual_close': actuals,
        'predicted_close': predictions
    })
    
    # Strategy: if predicted close is higher than current close + 0.5 points, buy.
    # We compare predicted_close(t+1) with actual_close(t)
    results_df['prev_actual'] = results_df['actual_close'].shift(1)
    results_df = results_df.dropna()
    
    results_df['signal'] = 0
    # Buy if predicted to go up by 0.5 points (approx 5 pips on gold)
    results_df.loc[results_df['predicted_close'] > results_df['prev_actual'] + 0.5, 'signal'] = 1
    # Sell if predicted to go down by 0.5 points
    results_df.loc[results_df['predicted_close'] < results_df['prev_actual'] - 0.5, 'signal'] = -1
    
    # Calculate returns (using actual future returns)
    results_df['actual_return'] = results_df['actual_close'] - results_df['prev_actual']
    results_df['strategy_pnl'] = results_df['signal'] * results_df['actual_return']
    
    # Assume 0.2 spread/slippage cost per trade
    results_df['cost'] = (results_df['signal'] != 0).astype(int) * 0.2
    results_df['net_pnl'] = results_df['strategy_pnl'] - results_df['cost']
    results_df['cumulative_pnl'] = results_df['net_pnl'].cumsum()
    
    win_rate = (results_df[results_df['signal'] != 0]['net_pnl'] > 0).mean()
    total_trades = (results_df['signal'] != 0).sum()
    total_pnl = results_df['cumulative_pnl'].iloc[-1] if len(results_df) > 0 else 0
    
    print("\n--- Backtest Results (100 Steps) ---")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Net PnL (Points): {total_pnl:.2f}")
    
    out_path = os.path.join(os.path.dirname(__file__), "kronos_true_backtest_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"Detailed results saved to {out_path}")

if __name__ == "__main__":
    run_backtest()

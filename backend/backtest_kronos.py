import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pickle
import ta
import logging
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "USDCHF"]
TIMEFRAME = mt5.TIMEFRAME_M15
CANDLES = 2000  # Backtest on the last 2000 candles (approx 1 month)

def get_real_symbol(base_symbol):
    symbols = mt5.symbols_get()
    if symbols is None: return base_symbol
    for s in symbols:
        if base_symbol in s.name:
            return s.name
    return base_symbol

def get_data(symbol):
    real_symbol = get_real_symbol(symbol)
    rates = mt5.copy_rates_from_pos(real_symbol, TIMEFRAME, 0, CANDLES)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def add_features(df):
    if df.empty: return df
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_width'] = df['bb_high'] - df['bb_low']
    df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
    df['dist_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    df.dropna(inplace=True)
    return df

def run_backtest():
    print("\n" + "="*60)
    print("🚀 KRONOS AI DEEP BACKTEST REPORT (OUT-OF-SAMPLE)")
    print("="*60)
    
    if not mt5.initialize():
        print("MT5 initialization failed!")
        return
        
    try:
        with open("kronos_model.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("Model not found! Please run train_kronos.py first.")
        return

    features = ['rsi', 'bb_width', 'dist_sma20']
    
    total_trades = 0
    total_wins = 0
    total_losses = 0
    
    # We will simulate a fixed Risk:Reward Ratio of 1:1.5
    # If prediction is 1 (UP), and future_close > close, it's a win (+1.5R)
    # If prediction is 0 (DOWN), and future_close < close, it's a win (+1.5R)
    RRR = 1.5 
    
    print(f"{'PAIR':<10} | {'TRADES':<8} | {'WIN RATE':<10} | {'ACCURACY':<10} | {'EST. PROFIT (R)':<15}")
    print("-" * 60)

    for symbol in PAIRS:
        df = get_data(symbol)
        if df.empty:
            continue
            
        df = add_features(df)
        if df.empty:
            continue
            
        # Get ground truth
        df['future_close'] = df['close'].shift(-5)
        df.dropna(inplace=True)
        
        features = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'bb_width', 'dist_sma20']
        X = df[features]
        predictions = model.predict(X)
        df['prediction'] = predictions
        
        wins = 0
        losses = 0
        
        for index, row in df.iterrows():
            if row['prediction'] == 1: # AI says Buy
                if row['future_close'] > row['close']:
                    wins += 1
                else:
                    losses += 1
            else: # AI says Sell
                if row['future_close'] < row['close']:
                    wins += 1
                else:
                    losses += 1
                    
        pair_trades = wins + losses
        if pair_trades == 0: continue
        
        win_rate = (wins / pair_trades) * 100
        # Simulated profit in terms of R (Risk Units). Win = +1.5R, Loss = -1R
        profit_r = (wins * RRR) - (losses * 1.0)
        
        total_trades += pair_trades
        total_wins += wins
        total_losses += losses
        
        print(f"{symbol:<10} | {pair_trades:<8} | {win_rate:>6.2f}%   | {win_rate:>6.2f}%   | {profit_r:>+10.2f} R")

    print("-" * 60)
    
    if total_trades > 0:
        overall_win_rate = (total_wins / total_trades) * 100
        overall_profit_r = (total_wins * RRR) - (total_losses * 1.0)
        print(f"OVERALL    | {total_trades:<8} | {overall_win_rate:>6.2f}%   | {overall_win_rate:>6.2f}%   | {overall_profit_r:>+10.2f} R")
        print("\nNote: 'EST. PROFIT (R)' is calculated using a 1:1.5 Risk-to-Reward Ratio.")
        print("Positive R means the model is statistically profitable over the sample period.")
    else:
        print("Not enough data to calculate overall stats.")
        
    print("="*60 + "\n")
    mt5.shutdown()

if __name__ == "__main__":
    run_backtest()

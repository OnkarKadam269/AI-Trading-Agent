import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pickle
import ta
import logging
import warnings
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - KRONOS BACKTEST COACH - %(message)s')

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "USDCHF"]
TIMEFRAME = mt5.TIMEFRAME_M15
CANDLES = 5000  # Look back across the last 5000 candles to find historical losses

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
RULES_FILE = os.path.join(os.path.dirname(__file__), 'coach_rules.json')

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

def seed_coach():
    if not GEMINI_API_KEY:
        logging.error("GEMINI_API_KEY not found in .env!")
        return

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-pro-exp-02-05')

    logging.info("Initializing MT5 to download historical data for Coach Seeding...")
    if not mt5.initialize():
        logging.error("MT5 initialization failed!")
        return
        
    try:
        with open(os.path.join(os.path.dirname(__file__), "kronos_model.pkl"), "rb") as f:
            xgb_model = pickle.load(f)
    except FileNotFoundError:
        logging.error("Model not found! Please run train_kronos.py first.")
        return

    losing_trades = []

    for symbol in PAIRS:
        df = get_data(symbol)
        if df.empty: continue
            
        df = add_features(df)
        if df.empty: continue
            
        # Get ground truth
        df['future_close'] = df['close'].shift(-5)
        df.dropna(inplace=True)
        
        features = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'bb_width', 'dist_sma20']
        X = df[features]
        predictions = xgb_model.predict(X)
        probabilities = xgb_model.predict_proba(X)[:, 1]
        
        df['prediction'] = predictions
        df['probability'] = probabilities
        
        for index, row in df.iterrows():
            is_loss = False
            direction = ""
            
            # Simulate the 60/40 rule
            if row['probability'] > 0.60:
                direction = "BUY"
                if row['future_close'] < row['close']:
                    is_loss = True
            elif row['probability'] < 0.40:
                direction = "SELL"
                if row['future_close'] > row['close']:
                    is_loss = True
                    
            if is_loss:
                losing_trades.append({
                    "pair": symbol,
                    "direction": direction,
                    "probability": float(row['probability']),
                    "indicators": {
                        "rsi": float(row['rsi']),
                        "bb_width": float(row['bb_width']),
                        "dist_sma20": float(row['dist_sma20'])
                    }
                })

    mt5.shutdown()
    
    if len(losing_trades) == 0:
        logging.info("No losing trades found in the backtest! Model is perfect.")
        return

    # To save Gemini context window, we only take the 50 worst/most confident losses
    # A confident loss is where the probability was very high or very low, but it still lost
    losing_trades.sort(key=lambda x: abs(x['probability'] - 0.5), reverse=True)
    worst_losses = losing_trades[:50]
    
    logging.info(f"Collected {len(worst_losses)} high-confidence historical losses.")
    logging.info("Sending historical losses to Gemini AI to generate initial Avoidance Rules...")
    
    prompt = f"""
    You are an elite quantitative trading coach. 
    I just ran a backtest of our XGBoost trading algorithm on the last 5000 15-minute candles.
    Here are the 50 worst losing trades where the AI was highly confident but still lost:
    
    {json.dumps(worst_losses, indent=2)}
    
    Identify specific mathematical patterns in the technical indicators (RSI, Bollinger Band Width, Distance from SMA) that caused these historical losses.
    
    Output a JSON array of "Avoidance Rules" that we can use to veto future trades before they happen.
    Output EXACTLY this format and nothing else:
    [
      {{
        "rule_name": "High RSI Reversal Failure",
        "condition_python": "features['rsi'] > 70 and features['bb_width'] < 0.002",
        "reason": "When RSI is overbought but volatility is low, the breakout usually fakes out."
      }}
    ]
    Make sure the condition_python uses the dictionary `features` and uses valid python syntax (e.g. `features['rsi']`).
    """
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
            
        rules = json.loads(text.strip())
        
        # Merge with existing rules if they exist
        existing_rules = []
        if os.path.exists(RULES_FILE):
            with open(RULES_FILE, 'r') as f:
                try:
                    existing_rules = json.load(f)
                except:
                    pass
                    
        existing_rules.extend(rules)
        
        with open(RULES_FILE, 'w') as f:
            json.dump(existing_rules, f, indent=4)
            
        logging.info(f"Coach Successfully Seeded! {len(rules)} historical Avoidance Rules have been permanently added to the AI's Brain.")
        
    except Exception as e:
        logging.error(f"Failed to seed coach: {e}")

if __name__ == "__main__":
    seed_coach()

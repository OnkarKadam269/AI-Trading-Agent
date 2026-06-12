import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import ta
import time
import logging
import requests
import os
from datetime import datetime
import warnings
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# RunPod Configuration
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

if not RUNPOD_ENDPOINT_ID or not RUNPOD_API_KEY:
    raise ValueError("Missing RUNPOD_ENDPOINT_ID or RUNPOD_API_KEY in environment variables (.env)")

PAIRS = ['EURUSD', 'GBPUSD', 'XAUUSD', 'USDJPY', 'AUDUSD']
TIMEFRAME = mt5.TIMEFRAME_M15
LOT_SIZE = 0.01

def init_mt5():
    if not mt5.initialize():
        logging.error("initialize() failed")
        mt5.shutdown()
        return False
    logging.info("MT5 Initialized Successfully")
    return True

def get_data(symbol, n_candles=100):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, n_candles)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def calculate_features(df):
    try:
        # Calculate indicators identically to training
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_width'] = bb.bollinger_wband()
        
        sma20 = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['dist_sma20'] = (df['close'] - sma20) / sma20 * 100
        
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error calculating features: {e}")
        return None

def ask_kronos_brain(features_dict):
    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": features_dict
    }
    try:
        response = requests.post(url, headers=headers, json=payload).json()
        if 'output' in response and response['output']['status'] == 'success':
            return response['output']['prediction'], response['output']['probability']
        else:
            logging.error(f"RunPod Error: {response}")
            return None, None
    except Exception as e:
        logging.error(f"API Error: {e}")
        return None, None

def place_trade(symbol, prediction, probability, current_price):
    # Check if we already have an open position for this symbol
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        logging.error(f"Failed to get positions for {symbol}")
        return
        
    if len(positions) > 0:
        logging.info(f"Already in a trade for {symbol}. Waiting for it to close.")
        return

    # Calculate 1:1.5 RRR
    # Using a simple fixed percentage for SL/TP as an example
    # For a real bot, ATR is highly recommended
    stop_loss_pct = 0.002 # 0.2%
    take_profit_pct = stop_loss_pct * 1.5 # 0.3%

    point = mt5.symbol_info(symbol).point
    
    if prediction == 1: # BUY
        sl = current_price * (1 - stop_loss_pct)
        tp = current_price * (1 + take_profit_pct)
        order_type = mt5.ORDER_TYPE_BUY
        logging.info(f"[{symbol}] KRONOS SAYS BUY! (Prob: {probability:.2f})")
    else: # SELL
        sl = current_price * (1 + stop_loss_pct)
        tp = current_price * (1 - take_profit_pct)
        order_type = mt5.ORDER_TYPE_SELL
        logging.info(f"[{symbol}] KRONOS SAYS SELL! (Prob: {1-probability:.2f})")

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": LOT_SIZE,
        "type": order_type,
        "price": mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 234000,
        "comment": "Kronos AI",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Order failed for {symbol}: {result.comment}")
    else:
        logging.info(f"Order SUCCESS for {symbol}! Ticket: {result.order}")


def run_bot():
    if not init_mt5():
        return

    logging.info(f"KRONOS AI BOT STARTED! Connected to RunPod: {RUNPOD_ENDPOINT_ID}")
    
    while True:
        # Run every 15 minutes
        current_time = datetime.now()
        logging.info(f"Scanning markets at {current_time.strftime('%H:%M:%S')}...")
        
        for pair in PAIRS:
            # 1. Fetch data
            df = get_data(pair)
            if df is None:
                continue
                
            # 2. Calculate technicals
            df = calculate_features(df)
            if df is None or len(df) == 0:
                continue
                
            # 3. Get the most recently closed candle (index -2)
            # Index -1 is the currently forming, unfinished candle!
            latest_candle = df.iloc[-2] 
            
            # 4. Prepare features for the brain
            features_dict = {
                'open': float(latest_candle['open']),
                'high': float(latest_candle['high']),
                'low': float(latest_candle['low']),
                'close': float(latest_candle['close']),
                'tick_volume': float(latest_candle['tick_volume']),
                'rsi': float(latest_candle['rsi']),
                'bb_width': float(latest_candle['bb_width']),
                'dist_sma20': float(latest_candle['dist_sma20'])
            }
            
            # 5. Ask Kronos Cloud Brain
            prediction, probability = ask_kronos_brain(features_dict)
            
            if prediction is not None:
                # 6. Execute Trade based on probability confidence
                # We only trade if probability is > 60% or < 40% to avoid noisy markets
                if probability > 0.60:
                    place_trade(pair, 1, probability, float(latest_candle['close']))
                elif probability < 0.40:
                    place_trade(pair, 0, probability, float(latest_candle['close']))
                else:
                    logging.info(f"[{pair}] Market is too noisy (Prob: {probability:.2f}). Sitting out.")
            
        logging.info("Sleeping for 15 minutes until next candle closes...")
        # Sleep for exactly 15 minutes (900 seconds)
        time.sleep(900)

if __name__ == "__main__":
    run_bot()

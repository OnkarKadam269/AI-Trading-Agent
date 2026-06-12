import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import ta
import time
import logging
import requests
import os
from datetime import datetime, timedelta
import warnings
from dotenv import load_dotenv
import pickle

load_dotenv()

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Credentials
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not RUNPOD_ENDPOINT_ID or not RUNPOD_API_KEY:
    raise ValueError("Missing RUNPOD credentials in .env")

PAIRS = ['EURUSDm', 'GBPUSDm', 'XAUUSDm', 'USDJPYm', 'AUDUSDm']
TIMEFRAME = mt5.TIMEFRAME_M15
LOT_SIZE = 0.01

# Load Local AI Model
model_path = os.path.join(os.path.dirname(__file__), 'kronos_model.pkl')
try:
    with open(model_path, 'rb') as f:
        kronos_model = pickle.load(f)
    logging.info("Kronos AI Model loaded locally successfully!")
except Exception as e:
    raise ValueError(f"Failed to load AI model: {e}")

# System Tracking
system_errors = []
trades_taken_this_session = 0

def send_telegram(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        logging.error(f"Failed to send Telegram message: {e}")

def log_error(msg):
    logging.error(msg)
    time_str = datetime.now().strftime("%H:%M")
    system_errors.append(f"[{time_str}] {msg}")

def init_mt5():
    if not mt5.initialize():
        log_error("MT5 initialize() failed. Check if MetaTrader is open.")
        mt5.shutdown()
        return False
    logging.info("MT5 Initialized Successfully")
    return True

def get_data(symbol, n_candles=100):
    if not mt5.symbol_select(symbol, True):
        log_error(f"Failed to select {symbol} in MT5")
        return None
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, n_candles)
    if rates is None or len(rates) == 0:
        log_error(f"Failed to fetch data for {symbol}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def calculate_features(df):
    try:
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_width'] = bb.bollinger_wband()
        sma20 = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['dist_sma20'] = (df['close'] - sma20) / sma20 * 100
        df.dropna(inplace=True)
        return df
    except Exception as e:
        log_error(f"Error calculating features: {e}")
        return None

def ask_kronos_brain(features_dict):
    try:
        df = pd.DataFrame([features_dict])
        features = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'bb_width', 'dist_sma20']
        X = df[features]
        
        prediction = int(kronos_model.predict(X)[0])
        probability = float(kronos_model.predict_proba(X)[0][1])
        
        return prediction, probability
    except Exception as e:
        log_error(f"Local AI Error: {e}")
        return None, None

def place_trade(symbol, prediction, probability, current_price):
    global trades_taken_this_session
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        log_error(f"Failed to get open positions for {symbol}")
        return
        
    if len(positions) > 0:
        logging.info(f"Already in a trade for {symbol}. Waiting for it to close.")
        return

    stop_loss_pct = 0.002 # 0.2%
    take_profit_pct = 0.003 # 0.3% (1:1.5 RRR)
    
    if prediction == 1:
        sl = current_price * (1 - stop_loss_pct)
        tp = current_price * (1 + take_profit_pct)
        order_type = mt5.ORDER_TYPE_BUY
        logging.info(f"[{symbol}] KRONOS SAYS BUY! (Prob: {probability:.2f})")
    else:
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
        log_error(f"Order failed for {symbol}: {result.comment}")
    else:
        trades_taken_this_session += 1
        msg = f"🟢 <b>KRONOS EXECUTED TRADE</b>\n\n<b>Pair:</b> {symbol}\n<b>Action:</b> {'BUY' if prediction == 1 else 'SELL'}\n<b>Confidence:</b> {probability:.2f}\n<b>Price:</b> {current_price}\n<b>TP/SL:</b> Secured on Broker Server"
        send_telegram(msg)
        logging.info(f"Order SUCCESS for {symbol}! Ticket: {result.order}")

def generate_4h_report():
    global system_errors, trades_taken_this_session
    account_info = mt5.account_info()
    balance = account_info.balance if account_info else "Unknown"
    equity = account_info.equity if account_info else "Unknown"
    
    report = f"📊 <b>KRONOS 4-HOUR PERFORMANCE REPORT</b> 📊\n\n"
    report += f"💰 <b>Balance:</b> ${balance}\n"
    report += f"📈 <b>Equity:</b> ${equity}\n"
    report += f"🎯 <b>Trades Taken:</b> {trades_taken_this_session}\n\n"
    
    if len(system_errors) > 0:
        report += f"⚠️ <b>System Breakdowns ({len(system_errors)}):</b>\n"
        for err in system_errors[-5:]: # Only show last 5
            report += f"- {err}\n"
    else:
        report += "✅ <b>System Health:</b> 100% Perfect (No Breakdowns)\n"
        
    send_telegram(report)
    
    # Reset tracking
    system_errors = []
    trades_taken_this_session = 0

def run_agent():
    if not init_mt5():
        return

    logging.info(f"KRONOS AI AGENT STARTED! Running Locally for Instant Execution.")
    send_telegram("🚀 <b>Kronos AI Agent is officially ONLINE!</b>\nRunning Locally for 0.01s Instant Execution.")
    
    last_report_time = datetime.now()
    
    while True:
        current_time = datetime.now()
        logging.info(f"Scanning markets at {current_time.strftime('%H:%M:%S')}...")
        
        for pair in PAIRS:
            df = get_data(pair)
            if df is None: continue
            df = calculate_features(df)
            if df is None or len(df) == 0: continue
            
            latest_candle = df.iloc[-2] 
            features_dict = {
                'open': float(latest_candle['open']), 'high': float(latest_candle['high']),
                'low': float(latest_candle['low']), 'close': float(latest_candle['close']),
                'tick_volume': float(latest_candle['tick_volume']), 'rsi': float(latest_candle['rsi']),
                'bb_width': float(latest_candle['bb_width']), 'dist_sma20': float(latest_candle['dist_sma20'])
            }
            
            prediction, probability = ask_kronos_brain(features_dict)
            
            if prediction is not None:
                if probability > 0.60:
                    place_trade(pair, 1, probability, float(latest_candle['close']))
                elif probability < 0.40:
                    place_trade(pair, 0, probability, float(latest_candle['close']))
                else:
                    logging.info(f"[{pair}] Market is too noisy (Prob: {probability:.2f}). Sitting out.")
            
        # Check if 4 hours have passed for the report
        if (datetime.now() - last_report_time).total_seconds() >= 4 * 3600:
            generate_4h_report()
            last_report_time = datetime.now()
            
        # Calculate seconds until the exact next 15-minute candle closes
        now = datetime.now()
        next_minute = ((now.minute // 15) + 1) * 15
        next_run = now.replace(minute=0, second=2, microsecond=0) + timedelta(minutes=next_minute)
        sleep_seconds = (next_run - now).total_seconds()
        
        if sleep_seconds <= 0:
            sleep_seconds = 900
            
        logging.info(f"Sleeping for {int(sleep_seconds)} seconds until exact candle close at {next_run.strftime('%H:%M:%S')}...")
        time.sleep(sleep_seconds)

if __name__ == "__main__":
    run_agent()

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
import sys
import json
sys.path.append(os.path.dirname(__file__))
from data.database import SessionLocal, Trade

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

PAIRS = ['GBPUSDm', 'XAUUSDm', 'USDJPYm', 'AUDUSDm']
TIMEFRAME = mt5.TIMEFRAME_M15
RISK_PER_TRADE_PCT = 0.002

# Load Local AI Model
model_path = os.path.join(os.path.dirname(__file__), 'kronos_model.pkl')
try:
    with open(model_path, 'rb') as f:
        kronos_model = pickle.load(f)
    logging.info("Kronos AI Model loaded locally successfully!")
except Exception as e:
    raise ValueError(f"Failed to load AI model: {e}")

# Persistent System Tracking
STATE_FILE = os.path.join(os.path.dirname(__file__), "reporter_state.json")

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"last_report_time": 0.0, "errors": [], "trades": 0}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

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
    state = load_state()
    state["errors"].append(f"[{time_str}] {msg}")
    save_state(state)

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

def calculate_position_size(symbol, current_price, sl_price):
    try:
        account = mt5.account_info()
        if account is None:
            log_error("Failed to fetch account info for position sizing")
            return 0.01 # Fallback
            
        equity = account.equity
        risk_amount = equity * RISK_PER_TRADE_PCT
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return 0.01
            
        tick_size = symbol_info.trade_tick_size
        tick_value = symbol_info.trade_tick_value
        
        if tick_size == 0 or tick_value == 0:
            return 0.01
            
        sl_distance_points = abs(current_price - sl_price) / tick_size
        
        if sl_distance_points == 0:
            return 0.01
            
        raw_lot_size = risk_amount / (sl_distance_points * tick_value)
        
        vol_step = symbol_info.volume_step
        min_vol = symbol_info.volume_min
        max_vol = symbol_info.volume_max
        
        rounded_lot = round(raw_lot_size / vol_step) * vol_step
        final_lot = max(min_vol, min(rounded_lot, max_vol))
        
        return float(round(final_lot, 2))
    except Exception as e:
        log_error(f"Error calculating position size: {e}")
        return 0.01

def place_trade(symbol, prediction, probability, current_price, features_dict):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        log_error(f"Failed to get open positions for {symbol}")
        return
        
    if len(positions) > 0:
        logging.info(f"Already in a trade for {symbol}. Waiting for it to close.")
        return

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log_error(f"Failed to get tick for {symbol}")
        return
        
    ask = tick.ask
    bid = tick.bid

    if symbol == 'XAUUSDm':
        stop_loss_pct = 0.002 # 0.2%
        take_profit_pct = 0.003 # 0.3% (1:1.5 RRR)
    else:
        stop_loss_pct = 0.001 # 0.1%
        take_profit_pct = 0.0015 # 0.15% (1:1.5 RRR)
    
    if prediction == 1:
        order_price = ask
        sl = order_price * (1 - stop_loss_pct)
        tp = order_price * (1 + take_profit_pct)
        order_type = mt5.ORDER_TYPE_BUY
        logging.info(f"[{symbol}] KRONOS SAYS BUY! (Prob: {probability:.2f})")
    else:
        order_price = bid
        sl = order_price * (1 + stop_loss_pct)
        tp = order_price * (1 - take_profit_pct)
        order_type = mt5.ORDER_TYPE_SELL
        logging.info(f"[{symbol}] KRONOS SAYS SELL! (Prob: {1-probability:.2f})")

    # Dynamic Volatility-Adjusted Lot Size
    calculated_lot_size = calculate_position_size(symbol, order_price, sl)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": calculated_lot_size,
        "type": order_type,
        "price": order_price,
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
        state = load_state()
        state["trades"] = state.get("trades", 0) + 1
        save_state(state)
        # Save to Memory System (DB)
        try:
            db = SessionLocal()
            new_trade = Trade(
                ticket=result.order,
                symbol=symbol,
                direction='BUY' if prediction == 1 else 'SELL',
                open_time=datetime.utcnow(),
                entry_price=current_price,
                stop_loss=sl,
                take_profit_1=tp,
                take_profit_2=tp,
                lot_size=calculated_lot_size,
                confidence=probability,
                reasoning=features_dict
            )
            db.add(new_trade)
            db.commit()
            db.close()
            logging.info(f"Trade successfully recorded into Memory System.")
        except Exception as e:
            log_error(f"Failed to save trade to DB: {e}")

        msg = f"🟢 <b>KRONOS EXECUTED TRADE</b>\n\n<b>Pair:</b> {symbol}\n<b>Action:</b> {'BUY' if prediction == 1 else 'SELL'}\n<b>Confidence:</b> {probability:.2f}\n<b>Price:</b> {current_price}\n<b>TP/SL:</b> Secured on Broker Server"
        send_telegram(msg)
        logging.info(f"Order SUCCESS for {symbol}! Ticket: {result.order}")

def check_closed_trades():
    try:
        db = SessionLocal()
        # Find trades that haven't been marked as closed in our DB
        open_trades = db.query(Trade).filter(Trade.exit_price == None).all()
        if not open_trades:
            db.close()
            return
            
        # Get history from yesterday to tomorrow
        from_date = datetime.now() - timedelta(days=2)
        to_date = datetime.now() + timedelta(days=1)
        history = mt5.history_deals_get(from_date, to_date)
        
        if history is None:
            db.close()
            return
            
        closed_any = False
        for trade in open_trades:
            # Look for a closing deal for this position
            for deal in history:
                if deal.position_id == trade.ticket and deal.entry == mt5.DEAL_ENTRY_OUT:
                    trade.exit_price = deal.price
                    trade.pnl = deal.profit
                    trade.close_time = datetime.utcnow()
                    trade.close_reason = "TP/SL/Manual"
                    db.commit()
                    logging.info(f"Memory System Updated: Trade {trade.ticket} Closed. PnL: ${deal.profit:.2f}")
                    closed_any = True
                    break
        db.close()
        

    except Exception as e:
        log_error(f"Error checking closed trades: {e}")

def generate_4h_report():
    state = load_state()
    account_info = mt5.account_info()
    balance = account_info.balance if account_info else "Unknown"
    equity = account_info.equity if account_info else "Unknown"
    
    report = f"📊 <b>KRONOS 4-HOUR PERFORMANCE REPORT</b> 📊\n\n"
    report += f"💰 <b>Balance:</b> ${balance}\n"
    report += f"📈 <b>Equity:</b> ${equity}\n"
    report += f"🎯 <b>Trades Taken:</b> {state.get('trades', 0)}\n\n"
    
    errors = state.get("errors", [])
    if len(errors) > 0:
        report += f"⚠️ <b>System Breakdowns ({len(errors)}):</b>\n"
        for err in errors[-5:]: # Only show last 5
            report += f"- {err}\n"
    else:
        report += "✅ <b>System Health:</b> 100% Perfect (No Breakdowns)\n"
        
    send_telegram(report)
    
    # Reset tracking
    state["errors"] = []
    state["trades"] = 0
    state["last_report_time"] = datetime.now().timestamp()
    save_state(state)

def run_agent():
    if not init_mt5():
        return

    logging.info(f"KRONOS AI AGENT STARTED! Running Locally for Instant Execution.")
    send_telegram("🚀 <b>Kronos AI Agent is officially ONLINE!</b>\nRunning Locally for 0.01s Instant Execution.")
    
    state = load_state()
    if state.get("last_report_time", 0.0) == 0.0:
        state["last_report_time"] = datetime.now().timestamp()
        save_state(state)
    
    while True:
        current_time = datetime.now()
        utc_now = datetime.utcnow()
        
        # Market is globally closed on Saturdays, and most of Sunday (until 21:00 UTC)
        if utc_now.weekday() == 5 or (utc_now.weekday() == 6 and utc_now.hour < 21):
            logging.info("Weekend detected. Markets are closed. Sleeping for 1 hour...")
            time.sleep(3600)
            continue
            
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        is_rollover = (2 <= ist_now.hour < 5)
        
        if is_rollover:
            logging.info(f"High Spread Rollover Period ({ist_now.strftime('%H:%M')} IST). Skipping market scan.")
            pairs_to_scan = []
        else:
            logging.info(f"Scanning markets at {current_time.strftime('%H:%M:%S')}...")
            pairs_to_scan = PAIRS
        
        signals = []
        for pair in pairs_to_scan:
            df = get_data(pair)
            if df is None: continue
            df = calculate_features(df)
            if df is None or len(df) == 0: continue
            
            latest_candle = df.iloc[-1] 
            features_dict = {
                'open': float(latest_candle['open']), 'high': float(latest_candle['high']),
                'low': float(latest_candle['low']), 'close': float(latest_candle['close']),
                'tick_volume': float(latest_candle['tick_volume']), 'rsi': float(latest_candle['rsi']),
                'bb_width': float(latest_candle['bb_width']), 'dist_sma20': float(latest_candle['dist_sma20'])
            }
            
            prediction, probability = ask_kronos_brain(features_dict)
            
            if prediction is not None:
                if probability > 0.60:
                    signals.append((pair, 1, probability, float(latest_candle['close']), features_dict))
                elif probability < 0.40:
                    signals.append((pair, 0, probability, float(latest_candle['close']), features_dict))
                else:
                    logging.info(f"[{pair}] Market is too noisy (Prob: {probability:.2f}). Sitting out.")
            
        if signals:
            logging.info(f"Pre-computed {len(signals)} trade signals! Waiting for the exact 00.00 clock strike...")
            target_time = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=((current_time.minute // 15) + 1) * 15)
            # Sleep in tiny bursts until the exact second hits 00
            while datetime.now() < target_time:
                time.sleep(0.01)
                
            logging.info(f"CLOCK STRUCK 00! FIRING {len(signals)} TRADES CONCURRENTLY!")
            for sig in signals:
                place_trade(sig[0], sig[1], sig[2], sig[3], sig[4])
                
        # Check if 4 hours have passed for the report
        state = load_state()
        if (datetime.now().timestamp() - state.get("last_report_time", 0.0)) >= 4 * 3600:
            generate_4h_report()
            
        # Update Memory System with any closed trades
        check_closed_trades()

        # Calculate seconds to wake up EXACTLY 2 seconds before the next 15-minute candle closes
        now = datetime.now()
        next_minute = ((now.minute // 15) + 1) * 15
        next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=next_minute) - timedelta(seconds=2)
        sleep_seconds = (next_run - now).total_seconds()
        
        if sleep_seconds <= 0:
            sleep_seconds = 898 # 15 minutes minus 2 seconds
            
        logging.info(f"Sleeping for {int(sleep_seconds)} seconds... Waking up at {next_run.strftime('%H:%M:%S')} to pre-compute.")
        time.sleep(sleep_seconds)

if __name__ == "__main__":
    run_agent()

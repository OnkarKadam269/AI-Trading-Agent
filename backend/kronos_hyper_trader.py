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
import sys
import json
import torch
import news_filter

sys.path.append(os.path.dirname(__file__))
from data.database import SessionLocal, Trade
sys.path.append(os.path.join(os.path.dirname(__file__), "kronos_real"))
from model import Kronos, KronosTokenizer, KronosPredictor

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

PAIRS = ['XAUUSDm']
TIMEFRAME = mt5.TIMEFRAME_M30
RISK_PER_TRADE_PCT = 0.002

# Load True HuggingFace AI Model Globally
try:
    logging.info("Loading true PyTorch Kronos Model (24M Parameters)...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    tokenizer = tokenizer.to(device)
    kronos_predictor = KronosPredictor(model, tokenizer, max_context=512)
    logging.info(f"Kronos AI Model loaded successfully on {device.upper()}!")
except Exception as e:
    raise ValueError(f"Failed to load HuggingFace AI model: {e}")

# Persistent System Tracking
STATE_FILE = os.path.join(os.path.dirname(__file__), "hyper_state.json")

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

def get_data(symbol, n_candles=300):
    if not mt5.symbol_select(symbol, True):
        log_error(f"Failed to select {symbol} in MT5")
        return None
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, n_candles)
    if rates is None or len(rates) == 0:
        log_error(f"Failed to fetch data for {symbol}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={'tick_volume': 'volume'})
    if 'amount' not in df.columns and 'volume' in df.columns and 'close' in df.columns:
        df['amount'] = df['volume'] * df['close']
    df.set_index('time', inplace=True)
    return df

def calculate_features(df):
    try:
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_width'] = bb.bollinger_wband()
        sma20 = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['dist_sma20'] = (df['close'] - sma20) / sma20 * 100
        
        # New Macro Features
        df['tr0'] = df['high'] - df['low']
        df['tr1'] = abs(df['high'] - df['close'].shift())
        df['tr2'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()
        
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # ADX Calculation
        period = 14
        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']
        df['+dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['-dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        df['+di'] = 100 * (df['+dm'].rolling(period).mean() / df['atr'])
        df['-di'] = 100 * (df['-dm'].rolling(period).mean() / df['atr'])
        df['dx'] = 100 * abs(df['+di'] - df['-di']) / (df['+di'] + df['-di'])
        df['adx'] = df['dx'].rolling(period).mean()
        
        df.dropna(inplace=True)
        return df
    except Exception as e:
        log_error(f"Error calculating features: {e}")
        return None

def ask_kronos_brain(df_slice):
    try:
        x_df = df_slice[['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
        x_timestamp = df_slice.index.copy()
        
        last_time = x_timestamp[-1]
        future_time = last_time + timedelta(minutes=30)
        y_timestamp = pd.Series([future_time])
        
        pred_df_list = kronos_predictor.predict_batch(
            df_list=[x_df],
            x_timestamp_list=[x_timestamp],
            y_timestamp_list=[y_timestamp],
            pred_len=1,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=False
        )
        
        pred_close = pred_df_list[0]['close'].iloc[-1]
        current_close = x_df['close'].iloc[-1]
        return pred_close, current_close
        
    except Exception as e:
        log_error(f"True AI Error: {e}")
        return None, None

def calculate_position_size(symbol, current_price, sl_price):
    try:
        account = mt5.account_info()
        if account is None:
            log_error("Failed to fetch account info for position sizing")
            return 0.01 
            
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

def place_trade(symbol, prediction, pred_close, current_price, gap, atr):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log_error(f"Failed to get tick for {symbol}")
        return
        
    ask = tick.ask
    bid = tick.bid
    
    sl_dist = 1.5 * atr

    if prediction == 1:
        order_price = ask
        sl = order_price - sl_dist
        tp = 0.0 # Uncapped
        order_type = mt5.ORDER_TYPE_BUY
        logging.info(f"[{symbol}] KRONOS SAYS BUY! (Pred: {pred_close:.2f} | Gap: {gap:.2f})")
    else:
        order_price = bid
        sl = order_price + sl_dist
        tp = 0.0 # Uncapped
        order_type = mt5.ORDER_TYPE_SELL
        logging.info(f"[{symbol}] KRONOS SAYS SELL! (Pred: {pred_close:.2f} | Gap: {gap:.2f})")

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
        "magic": 999000,
        "comment": "Kronos Hyper",
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
                confidence=float(gap),
                reasoning=f"Pred: {pred_close:.2f}, Gap: {gap:.2f}, ATR: {atr:.2f}"
            )
            db.add(new_trade)
            db.commit()
            db.close()
            logging.info(f"Trade successfully recorded into Memory System.")
        except Exception as e:
            log_error(f"Failed to save trade to DB: {e}")

        msg = f"🔥 <b>KRONOS HYPER EXECUTED 30M HYBRID TRADE</b>\n\n<b>Pair:</b> {symbol}\n<b>Action:</b> {'BUY' if prediction == 1 else 'SELL'}\n<b>Pred Close:</b> {pred_close:.2f}\n<b>Price:</b> {current_price}\n<b>TP/SL:</b> Uncapped TP / 1.5 ATR Trailing"
        send_telegram(msg)
        logging.info(f"Order SUCCESS for {symbol}! Ticket: {result.order}")

def manage_trailing_stops():
    try:
        positions = mt5.positions_get(symbol="XAUUSDm")
        if positions is None or len(positions) == 0:
            return
            
        df = get_data("XAUUSDm", n_candles=100)
        if df is None: return
        df = calculate_features(df)
        if df is None: return
        
        latest_candle = df.iloc[-1]
        atr = float(latest_candle['atr'])
        current_close = float(latest_candle['close'])
        
        for pos in positions:
            ticket = pos.ticket
            pos_type = pos.type
            current_sl = pos.sl
            
            if pos_type == mt5.ORDER_TYPE_BUY:
                new_sl = current_close - (1.5 * atr)
                if new_sl > current_sl and (current_close - new_sl) > 0:
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": ticket,
                        "sl": new_sl,
                        "tp": pos.tp
                    }
                    res = mt5.order_send(request)
                    if res.retcode == mt5.TRADE_RETCODE_DONE:
                        logging.info(f"Trailing SL updated for BUY position {ticket} to {new_sl}")
                        
            elif pos_type == mt5.ORDER_TYPE_SELL:
                new_sl = current_close + (1.5 * atr)
                if (current_sl == 0.0 or new_sl < current_sl) and (new_sl - current_close) > 0:
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": ticket,
                        "sl": new_sl,
                        "tp": pos.tp
                    }
                    res = mt5.order_send(request)
                    if res.retcode == mt5.TRADE_RETCODE_DONE:
                        logging.info(f"Trailing SL updated for SELL position {ticket} to {new_sl}")
    except Exception as e:
        log_error(f"Error trailing stops: {e}")

def check_closed_trades():
    try:
        db = SessionLocal()
        open_trades = db.query(Trade).filter(Trade.exit_price == None).all()
        if not open_trades:
            db.close()
            return
            
        from_date = datetime.now() - timedelta(days=2)
        to_date = datetime.now() + timedelta(days=1)
        history = mt5.history_deals_get(from_date, to_date)
        
        if history is None:
            db.close()
            return
            
        for trade in open_trades:
            for deal in history:
                if deal.position_id == trade.ticket and deal.entry == mt5.DEAL_ENTRY_OUT:
                    trade.exit_price = deal.price
                    trade.pnl = deal.profit
                    trade.close_time = datetime.utcnow()
                    trade.close_reason = "TP/SL/Manual"
                    db.commit()
                    logging.info(f"Memory System Updated: Trade {trade.ticket} Closed. PnL: ${deal.profit:.2f}")
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
        for err in errors[-5:]:
            report += f"- {err}\n"
    else:
        report += "✅ <b>System Health:</b> 100% Perfect (No Breakdowns)\n"
        
    send_telegram(report)
    
    state["errors"] = []
    state["trades"] = 0
    state["last_report_time"] = datetime.now().timestamp()
    save_state(state)

def run_agent():
    if not init_mt5():
        return

    logging.info(f"KRONOS HYPER AGENT STARTED! (30M HYBRID ARCHITECTURE)")
    send_telegram("🚀 <b>Kronos HYPER Agent is officially ONLINE!</b>\nTrading XAUUSD M30 exclusively using the PyTorch Foundation Model + Hybrid Trailing logic.")
    
    state = load_state()
    if state.get("last_report_time", 0.0) == 0.0:
        state["last_report_time"] = datetime.now().timestamp()
        save_state(state)
    
    while True:
        current_time = datetime.now()
        utc_now = datetime.utcnow()
        
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
            
            if len(df) < 64:
                logging.info(f"[{pair}] Not enough data (got {len(df)} candles). Needs 64.")
                continue
                
            df_slice = df.iloc[-64:]
            
            latest_candle = df.iloc[-1] 
            
            adx = float(latest_candle['adx'])
            ema_200 = float(latest_candle['ema_200'])
            atr = float(latest_candle['atr'])
            current_close = float(latest_candle['close'])
            
            pred_close, cur_close = ask_kronos_brain(df_slice)
            
            if pred_close is not None:
                gap = pred_close - current_close
                
                if gap > (0.1 * atr) or gap < -(0.1 * atr):
                    safe, reason = news_filter.is_safe_to_trade(pair, buffer_minutes=15)
                    if not safe:
                        msg = f"⚠️ [{pair}] TRADE BLOCKED by News Filter: {reason}"
                        logging.warning(msg)
                        send_telegram(msg)
                        continue
                        
                # CONFLUENCE MACRO FILTERS
                if adx < 20:
                    logging.info(f"[{pair}] ADX < 20 ({adx:.2f}). Market chopping. Blocked.")
                    continue
                    
                if gap > (0.1 * atr):
                    if current_close > ema_200:
                        signals.append((pair, 1, pred_close, current_close, gap, atr))
                    else:
                        logging.info(f"[{pair}] BUY predicted (gap {gap:.2f}) but Price < 200 EMA. Blocked.")
                elif gap < -(0.1 * atr):
                    if current_close < ema_200:
                        signals.append((pair, 0, pred_close, current_close, gap, atr))
                    else:
                        logging.info(f"[{pair}] SELL predicted (gap {gap:.2f}) but Price > 200 EMA. Blocked.")
                else:
                    logging.info(f"[{pair}] Gap too small ({gap:.2f}). Sitting out.")
            
        if signals:
            logging.info(f"Pre-computed {len(signals)} trade signals! Waiting for the exact 00.00 clock strike...")
            target_time = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=((current_time.minute // 30) + 1) * 30)
            while datetime.now() < target_time:
                time.sleep(0.01)
                
            logging.info(f"CLOCK STRUCK 00! FIRING {len(signals)} TRADES CONCURRENTLY!")
            for sig in signals:
                place_trade(sig[0], sig[1], sig[2], sig[3], sig[4], sig[5])
                
        if (datetime.now().timestamp() - state.get("last_report_time", 0.0)) >= 4 * 3600:
            generate_4h_report()
            
        check_closed_trades()
        manage_trailing_stops()

        now = datetime.now()
        next_minute = ((now.minute // 30) + 1) * 30
        next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=next_minute) - timedelta(seconds=2)
        sleep_seconds = (next_run - now).total_seconds()
        
        if sleep_seconds <= 0:
            sleep_seconds = 1798 # 30 minutes minus 2 seconds
            
        logging.info(f"Sleeping for {int(sleep_seconds)} seconds... Waking up at {next_run.strftime('%H:%M:%S')} to pre-compute.")
        time.sleep(sleep_seconds)

if __name__ == "__main__":
    run_agent()

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pytz
import psutil
import os
import time

def download_data():
    terminal_paths = []
    for p in psutil.process_iter(['name', 'exe']):
        if p.info['name'] == 'terminal64.exe' and p.info['exe']:
            terminal_paths.append(p.info['exe'])
            
    connected = False
    for path in terminal_paths:
        if mt5.initialize(path=path):
            connected = True
            break
            
    if not connected:
        print("initialize() failed. No running MT5 terminal found or failed to connect.")
        return

    print(f"Connected to Account: {mt5.account_info().login}")

    pairs = ['GBPUSDm', 'USDJPYm', 'XAUUSDm']
    
    timeframes = {
        'M1': mt5.TIMEFRAME_M1,
        'M3': mt5.TIMEFRAME_M3,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4
    }

    # Timezone: MT5 servers usually run on UTC+2 / UTC+3. Let's pull using UTC timezone for safety.
    timezone = pytz.timezone("Etc/UTC")
    
    # 7 Months Ago from today
    utc_to = datetime.now(tz=timezone)
    utc_from = utc_to - timedelta(days=7*30) # Roughly 7 months (210 days)

    print(f"Fetching data from {utc_from.strftime('%Y-%m-%d')} to {utc_to.strftime('%Y-%m-%d')}...")

    raw_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)

    for pair in pairs:
        print(f"\nProcessing {pair}...")
        for tf_name, tf_code in timeframes.items():
            print(f"  Downloading {tf_name}...", end=" ")
            
            # Fetch data
            rates = mt5.copy_rates_range(pair, tf_code, utc_from, utc_to)
            
            if rates is None or len(rates) == 0:
                print("FAILED! (No data returned)")
                continue
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Save to CSV
            filename = f"{pair}_{tf_name}_7M.csv"
            filepath = os.path.join(raw_dir, filename)
            df.to_csv(filepath, index=False)
            
            print(f"SUCCESS! ({len(df)} candles saved to {filename})")
            time.sleep(1) # Tiny pause so we don't overload the terminal

    mt5.shutdown()
    print("\n✅ All 7-Month Data Downloading Complete!")

if __name__ == "__main__":
    download_data()

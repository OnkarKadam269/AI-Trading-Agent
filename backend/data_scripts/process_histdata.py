import os
import zipfile
import pandas as pd
import ta

def process_data():
    pairs = ['GBPUSD', 'USDJPY', 'XAUUSD']
    years = ['2021', '2022', '2023']
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'histdata')
    processed_dir = os.path.join(os.path.dirname(__file__), 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    for pair in pairs:
        print(f"--- Processing {pair} ---")
        df_list = []
        for year in years:
            zip_path = os.path.join(data_dir, f'DAT_ASCII_{pair}_M1_{year}.zip')
            if not os.path.exists(zip_path):
                print(f"Skipping {pair} {year}, zip not found.")
                continue
                
            csv_filename = ''
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                csv_filename = [name for name in zip_ref.namelist() if name.endswith('.csv')][0]
                zip_ref.extract(csv_filename, data_dir)
                
            csv_path = os.path.join(data_dir, csv_filename)
            
            # format: 20230101 180400;1.206150;1.206150;1.206130;1.206130;0
            df_year = pd.read_csv(csv_path, sep=';', header=None, names=['datetime_str', 'open', 'high', 'low', 'close', 'tick_volume'])
            df_year['time'] = pd.to_datetime(df_year['datetime_str'], format='%Y%m%d %H%M%S')
            df_year.set_index('time', inplace=True)
            df_year.drop(columns=['datetime_str'], inplace=True)
            
            df_list.append(df_year)
            os.remove(csv_path)
            
        if not df_list:
            continue
            
        # Merge all 3 years
        df = pd.concat(df_list)
        df.sort_index(inplace=True)
        
        # Save merged M1 data for execution engine
        out_m1_path = os.path.join(processed_dir, f'{pair}_M1_3Y.csv')
        df.to_csv(out_m1_path)
        print(f"Saved M1 Data to {out_m1_path} - Rows: {len(df)}")
        
        # Resample to 15 Min for AI Model
        print(f"Resampling {pair} to M15...")
        df_15m = df.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'tick_volume': 'sum'
        })
        
        df_15m.dropna(inplace=True)
        
        # Calculate features
        try:
            df_15m['rsi'] = ta.momentum.RSIIndicator(df_15m['close'], window=14).rsi()
            bb = ta.volatility.BollingerBands(df_15m['close'], window=20, window_dev=2)
            df_15m['bb_width'] = bb.bollinger_wband()
            sma20 = ta.trend.SMAIndicator(df_15m['close'], window=20).sma_indicator()
            df_15m['dist_sma20'] = (df_15m['close'] - sma20) / sma20 * 100
        except Exception as e:
            print(f"Error calculating features: {e}")
            
        df_15m.dropna(inplace=True)
        
        # Save M15 data
        out_m15_path = os.path.join(processed_dir, f'{pair}_M15_3Y.csv')
        df_15m.to_csv(out_m15_path)
        print(f"Saved M15 Data to {out_m15_path} - Rows: {len(df_15m)}")

if __name__ == "__main__":
    process_data()

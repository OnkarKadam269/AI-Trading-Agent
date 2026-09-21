import os
from histdata import download_hist_data as dl
from histdata.api import Platform as P, TimeFrame as TF

def download_data():
    pairs = ['gbpusd', 'usdjpy', 'xauusd']
    years = ['2021', '2022', '2023']
    
    download_dir = os.path.join(os.path.dirname(__file__), 'data', 'histdata')
    os.makedirs(download_dir, exist_ok=True)
    
    for pair in pairs:
        for year in years:
            # Check if file already exists to avoid re-downloading
            zip_filename = f'DAT_ASCII_{pair.upper()}_M1_{year}.zip'
            if os.path.exists(os.path.join(download_dir, zip_filename)):
                print(f"Skipping {pair} {year}, already downloaded.")
                continue
                
            print(f"Downloading {pair} for {year}...")
            try:
                dl(year=year, pair=pair, platform=P.GENERIC_ASCII, time_frame=TF.ONE_MINUTE, output_directory=download_dir)
                print(f"Successfully downloaded {pair} for {year}")
            except Exception as e:
                print(f"Failed to download {pair} for {year}: {e}")
                print(f"Trying month by month for {pair} {year}...")
                for month in range(1, 13):
                    try:
                        dl(year=year, month=str(month), pair=pair, platform=P.GENERIC_ASCII, time_frame=TF.ONE_MINUTE, output_directory=download_dir)
                        print(f"Downloaded {pair} {year}-{month}")
                    except Exception as ex:
                        print(f"Failed to download {pair} {year}-{month}: {ex}")

if __name__ == "__main__":
    download_data()

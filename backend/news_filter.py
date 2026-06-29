import os
import time
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import pytz
import logging

CACHE_FILE = os.path.join(os.path.dirname(__file__), 'ff_calendar_cache.xml')
CACHE_EXPIRY_SECONDS = 3600  # 1 hour

def fetch_and_cache_xml():
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            with open(CACHE_FILE, 'wb') as f:
                f.write(response.content)
            return response.content
    except Exception as e:
        logging.error(f"Failed to fetch Forex Factory XML: {e}")
    return None

def get_forex_factory_news():
    # Check if cache exists and is fresh
    if os.path.exists(CACHE_FILE):
        file_mod_time = os.path.getmtime(CACHE_FILE)
        if (time.time() - file_mod_time) < CACHE_EXPIRY_SECONDS:
            with open(CACHE_FILE, 'rb') as f:
                return f.read()
                
    # Otherwise fetch and cache
    content = fetch_and_cache_xml()
    if content:
        return content
        
    # Fallback to old cache if fetch fails
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            return f.read()
    return None

def is_safe_to_trade(symbol, buffer_minutes=15):
    """
    Returns (True, None) if safe.
    Returns (False, reason_string) if a high-impact news event is near.
    """
    # 1. Determine relevant currencies from symbol (e.g. GBPUSDm -> GBP, USD)
    base_currency = symbol[:3].upper()
    quote_currency = symbol[3:6].upper()
    relevant_currencies = [base_currency, quote_currency]
    
    # 2. Get XML
    xml_content = get_forex_factory_news()
    if not xml_content:
        logging.warning("Could not load news data. Proceeding with caution.")
        return True, None
        
    try:
        root = ET.fromstring(xml_content)
    except Exception as e:
        logging.error(f"Failed to parse news XML: {e}")
        return True, None

    eastern = pytz.timezone('US/Eastern')
    now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)

    for event in root.findall('event'):
        impact = event.find('impact').text
        country = event.find('country').text
        
        # Only care about High Impact (Red Folder) and relevant currencies
        if impact != 'High' or country not in relevant_currencies:
            continue
            
        date_str = event.find('date').text
        time_str = event.find('time').text
        
        # Some events are "All Day" or "Tentative" with no specific time
        if not time_str or time_str.lower() in ['all day', 'tentative']:
            continue
            
        # Parse the EST time
        try:
            event_datetime_str = f"{date_str} {time_str}"
            event_dt = datetime.strptime(event_datetime_str, "%m-%d-%Y %I:%M%p")
            
            # Localize to US/Eastern then convert to UTC
            event_dt_est = eastern.localize(event_dt)
            event_dt_utc = event_dt_est.astimezone(pytz.utc)
            
            # Check time difference
            diff_seconds = abs((event_dt_utc - now_utc).total_seconds())
            if diff_seconds <= (buffer_minutes * 60):
                title = event.find('title').text
                reason = f"High-Impact News ({country}): '{title}' is within {buffer_minutes} mins."
                return False, reason
        except Exception as e:
            # Skip unparseable dates
            continue
            
    return True, None

# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pairs = ['GBPUSDm', 'USDJPYm', 'XAUUSDm']
    for p in pairs:
        safe, reason = is_safe_to_trade(p)
        if not safe:
            print(f"{p}: UNSAFE - {reason}")
        else:
            print(f"{p}: SAFE")

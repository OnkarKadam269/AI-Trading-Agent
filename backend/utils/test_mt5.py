import MetaTrader5 as mt5

def test_connection():
    if not mt5.initialize():
        print(f"initialize() failed, error code: {mt5.last_error()}")
        return

    account = mt5.account_info()
    if account:
        print(f"Connected to {account.company} (Server: {account.server}, Login: {account.login})")
    
    symbols = mt5.symbols_get()
    if symbols:
        # Just grab the first few forex symbols to see the format
        forex_symbols = [s.name for s in symbols if "USD" in s.name][:10]
        print(f"Sample Symbols available: {forex_symbols}")
    
    mt5.shutdown()

if __name__ == "__main__":
    test_connection()

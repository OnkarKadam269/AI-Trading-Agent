import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # MT5 Configurations
    MT5_ACCOUNT: int = int(os.getenv("MT5_ACCOUNT", "0"))
    MT5_PASSWORD: str = os.getenv("MT5_PASSWORD", "")
    MT5_SERVER: str = os.getenv("MT5_SERVER", "")
    MT5_PATH: str = os.getenv("MT5_PATH", "C:\\Program Files\\MetaTrader 5\\terminal64.exe")

    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_API_KEY_FALLBACK: str = os.getenv("GEMINI_API_KEY_FALLBACK", "")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")

    # Application Settings
    MAX_CONCURRENT_TRADES: int = 3
    MAX_DAILY_DRAWDOWN_PCT: float = 3.0
    RISK_PER_TRADE_PCT: float = 1.0

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

"""
App-wide configuration — all secrets loaded from .env, never hardcoded.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# Alpaca paper trading (never use live keys in dev)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

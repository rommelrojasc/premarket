"""Configuration for pre-market analysis system."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models_store")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Ticker
TICKER = "SPY"

# IBKR Gateway connection
IB_HOST = os.environ.get("IB_HOST", "127.0.0.1")
IB_PORT = int(os.environ.get("IB_PORT", "4002"))
IB_CLIENT_ID = int(os.environ.get("IB_CLIENT_ID", "10"))  # Different from op3's ID=2

# Time windows (Eastern Time)
PREMARKET_START = "04:00"
PREMARKET_END = "09:30"
RTH_OPEN = "09:30"
RTH_CLOSE = "16:00"

# Prediction horizons
HORIZONS = {
    "30m": {"label": "FIRST 30 MIN", "sublabel": "9:30-10:00", "minutes": 30},
    "morning": {"label": "MORNING", "sublabel": "9:30-11:00", "minutes": 90},
    "day": {"label": "FULL DAY", "sublabel": "9:30-16:00", "minutes": 390},
}

# ML settings
TRAINING_YEARS = 5
CONFIDENCE_THRESHOLD = 0.60  # Below this, show "NO EDGE"
HIGH_CONF_THRESHOLD = 0.75

# CPCV settings
N_SPLITS = 10
N_TEST_GROUPS = 2
PURGE_DAYS = 5
EMBARGO_DAYS = 2

# Options/GEX
OPTIONS_EXPIRY_DAYS = 7  # Look at expirations within this window

# Scorecard
SCORECARD_DAYS = 15

# Macro events calendar (date -> event name)
MACRO_EVENTS = {
    "2026-03-11": "CPI",
    "2026-03-12": "PPI",
    "2026-03-19": "FOMC",
}

# Flat threshold for actual returns
FLAT_THRESHOLD = 0.10  # +/- 0.10% considered FLAT

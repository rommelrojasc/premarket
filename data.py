"""Data fetching via IBKR Gateway for pre-market analysis."""
import datetime as dt
import math
import os
import json
import logging

import numpy as np
import pandas as pd
import ib_compat  # noqa: F401 — Python 3.14+ event loop fix
from ib_insync import Stock, Index, util

import config
from ib_client import ib_client

logger = logging.getLogger(__name__)


def _build_spy_contract():
    return Stock(config.TICKER, "SMART", "USD")


def _build_vix_contract():
    return Index("VIX", "CBOE", "USD")


def _qualify(contract):
    """Qualify a contract (required before data requests)."""
    ib = ib_client.ib
    ib.qualifyContracts(contract)
    return contract


def fetch_historical_daily(ticker=config.TICKER, years=config.TRAINING_YEARS):
    """Fetch historical daily OHLCV data from IBKR."""
    ib = ib_client.ib

    if ticker == "SPY":
        contract = _build_spy_contract()
    else:
        contract = Stock(ticker, "SMART", "USD")
    _qualify(contract)

    duration = f"{years} Y"
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=duration,
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )

    if not bars:
        return pd.DataFrame()

    rows = []
    for bar in bars:
        bar_date = bar.date
        if isinstance(bar_date, dt.date) and not isinstance(bar_date, dt.datetime):
            bar_date = dt.datetime.combine(bar_date, dt.time.min)
        rows.append({
            "Open": float(bar.open),
            "High": float(bar.high),
            "Low": float(bar.low),
            "Close": float(bar.close),
            "Volume": int(bar.volume),
        })

    df = pd.DataFrame(rows, index=pd.DatetimeIndex(
        [bar.date if isinstance(bar.date, dt.datetime)
         else dt.datetime.combine(bar.date, dt.time.min)
         for bar in bars]
    ))
    return df


def fetch_intraday_history(ticker=config.TICKER, duration="30 D", bar_size="5 mins",
                           use_rth=False):
    """
    Fetch recent intraday data including pre-market and RTH.
    use_rth=False includes extended hours (pre-market + after-hours).
    """
    ib = ib_client.ib

    if ticker == "SPY":
        contract = _build_spy_contract()
    else:
        contract = Stock(ticker, "SMART", "USD")
    _qualify(contract)

    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow="TRADES",
        useRTH=use_rth,
        formatDate=1,
    )

    if not bars:
        return pd.DataFrame()

    rows = []
    times = []
    for bar in bars:
        bar_date = bar.date
        if isinstance(bar_date, dt.date) and not isinstance(bar_date, dt.datetime):
            bar_date = dt.datetime.combine(bar_date, dt.time.min)
        rows.append({
            "Open": float(bar.open),
            "High": float(bar.high),
            "Low": float(bar.low),
            "Close": float(bar.close),
            "Volume": int(bar.volume),
        })
        times.append(bar_date)

    df = pd.DataFrame(rows, index=pd.DatetimeIndex(times))
    # IB returns timestamps in the gateway's configured timezone (US/Eastern)
    # If the index is tz-naive, localize it
    if df.index.tz is None:
        try:
            import pytz
            df.index = df.index.tz_localize("US/Eastern")
        except Exception:
            pass
    return df


def fetch_current_premarket(ticker=config.TICKER):
    """
    Fetch today's pre-market data (4:00-9:30 AM ET) from IBKR.
    Uses 1-min bars with useRTH=False to get extended hours.
    """
    ib = ib_client.ib

    if ticker == "SPY":
        contract = _build_spy_contract()
    else:
        contract = Stock(ticker, "SMART", "USD")
    _qualify(contract)

    # Fetch today's data at 1-min resolution including pre-market
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="1 D",
        barSizeSetting="1 min",
        whatToShow="TRADES",
        useRTH=False,  # Include pre-market
        formatDate=1,
    )

    if not bars:
        return pd.DataFrame()

    rows = []
    times = []
    for bar in bars:
        bar_date = bar.date
        if isinstance(bar_date, dt.date) and not isinstance(bar_date, dt.datetime):
            bar_date = dt.datetime.combine(bar_date, dt.time.min)
        rows.append({
            "Open": float(bar.open),
            "High": float(bar.high),
            "Low": float(bar.low),
            "Close": float(bar.close),
            "Volume": int(bar.volume),
        })
        times.append(bar_date)

    df = pd.DataFrame(rows, index=pd.DatetimeIndex(times))

    if df.index.tz is None:
        try:
            import pytz
            df.index = df.index.tz_localize("US/Eastern")
        except Exception:
            pass

    # Filter to pre-market window (4:00 - 9:30 ET)
    import pytz
    ET = pytz.timezone("US/Eastern")
    today = dt.datetime.now(ET).date()
    pm_start = ET.localize(dt.datetime.combine(today, dt.time(4, 0)))
    pm_end = ET.localize(dt.datetime.combine(today, dt.time(9, 30)))
    mask = (df.index >= pm_start) & (df.index < pm_end)
    return df[mask]


def fetch_vix():
    """Fetch current VIX level from IBKR."""
    try:
        ib = ib_client.ib
        contract = _build_vix_contract()
        _qualify(contract)

        # Get last VIX close
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr="5 D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        if bars:
            return float(bars[-1].close)
    except Exception as e:
        logger.warning("Failed to fetch VIX from IBKR: %s", e)
    return None


def get_previous_close(ticker=config.TICKER):
    """Get previous and last trading day closes from IBKR."""
    ib = ib_client.ib

    if ticker == "SPY":
        contract = _build_spy_contract()
    else:
        contract = Stock(ticker, "SMART", "USD")
    _qualify(contract)

    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="5 D",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )

    if len(bars) >= 2:
        return float(bars[-2].close), float(bars[-1].close)
    return None, None


def get_spy_current_price(ticker=config.TICKER):
    """Get current SPY price via IBKR snapshot."""
    try:
        ib = ib_client.ib
        if ticker == "SPY":
            contract = _build_spy_contract()
        else:
            contract = Stock(ticker, "SMART", "USD")
        _qualify(contract)

        # Request a market data snapshot
        ib.reqMarketDataType(3)  # Delayed data as fallback
        ticker_data = ib.reqTickers(contract)
        if ticker_data:
            t = ticker_data[0]
            price = t.marketPrice()
            if price and math.isfinite(price):
                return float(price)
            if t.last and math.isfinite(t.last):
                return float(t.last)
            if t.close and math.isfinite(t.close):
                return float(t.close)
    except Exception as e:
        logger.warning("Failed to get current price from IBKR: %s", e)
    return None


def split_premarket_rth(intraday_df):
    """Split intraday data into pre-market and RTH sessions per day."""
    import pytz
    ET = pytz.timezone("US/Eastern")
    sessions = {}

    for date_val, group in intraday_df.groupby(intraday_df.index.date):
        pm_start = ET.localize(dt.datetime.combine(date_val, dt.time(4, 0)))
        pm_end = ET.localize(dt.datetime.combine(date_val, dt.time(9, 30)))
        rth_start = pm_end
        rth_end = ET.localize(dt.datetime.combine(date_val, dt.time(16, 0)))

        pm = group[(group.index >= pm_start) & (group.index < pm_end)]
        rth = group[(group.index >= rth_start) & (group.index <= rth_end)]

        if not pm.empty and not rth.empty:
            sessions[date_val] = {"premarket": pm, "rth": rth}
    return sessions


def save_predictions(predictions, filepath=None):
    """Save predictions to JSON file."""
    if filepath is None:
        filepath = os.path.join(config.DATA_DIR, "predictions.json")
    existing = load_predictions(filepath)
    date_str = predictions["date"]
    existing[date_str] = predictions
    with open(filepath, "w") as f:
        json.dump(existing, f, indent=2, default=str)


def load_predictions(filepath=None):
    """Load predictions from JSON file."""
    if filepath is None:
        filepath = os.path.join(config.DATA_DIR, "predictions.json")
    if os.path.exists(filepath):
        with open(filepath) as f:
            return json.load(f)
    return {}


def save_actuals(actuals, filepath=None):
    """Save actual outcomes to JSON file."""
    if filepath is None:
        filepath = os.path.join(config.DATA_DIR, "actuals.json")
    existing = load_actuals(filepath)
    date_str = actuals["date"]
    existing[date_str] = actuals
    with open(filepath, "w") as f:
        json.dump(existing, f, indent=2, default=str)


def load_actuals(filepath=None):
    """Load actual outcomes from JSON file."""
    if filepath is None:
        filepath = os.path.join(config.DATA_DIR, "actuals.json")
    if os.path.exists(filepath):
        with open(filepath) as f:
            return json.load(f)
    return {}

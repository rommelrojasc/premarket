"""Options walls and Gamma Exposure (GEX) analysis via IBKR."""
import datetime as dt
import math
import logging
from math import log, sqrt, exp, pi
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import ib_compat  # noqa: F401 — Python 3.14+ event loop fix
from ib_insync import Stock, Option

import config
from ib_client import ib_client

logger = logging.getLogger(__name__)


def _get_near_expirations(symbol=config.TICKER) -> List[str]:
    """Get option expirations within the configured window from IBKR."""
    ib = ib_client.ib

    contract = Stock(symbol, "SMART", "USD")
    ib.qualifyContracts(contract)

    chains = ib.reqSecDefOptParams(symbol, "", "STK", contract.conId)
    if not chains:
        return []

    today = dt.date.today()
    max_date = today + dt.timedelta(days=config.OPTIONS_EXPIRY_DAYS)

    all_expirations = set()
    all_strikes_by_exp = {}

    for chain in chains:
        for exp in chain.expirations:
            # IB returns YYYYMMDD
            if len(exp) == 8:
                exp_date = dt.datetime.strptime(exp, "%Y%m%d").date()
                exp_formatted = f"{exp[0:4]}-{exp[4:6]}-{exp[6:8]}"
            else:
                exp_date = dt.datetime.strptime(exp, "%Y-%m-%d").date()
                exp_formatted = exp
            if exp_date <= max_date and exp_date >= today:
                all_expirations.add(exp_formatted)
                if exp_formatted not in all_strikes_by_exp:
                    all_strikes_by_exp[exp_formatted] = set()
                all_strikes_by_exp[exp_formatted].update(float(s) for s in chain.strikes)

    sorted_exps = sorted(all_expirations)
    return sorted_exps, all_strikes_by_exp


def _get_current_price(symbol=config.TICKER) -> Optional[float]:
    """Get current underlying price from IBKR."""
    try:
        ib = ib_client.ib
        contract = Stock(symbol, "SMART", "USD")
        ib.qualifyContracts(contract)

        now = dt.datetime.now(ZoneInfo("America/New_York"))
        is_market_open = (now.weekday() < 5 and
                          dt.time(9, 30) <= now.time() <= dt.time(16, 0))
        ib.reqMarketDataType(1 if is_market_open else 3)

        tickers = ib.reqTickers(contract)
        if tickers:
            t = tickers[0]
            price = t.marketPrice()
            if price and math.isfinite(price):
                return float(price)
            if t.last and math.isfinite(t.last):
                return float(t.last)
            if t.close and math.isfinite(t.close):
                return float(t.close)
    except Exception as e:
        logger.warning("Failed to get price: %s", e)
    return None


def _get_prev_close(symbol=config.TICKER):
    """Get previous close from IBKR."""
    try:
        ib = ib_client.ib
        contract = Stock(symbol, "SMART", "USD")
        ib.qualifyContracts(contract)
        bars = ib.reqHistoricalData(
            contract, endDateTime="", durationStr="5 D",
            barSizeSetting="1 day", whatToShow="TRADES",
            useRTH=True, formatDate=1,
        )
        if len(bars) >= 2:
            return float(bars[-2].close), float(bars[-1].close)
    except Exception:
        pass
    return None, None


def fetch_options_data(symbol=config.TICKER):
    """
    Fetch options chain data with OI, IV, delta from IBKR.
    Returns aggregated call/put data by strike.
    """
    ib = ib_client.ib
    current_price = _get_current_price(symbol)
    if current_price is None:
        logger.error("Cannot fetch options: no current price")
        return None, []

    try:
        expirations, strikes_by_exp = _get_near_expirations(symbol)
    except Exception as e:
        logger.error("Failed to get expirations: %s", e)
        return None, []

    if not expirations:
        return None, []

    # Limit to first 3 expirations to keep IBKR requests manageable
    expirations = expirations[:3]

    now = dt.datetime.now(ZoneInfo("America/New_York"))
    is_market_open = (now.weekday() < 5 and
                      dt.time(9, 30) <= now.time() <= dt.time(16, 0))
    ib.reqMarketDataType(1 if is_market_open else 3)

    all_calls = []
    all_puts = []

    for exp in expirations:
        strikes = strikes_by_exp.get(exp, set())
        if not strikes:
            continue

        # Filter to integer strikes within ±30 points of current price
        in_range = [s for s in strikes
                    if abs(s - current_price) <= 30 and s == int(s)]
        nearby = sorted(in_range, key=lambda s: abs(s - current_price))[:20]
        if not nearby:
            logger.debug("No valid strikes for %s near %.1f (had %d raw strikes)",
                         exp, current_price, len(strikes))
            continue
        exp_ib = exp.replace("-", "")

        contracts = []
        contract_meta = []
        for strike in nearby:
            c_call = Option(symbol, exp_ib, float(strike), "C", "SMART")
            c_put = Option(symbol, exp_ib, float(strike), "P", "SMART")
            contracts.extend([c_call, c_put])
            contract_meta.extend([
                {"strike": strike, "right": "C", "exp": exp},
                {"strike": strike, "right": "P", "exp": exp},
            ])

        qualified = ib.qualifyContracts(*contracts)
        valid = [(meta, con) for meta, con in zip(contract_meta, qualified)
                 if getattr(con, "conId", 0)]

        if not valid:
            continue

        valid_contracts = [con for _, con in valid]
        tickers = ib.reqTickers(*valid_contracts)

        # Retry with delayed data if needed
        if is_market_open:
            has_prices = any(
                (t.last and t.last > 0) or (t.bid and t.bid > 0) or (t.ask and t.ask > 0)
                for t in tickers
            )
            if not has_prices:
                ib.reqMarketDataType(3)
                tickers = ib.reqTickers(*valid_contracts)

        # Compute DTE for this expiration (used for BS gamma fallback)
        exp_date = dt.datetime.strptime(exp, "%Y-%m-%d").date()
        dte = max((exp_date - dt.date.today()).days, 1)

        for (meta, _con), ticker in zip(valid, tickers):
            oi = (getattr(ticker, "openInterest", None) or
                  getattr(ticker, "callOpenInterest", None) or
                  getattr(ticker, "putOpenInterest", None))
            if oi is not None and isinstance(oi, float) and not math.isfinite(oi):
                oi = None

            iv = getattr(ticker, "impliedVolatility", None)
            delta = None
            gamma = None
            if ticker.modelGreeks is not None:
                if iv is None:
                    iv = ticker.modelGreeks.impliedVol
                raw_delta = ticker.modelGreeks.delta
                if raw_delta is not None and isinstance(raw_delta, float) and math.isfinite(raw_delta):
                    delta = raw_delta
                raw_gamma = getattr(ticker.modelGreeks, "gamma", None)
                if raw_gamma is not None and isinstance(raw_gamma, float) and math.isfinite(raw_gamma):
                    gamma = raw_gamma

            if iv is not None and isinstance(iv, float) and not math.isfinite(iv):
                iv = None

            # Fallback: compute gamma from IV via Black-Scholes if IBKR didn't provide it
            if gamma is None and iv and iv > 0 and current_price > 0:
                gamma = _bs_gamma(current_price, meta["strike"], iv, dte / 365.0)

            entry = {
                "strike": meta["strike"],
                "expiration": meta["exp"],
                "openInterest": int(oi) if oi else 0,
                "iv": float(iv) if iv else None,
                "delta": float(delta) if delta else None,
                "gamma": float(gamma) if gamma else None,
            }

            if meta["right"] == "C":
                all_calls.append(entry)
            else:
                all_puts.append(entry)

    if not all_calls and not all_puts:
        return None, expirations

    import pandas as pd
    calls_df = pd.DataFrame(all_calls)
    puts_df = pd.DataFrame(all_puts)

    return {"calls": calls_df, "puts": puts_df}, expirations


def _bs_gamma(spot, strike, sigma, T, r=0.05):
    """
    Compute option gamma via Black-Scholes.
    Used as fallback when IBKR modelGreeks doesn't provide gamma.
    Gamma is the same for calls and puts.
    """
    if T <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    try:
        d1 = (log(spot / strike) + (r + sigma ** 2 / 2) * T) / (sigma * sqrt(T))
        return exp(-d1 ** 2 / 2) / (spot * sigma * sqrt(2 * pi * T))
    except (ValueError, ZeroDivisionError):
        return 0.0


def compute_options_walls(options_data, current_price):
    """
    Compute put wall (support) and call wall (resistance) from open interest.
    """
    if options_data is None:
        return None

    calls = options_data["calls"]
    puts = options_data["puts"]

    call_oi = calls.groupby("strike")["openInterest"].sum().reset_index()
    put_oi = puts.groupby("strike")["openInterest"].sum().reset_index()

    # Call wall: highest OI strike above current price
    calls_above = call_oi[call_oi["strike"] > current_price]
    call_wall = None
    if not calls_above.empty:
        idx = calls_above["openInterest"].idxmax()
        call_wall = calls_above.loc[idx]

    # Put wall: highest OI strike below current price
    puts_below = put_oi[put_oi["strike"] < current_price]
    put_wall = None
    if not puts_below.empty:
        idx = puts_below["openInterest"].idxmax()
        put_wall = puts_below.loc[idx]

    return {
        "call_wall": {
            "strike": float(call_wall["strike"]),
            "oi": int(call_wall["openInterest"]),
        } if call_wall is not None else None,
        "put_wall": {
            "strike": float(put_wall["strike"]),
            "oi": int(put_wall["openInterest"]),
        } if put_wall is not None else None,
    }


def compute_max_pain(options_data):
    """Compute max pain: the strike where total option value is minimized."""
    if options_data is None:
        return None

    calls = options_data["calls"]
    puts = options_data["puts"]

    call_oi = calls.groupby("strike")["openInterest"].sum()
    put_oi = puts.groupby("strike")["openInterest"].sum()

    all_strikes = sorted(set(call_oi.index) | set(put_oi.index))
    if not all_strikes:
        return None

    min_pain = float("inf")
    max_pain_strike = all_strikes[0]

    for strike in all_strikes:
        call_pain = 0
        for s, oi in call_oi.items():
            if s < strike:
                call_pain += (strike - s) * oi * 100
        put_pain = 0
        for s, oi in put_oi.items():
            if s > strike:
                put_pain += (s - strike) * oi * 100

        total_pain = call_pain + put_pain
        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = strike

    return float(max_pain_strike)


def compute_gex(options_data, current_price):
    """
    Compute Gamma Exposure (GEX).

    Uses gamma from IBKR Greeks when available, falls back to
    Black-Scholes gamma computed from IV + DTE.

    Net GEX = sum(call_gamma * call_OI * 100 * spot) - sum(put_gamma * put_OI * 100 * spot)
    """
    if options_data is None:
        return None

    calls = options_data["calls"]
    puts = options_data["puts"]

    def _safe_gamma(row, spot):
        """Get gamma from row, falling back to BS if needed."""
        g = row.get("gamma")
        if g and isinstance(g, float) and math.isfinite(g) and g > 0:
            return g
        # Fallback: compute from IV
        iv = row.get("iv")
        if iv and isinstance(iv, float) and math.isfinite(iv) and iv > 0:
            strike = row["strike"]
            # Estimate DTE from expiration if available, else default 7 days
            dte_years = 7 / 365.0
            exp = row.get("expiration")
            if exp:
                try:
                    exp_date = dt.datetime.strptime(str(exp), "%Y-%m-%d").date()
                    dte_years = max((exp_date - dt.date.today()).days, 1) / 365.0
                except (ValueError, TypeError):
                    pass
            return _bs_gamma(spot, strike, iv, dte_years)
        # Last resort: rough BS estimate with 20% vol, 7 DTE
        return _bs_gamma(spot, row["strike"], 0.20, 7 / 365.0)

    call_gex = []
    for _, row in calls.iterrows():
        gamma = _safe_gamma(row, current_price)
        gex = gamma * row["openInterest"] * 100 * current_price
        call_gex.append({"strike": row["strike"], "gex": gex, "oi": row["openInterest"]})

    put_gex = []
    for _, row in puts.iterrows():
        gamma = _safe_gamma(row, current_price)
        gex = -gamma * row["openInterest"] * 100 * current_price  # Negative for puts
        put_gex.append({"strike": row["strike"], "gex": gex, "oi": row["openInterest"]})

    if not call_gex and not put_gex:
        return None

    # Net GEX by strike
    all_gex = {}
    for item in call_gex:
        all_gex[item["strike"]] = all_gex.get(item["strike"], 0) + item["gex"]
    for item in put_gex:
        all_gex[item["strike"]] = all_gex.get(item["strike"], 0) + item["gex"]

    net_gex = sum(all_gex.values())

    # GEX Flip: where cumulative GEX crosses zero
    sorted_strikes = sorted(all_gex.keys())
    gex_flip = None
    cumulative = 0
    prev_cum = 0
    for s in sorted_strikes:
        prev_cum = cumulative
        cumulative += all_gex[s]
        if prev_cum * cumulative < 0:
            gex_flip = s
            break

    if gex_flip is None:
        min_abs = float("inf")
        running = 0
        for s in sorted_strikes:
            running += all_gex[s]
            if abs(running) < min_abs:
                min_abs = abs(running)
                gex_flip = s

    # P/C ratio from OI
    total_call_oi = calls["openInterest"].sum()
    total_put_oi = puts["openInterest"].sum()
    pc_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0

    is_negative_gamma = current_price < gex_flip if gex_flip else net_gex < 0
    total_contracts = int(total_call_oi + total_put_oi)

    return {
        "net_gex": net_gex,
        "net_gex_millions": round(net_gex / 1e6, 1),
        "gex_flip": round(gex_flip, 1) if gex_flip else None,
        "is_negative_gamma": is_negative_gamma,
        "gamma_zone": "-γ (negative gamma)" if is_negative_gamma else "+γ (positive gamma)",
        "pc_ratio": round(pc_ratio, 3),
        "total_contracts": total_contracts,
    }


def build_trading_scenarios(current_price, walls, max_pain, gex_data):
    """Build trading scenario table based on GEX and options data."""
    scenarios = []
    if not gex_data or not walls:
        return scenarios

    call_wall = walls["call_wall"]["strike"] if walls.get("call_wall") else current_price + 30
    put_wall = walls["put_wall"]["strike"] if walls.get("put_wall") else current_price - 30
    gex_flip = gex_data.get("gex_flip", current_price)
    is_neg_gamma = gex_data.get("is_negative_gamma", False)
    atm_strike = round(current_price)

    if is_neg_gamma:
        scenarios = [
            {
                "scenario": f"SPY above {current_price:.1f}, trending up",
                "gex_behavior": "Negative gamma amplifies rally",
                "trade": f"Buy {atm_strike}C (ATM)",
                "target": f"Target: {int(call_wall)} call wall",
                "trade_color": "green",
            },
            {
                "scenario": f"SPY below {current_price:.1f}, trending down",
                "gex_behavior": "Negative gamma amplifies selloff",
                "trade": f"Buy {atm_strike}P (ATM)",
                "target": f"Target: {int(put_wall)} put wall",
                "trade_color": "red",
            },
            {
                "scenario": f"SPY breaks above {gex_flip}",
                "gex_behavior": "Crosses into +γ -- moves get dampened",
                "trade": "Fade the breakout, mean-reversion",
                "target": "",
                "trade_color": "yellow",
            },
            {
                "scenario": f"SPY chops between {atm_strike}-{int(gex_flip)}",
                "gex_behavior": "Right at γ flip -- tug of war",
                "trade": "No trade or scalps only",
                "target": "",
                "trade_color": "white",
            },
            {
                "scenario": f"SPY hits {int(call_wall)}",
                "gex_behavior": "Call wall -- MM selling hard",
                "trade": f"Close your {atm_strike}C here",
                "target": "Resistance -- expect rejection, take profits",
                "trade_color": "yellow",
            },
            {
                "scenario": f"SPY hits {int(put_wall)}",
                "gex_behavior": "Put wall -- MM buying hard",
                "trade": f"Close your {atm_strike}P here",
                "target": "Support -- expect bounce, take profits",
                "trade_color": "yellow",
            },
            {
                "scenario": "Into close (last 2hrs)",
                "gex_behavior": f"Max pain gravity pulls to {int(max_pain) if max_pain else '?'}",
                "trade": f"Slight long bias -- {atm_strike}C (ATM)",
                "target": f"Target: {int(max_pain) if max_pain else '?'} (max pain)",
                "trade_color": "green",
            },
        ]
    else:
        scenarios = [
            {
                "scenario": f"SPY above {current_price:.1f}, trending up",
                "gex_behavior": "Positive gamma dampens rally",
                "trade": "Fade rallies, sell into strength",
                "target": f"Target: mean-revert to {int(max_pain) if max_pain else '?'}",
                "trade_color": "yellow",
            },
            {
                "scenario": f"SPY below {current_price:.1f}, trending down",
                "gex_behavior": "Positive gamma dampens selloff",
                "trade": "Buy dips, expect support",
                "target": f"Target: bounce back to {int(max_pain) if max_pain else '?'}",
                "trade_color": "green",
            },
            {
                "scenario": f"SPY drops below {gex_flip}",
                "gex_behavior": "Crosses into -γ -- moves accelerate",
                "trade": f"Buy {atm_strike}P (ATM)",
                "target": f"Target: {int(put_wall)} put wall",
                "trade_color": "red",
            },
            {
                "scenario": f"SPY range-bound near {int(max_pain) if max_pain else '?'}",
                "gex_behavior": "Max pain pinning likely",
                "trade": "Sell premium / iron condors",
                "target": "",
                "trade_color": "white",
            },
            {
                "scenario": "Into close (last 2hrs)",
                "gex_behavior": f"Max pain gravity pulls to {int(max_pain) if max_pain else '?'}",
                "trade": f"Slight long bias -- {atm_strike}C (ATM)",
                "target": f"Target: {int(max_pain) if max_pain else '?'} (max pain)",
                "trade_color": "green",
            },
        ]

    return scenarios


def get_full_options_analysis(ticker=config.TICKER):
    """Run the complete options/GEX analysis via IBKR."""
    current_price = _get_current_price(ticker)
    if current_price is None:
        return None

    options_data, expirations = fetch_options_data(ticker)
    if options_data is None:
        return None

    walls = compute_options_walls(options_data, current_price)
    max_pain = compute_max_pain(options_data)
    gex = compute_gex(options_data, current_price)

    prev_close_val, _ = _get_prev_close(ticker)
    price_change = current_price - prev_close_val if prev_close_val else 0
    price_change_pct = (price_change / prev_close_val * 100) if prev_close_val else 0

    scenarios = build_trading_scenarios(current_price, walls, max_pain, gex)

    return {
        "current_price": round(current_price, 2),
        "prev_close": round(prev_close_val, 2) if prev_close_val else None,
        "price_change": round(price_change, 2),
        "price_change_pct": round(price_change_pct, 2),
        "expirations": expirations,
        "walls": walls,
        "max_pain": round(max_pain, 0) if max_pain else None,
        "gex": gex,
        "scenarios": scenarios,
    }

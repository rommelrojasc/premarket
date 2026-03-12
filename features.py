"""Feature engineering from pre-market data window (4:00-9:30 AM ET)."""
import numpy as np
import pandas as pd


def compute_premarket_features(pm_data, daily_data=None, prev_close=None):
    """
    Build feature vector from pre-market price action.

    Features extracted:
    - Price-based: overnight gap, pre-market return, high/low range, VWAP deviation
    - Volume-based: volume profile, volume acceleration, relative volume
    - Momentum: pre-market RSI, momentum slopes, trend strength
    - Volatility: pre-market realized vol, ATR-like measures
    - Pattern: candle ratios, body/wick analysis
    - Context: day-of-week, IBS, VIX regime, recent RTH performance
    """
    if pm_data.empty:
        return None

    features = {}

    pm_open = float(pm_data["Open"].iloc[0])
    pm_close = float(pm_data["Close"].iloc[-1])
    pm_high = float(pm_data["High"].max())
    pm_low = float(pm_data["Low"].min())
    pm_volume = float(pm_data["Volume"].sum())

    # -- Overnight gap --
    if prev_close is not None and prev_close > 0:
        features["overnight_gap_pct"] = (pm_open - prev_close) / prev_close * 100
    else:
        features["overnight_gap_pct"] = 0.0

    # -- Pre-market return --
    if pm_open > 0:
        features["pm_return_pct"] = (pm_close - pm_open) / pm_open * 100
    else:
        features["pm_return_pct"] = 0.0

    # -- Total pre-market move from prev close --
    if prev_close is not None and prev_close > 0:
        features["pm_total_move_pct"] = (pm_close - prev_close) / prev_close * 100
    else:
        features["pm_total_move_pct"] = features["pm_return_pct"]

    # -- Pre-market range --
    if pm_open > 0:
        features["pm_range_pct"] = (pm_high - pm_low) / pm_open * 100
    else:
        features["pm_range_pct"] = 0.0

    # -- Pre-market high/low position (where close sits in range) --
    pm_range = pm_high - pm_low
    if pm_range > 0:
        features["pm_close_position"] = (pm_close - pm_low) / pm_range
    else:
        features["pm_close_position"] = 0.5

    # -- Volume features --
    features["pm_total_volume"] = pm_volume
    n_bars = len(pm_data)
    if n_bars > 1:
        half = n_bars // 2
        vol_first_half = float(pm_data["Volume"].iloc[:half].sum())
        vol_second_half = float(pm_data["Volume"].iloc[half:].sum())
        if vol_first_half > 0:
            features["pm_volume_acceleration"] = vol_second_half / vol_first_half
        else:
            features["pm_volume_acceleration"] = 1.0

        # Volume in last 30 min of premarket vs rest
        last_30 = pm_data.tail(30) if len(pm_data) >= 30 else pm_data.tail(n_bars // 3)
        rest = pm_data.iloc[:-len(last_30)] if len(last_30) < n_bars else pm_data.head(1)
        vol_last30 = float(last_30["Volume"].sum())
        vol_rest = float(rest["Volume"].sum())
        if vol_rest > 0:
            features["pm_volume_ramp"] = vol_last30 / vol_rest
        else:
            features["pm_volume_ramp"] = 1.0
    else:
        features["pm_volume_acceleration"] = 1.0
        features["pm_volume_ramp"] = 1.0

    # -- VWAP deviation --
    if pm_volume > 0:
        typical_price = (pm_data["High"] + pm_data["Low"] + pm_data["Close"]) / 3
        vwap = (typical_price * pm_data["Volume"]).sum() / pm_volume
        if vwap > 0:
            features["pm_vwap_deviation_pct"] = (pm_close - vwap) / vwap * 100
        else:
            features["pm_vwap_deviation_pct"] = 0.0
    else:
        features["pm_vwap_deviation_pct"] = 0.0

    # -- Momentum features --
    closes = pm_data["Close"].values.astype(float)
    if len(closes) > 5:
        # Simple momentum: last 1/3 vs first 1/3
        third = len(closes) // 3
        early_avg = np.mean(closes[:third])
        late_avg = np.mean(closes[-third:])
        if early_avg > 0:
            features["pm_momentum"] = (late_avg - early_avg) / early_avg * 100
        else:
            features["pm_momentum"] = 0.0

        # Linear regression slope
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]
        features["pm_trend_slope"] = slope

        # R-squared of linear fit
        pred = np.polyval(np.polyfit(x, closes, 1), x)
        ss_res = np.sum((closes - pred) ** 2)
        ss_tot = np.sum((closes - np.mean(closes)) ** 2)
        features["pm_trend_r2"] = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # RSI-like measure
        diffs = np.diff(closes)
        gains = np.where(diffs > 0, diffs, 0)
        losses = np.where(diffs < 0, -diffs, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            features["pm_rsi"] = 100 - (100 / (1 + rs))
        else:
            features["pm_rsi"] = 100.0 if avg_gain > 0 else 50.0
    else:
        features["pm_momentum"] = 0.0
        features["pm_trend_slope"] = 0.0
        features["pm_trend_r2"] = 0.0
        features["pm_rsi"] = 50.0

    # -- Volatility features --
    if len(closes) > 2:
        returns = np.diff(closes) / closes[:-1]
        features["pm_volatility"] = np.std(returns) * 100
        features["pm_max_drawdown"] = _max_drawdown(closes)
        features["pm_max_rally"] = _max_rally(closes)
    else:
        features["pm_volatility"] = 0.0
        features["pm_max_drawdown"] = 0.0
        features["pm_max_rally"] = 0.0

    # -- Candle pattern features --
    if n_bars >= 3:
        # Body-to-range ratio of the aggregate PM candle
        body = abs(pm_close - pm_open)
        if pm_range > 0:
            features["pm_body_ratio"] = body / pm_range
        else:
            features["pm_body_ratio"] = 0.0

        # Upper/lower wick ratios
        if pm_close >= pm_open:
            upper_wick = pm_high - pm_close
            lower_wick = pm_open - pm_low
        else:
            upper_wick = pm_high - pm_open
            lower_wick = pm_close - pm_low
        if pm_range > 0:
            features["pm_upper_wick_ratio"] = upper_wick / pm_range
            features["pm_lower_wick_ratio"] = lower_wick / pm_range
        else:
            features["pm_upper_wick_ratio"] = 0.0
            features["pm_lower_wick_ratio"] = 0.0
    else:
        features["pm_body_ratio"] = 0.0
        features["pm_upper_wick_ratio"] = 0.0
        features["pm_lower_wick_ratio"] = 0.0

    # -- Context features from daily data --
    if daily_data is not None and len(daily_data) >= 2:
        # Yesterday's IBS (Internal Bar Strength)
        yesterday = daily_data.iloc[-1]
        y_range = float(yesterday["High"]) - float(yesterday["Low"])
        if y_range > 0:
            features["yesterday_ibs"] = (float(yesterday["Close"]) - float(yesterday["Low"])) / y_range
        else:
            features["yesterday_ibs"] = 0.5

        # Yesterday's return
        if len(daily_data) >= 2:
            day_before = daily_data.iloc[-2]
            if float(day_before["Close"]) > 0:
                features["yesterday_return_pct"] = (
                    (float(yesterday["Close"]) - float(day_before["Close"]))
                    / float(day_before["Close"]) * 100
                )
            else:
                features["yesterday_return_pct"] = 0.0

        # 5-day return
        if len(daily_data) >= 6:
            five_ago = float(daily_data.iloc[-6]["Close"])
            if five_ago > 0:
                features["five_day_return_pct"] = (
                    (float(yesterday["Close"]) - five_ago) / five_ago * 100
                )
            else:
                features["five_day_return_pct"] = 0.0
        else:
            features["five_day_return_pct"] = 0.0

        # 5-day realized volatility
        if len(daily_data) >= 6:
            recent_closes = daily_data["Close"].iloc[-6:].astype(float).values
            daily_returns = np.diff(recent_closes) / recent_closes[:-1]
            features["five_day_vol"] = np.std(daily_returns) * 100
        else:
            features["five_day_vol"] = 0.0
    else:
        features["yesterday_ibs"] = 0.5
        features["yesterday_return_pct"] = 0.0
        features["five_day_return_pct"] = 0.0
        features["five_day_vol"] = 0.0

    # -- Day of week (one-hot would bloat, use cyclical encoding) --
    if not pm_data.empty:
        dow = pm_data.index[0].weekday()  # 0=Mon, 4=Fri
        features["day_sin"] = np.sin(2 * np.pi * dow / 5)
        features["day_cos"] = np.cos(2 * np.pi * dow / 5)
    else:
        features["day_sin"] = 0.0
        features["day_cos"] = 1.0

    return features


def _max_drawdown(prices):
    """Max drawdown from peak in percentage."""
    peak = prices[0]
    max_dd = 0
    for p in prices:
        if p > peak:
            peak = p
        dd = (peak - p) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _max_rally(prices):
    """Max rally from trough in percentage."""
    trough = prices[0]
    max_rally = 0
    for p in prices:
        if p < trough:
            trough = p
        rally = (p - trough) / trough * 100 if trough > 0 else 0
        if rally > max_rally:
            max_rally = rally
    return max_rally


def get_feature_names():
    """Return ordered list of feature names."""
    return [
        "overnight_gap_pct", "pm_return_pct", "pm_total_move_pct",
        "pm_range_pct", "pm_close_position",
        "pm_total_volume", "pm_volume_acceleration", "pm_volume_ramp",
        "pm_vwap_deviation_pct",
        "pm_momentum", "pm_trend_slope", "pm_trend_r2", "pm_rsi",
        "pm_volatility", "pm_max_drawdown", "pm_max_rally",
        "pm_body_ratio", "pm_upper_wick_ratio", "pm_lower_wick_ratio",
        "yesterday_ibs", "yesterday_return_pct",
        "five_day_return_pct", "five_day_vol",
        "day_sin", "day_cos",
    ]

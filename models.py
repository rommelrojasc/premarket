"""ML models for pre-market directional prediction."""
import os
import datetime as dt
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import config
import data
import features
from validation import CombinatorialPurgedCV, evaluate_with_cpcv


FEATURE_NAMES = features.get_feature_names()


def _build_training_dataset(intraday_df, daily_df, horizon_key):
    """
    Build feature matrix X and label vector y from historical intraday data.

    For each trading day with sufficient pre-market data:
    1. Extract pre-market features (4:00-9:30)
    2. Compute the label based on RTH performance for the given horizon
    """
    sessions = data.split_premarket_rth(intraday_df)
    X_rows = []
    y_labels = []
    dates = []

    horizon_minutes = config.HORIZONS[horizon_key]["minutes"]

    sorted_dates = sorted(sessions.keys())
    for session_date in sorted_dates:
        session = sessions[session_date]
        pm = session["premarket"]
        rth = session["rth"]

        if len(pm) < 10:  # Need enough pre-market bars
            continue

        # Get prev close from daily data
        daily_before = daily_df[daily_df.index.date < session_date]
        if len(daily_before) < 2:
            continue
        prev_close = float(daily_before["Close"].iloc[-1])

        # Compute features
        feat = features.compute_premarket_features(pm, daily_before, prev_close)
        if feat is None:
            continue

        # Compute label: direction from RTH open to horizon end
        if rth.empty:
            continue
        rth_open_price = float(rth["Open"].iloc[0])

        import pytz
        ET = pytz.timezone("US/Eastern")
        horizon_end_time = ET.localize(
            dt.datetime.combine(session_date, dt.time(9, 30))
        ) + dt.timedelta(minutes=horizon_minutes)

        rth_at_horizon = rth[rth.index <= horizon_end_time]
        if rth_at_horizon.empty:
            continue
        horizon_close = float(rth_at_horizon["Close"].iloc[-1])

        if rth_open_price == 0:
            continue
        ret = (horizon_close - rth_open_price) / rth_open_price * 100

        # Label: 1 = BULL (positive), 0 = BEAR (negative/flat)
        label = 1 if ret > 0 else 0

        feature_vec = [feat.get(fn, 0.0) for fn in FEATURE_NAMES]
        X_rows.append(feature_vec)
        y_labels.append(label)
        dates.append(session_date)

    X = np.array(X_rows, dtype=float)
    y = np.array(y_labels, dtype=int)
    return X, y, dates


def build_training_data_from_daily(daily_df, horizon_key):
    """
    Build training data using daily OHLCV only (for longer history).
    Uses previous day's data as proxy features when intraday isn't available.

    This allows training on 5+ years of data even though intraday
    is only available for ~60 days.
    """
    X_rows = []
    y_labels = []
    dates = []

    horizon_map = {"30m": 0.3, "morning": 0.6, "day": 1.0}
    horizon_weight = horizon_map.get(horizon_key, 1.0)

    for i in range(10, len(daily_df) - 1):
        row = daily_df.iloc[i]
        prev = daily_df.iloc[i - 1]
        window = daily_df.iloc[max(0, i - 5):i + 1]

        prev_close = float(prev["Close"])
        if prev_close == 0:
            continue

        today_open = float(row["Open"])
        today_close = float(row["Close"])
        today_high = float(row["High"])
        today_low = float(row["Low"])

        # Simulate pre-market features from daily data
        feat = {}
        feat["overnight_gap_pct"] = (today_open - prev_close) / prev_close * 100
        feat["pm_return_pct"] = feat["overnight_gap_pct"] * 0.5  # Proxy
        feat["pm_total_move_pct"] = feat["overnight_gap_pct"]

        day_range = today_high - today_low
        if today_open > 0:
            feat["pm_range_pct"] = day_range / today_open * 100 * 0.3  # PM has ~30% of day range
        else:
            feat["pm_range_pct"] = 0

        feat["pm_close_position"] = 0.5 + feat["overnight_gap_pct"] / 2  # Proxy
        feat["pm_close_position"] = np.clip(feat["pm_close_position"], 0, 1)

        feat["pm_total_volume"] = float(row["Volume"]) * 0.15  # PM ~15% of day volume
        feat["pm_volume_acceleration"] = 1.0 + np.random.normal(0, 0.1)
        feat["pm_volume_ramp"] = 1.2 + np.random.normal(0, 0.2)

        feat["pm_vwap_deviation_pct"] = feat["overnight_gap_pct"] * 0.3

        feat["pm_momentum"] = feat["overnight_gap_pct"] * 0.5
        feat["pm_trend_slope"] = feat["overnight_gap_pct"] * 0.01
        feat["pm_trend_r2"] = 0.5 + np.random.normal(0, 0.15)
        feat["pm_trend_r2"] = np.clip(feat["pm_trend_r2"], 0, 1)

        feat["pm_rsi"] = 50 + feat["overnight_gap_pct"] * 5
        feat["pm_rsi"] = np.clip(feat["pm_rsi"], 0, 100)

        # Yesterday's volatility as proxy
        if len(window) >= 5:
            wclose = window["Close"].astype(float).values
            wret = np.diff(wclose) / wclose[:-1]
            feat["pm_volatility"] = np.std(wret) * 100
        else:
            feat["pm_volatility"] = 1.0

        feat["pm_max_drawdown"] = max(0, -feat["overnight_gap_pct"]) * 0.5
        feat["pm_max_rally"] = max(0, feat["overnight_gap_pct"]) * 0.5

        feat["pm_body_ratio"] = abs(feat["overnight_gap_pct"]) / (feat["pm_range_pct"] + 0.01)
        feat["pm_body_ratio"] = np.clip(feat["pm_body_ratio"], 0, 1)
        feat["pm_upper_wick_ratio"] = 0.3
        feat["pm_lower_wick_ratio"] = 0.3

        # IBS
        prev_range = float(prev["High"]) - float(prev["Low"])
        if prev_range > 0:
            feat["yesterday_ibs"] = (float(prev["Close"]) - float(prev["Low"])) / prev_range
        else:
            feat["yesterday_ibs"] = 0.5

        # Yesterday return
        if i >= 2:
            day_before_prev = daily_df.iloc[i - 2]
            if float(day_before_prev["Close"]) > 0:
                feat["yesterday_return_pct"] = (
                    (prev_close - float(day_before_prev["Close"]))
                    / float(day_before_prev["Close"]) * 100
                )
            else:
                feat["yesterday_return_pct"] = 0
        else:
            feat["yesterday_return_pct"] = 0

        # 5-day return
        if i >= 6:
            five_ago_close = float(daily_df.iloc[i - 5]["Close"])
            if five_ago_close > 0:
                feat["five_day_return_pct"] = (prev_close - five_ago_close) / five_ago_close * 100
            else:
                feat["five_day_return_pct"] = 0
        else:
            feat["five_day_return_pct"] = 0

        # 5-day vol
        if i >= 6:
            rc = daily_df["Close"].iloc[i - 5:i + 1].astype(float).values
            dr = np.diff(rc) / rc[:-1]
            feat["five_day_vol"] = np.std(dr) * 100
        else:
            feat["five_day_vol"] = 1.0

        # Day of week
        if hasattr(daily_df.index[i], 'weekday'):
            dow = daily_df.index[i].weekday()
        else:
            dow = pd.Timestamp(daily_df.index[i]).weekday()
        feat["day_sin"] = np.sin(2 * np.pi * dow / 5)
        feat["day_cos"] = np.cos(2 * np.pi * dow / 5)

        # Label: next day direction scaled by horizon
        if i + 1 < len(daily_df):
            next_day = daily_df.iloc[i + 1]
            next_open = float(next_day["Open"])
            next_close = float(next_day["Close"])
            next_high = float(next_day["High"])
            next_low = float(next_day["Low"])

            if next_open == 0:
                continue

            # Approximate horizon return
            if horizon_key == "30m":
                # 30m return approximated by open-to-(open + fraction of range)
                ret = (next_open + (next_close - next_open) * 0.15 - next_open) / next_open * 100
                # Better proxy: use the first portion of the day's move
                day_ret = (next_close - next_open) / next_open * 100
                ret = day_ret * 0.3  # 30 min captures ~30% of daily
            elif horizon_key == "morning":
                day_ret = (next_close - next_open) / next_open * 100
                ret = day_ret * 0.6
            else:  # day
                ret = (next_close - next_open) / next_open * 100

            label = 1 if ret > 0 else 0
        else:
            continue

        feature_vec = [feat.get(fn, 0.0) for fn in FEATURE_NAMES]
        X_rows.append(feature_vec)
        y_labels.append(label)
        dates.append(daily_df.index[i])

    X = np.array(X_rows, dtype=float)
    y = np.array(y_labels, dtype=int)
    return X, y, dates


def create_ensemble():
    """Create the 3-classifier ensemble."""
    return {
        "rf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=20,
                max_features="sqrt", random_state=42, n_jobs=-1
            ))
        ]),
        "gb": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                min_samples_leaf=20, subsample=0.8, random_state=42
            ))
        ]),
        "lr": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.1, max_iter=1000, random_state=42
            ))
        ]),
    }


def train_models(force_retrain=False):
    """
    Train ensemble models for each horizon.
    Uses daily data for 5-year history.
    Returns training metadata.
    """
    meta_path = os.path.join(config.MODELS_DIR, "training_meta.json")

    if not force_retrain and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        # Check if models were trained today
        if meta.get("trained_date") == str(dt.date.today()):
            return meta

    print("Fetching historical data for training...")
    daily_df = data.fetch_historical_daily()
    if daily_df.empty:
        raise ValueError("No historical data available")

    # Replace NaN in daily_df
    daily_df = daily_df.ffill().bfill()

    meta = {"trained_date": str(dt.date.today()), "horizons": {}}

    for horizon_key in config.HORIZONS:
        print(f"Training {horizon_key} models...")
        X, y, dates = build_training_data_from_daily(daily_df, horizon_key)

        if len(X) < 100:
            print(f"  Insufficient data for {horizon_key}: {len(X)} samples")
            continue

        # Replace any NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        ensemble = create_ensemble()
        cpcv_results = {}

        for name, pipeline in ensemble.items():
            print(f"  Training {name}...")
            pipeline.fit(X, y)

            # Save model
            model_path = os.path.join(config.MODELS_DIR, f"{horizon_key}_{name}.pkl")
            joblib.dump(pipeline, model_path)

            # CPCV evaluation (use a subset for speed)
            print(f"  CPCV evaluation for {name}...")
            try:
                cv = CombinatorialPurgedCV(
                    n_splits=config.N_SPLITS,
                    n_test_groups=config.N_TEST_GROUPS,
                    purge_gap=config.PURGE_DAYS,
                    embargo_gap=config.EMBARGO_DAYS,
                )
                fold_accs = []
                for train_idx, test_idx in cv.split(X):
                    clone = type(pipeline.named_steps["clf"])(
                        **pipeline.named_steps["clf"].get_params()
                    )
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X[train_idx])
                    X_test = scaler.transform(X[test_idx])
                    clone.fit(X_train, y[train_idx])
                    acc = clone.score(X_test, y[test_idx])
                    fold_accs.append(acc)
                    if len(fold_accs) >= 15:  # Cap folds for speed
                        break
                cpcv_results[name] = {
                    "mean_accuracy": float(np.mean(fold_accs)),
                    "std_accuracy": float(np.std(fold_accs)),
                    "n_folds": len(fold_accs),
                }
            except Exception as e:
                cpcv_results[name] = {"error": str(e)}

        meta["horizons"][horizon_key] = {
            "n_samples": len(X),
            "class_balance": float(np.mean(y)),
            "cpcv": cpcv_results,
        }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("Training complete.")
    return meta


def predict(feature_dict, horizon_key):
    """
    Run ensemble prediction for a single observation.

    Returns:
        dict with direction, confidence, individual model predictions
    """
    feature_vec = np.array(
        [[feature_dict.get(fn, 0.0) for fn in FEATURE_NAMES]], dtype=float
    )
    feature_vec = np.nan_to_num(feature_vec, nan=0.0, posinf=0.0, neginf=0.0)

    results = {}
    probs = []

    for name in ["rf", "gb", "lr"]:
        model_path = os.path.join(config.MODELS_DIR, f"{horizon_key}_{name}.pkl")
        if not os.path.exists(model_path):
            continue
        pipeline = joblib.load(model_path)
        prob = pipeline.predict_proba(feature_vec)[0]
        bull_prob = prob[1] if len(prob) > 1 else prob[0]
        results[name] = {
            "bull_prob": float(bull_prob),
            "bear_prob": float(1 - bull_prob),
            "direction": "BULL" if bull_prob > 0.5 else "BEAR",
        }
        probs.append(bull_prob)

    if not probs:
        return None

    # Ensemble: average probabilities
    avg_bull_prob = float(np.mean(probs))
    confidence = max(avg_bull_prob, 1 - avg_bull_prob)
    direction = "BULL" if avg_bull_prob > 0.5 else "BEAR"

    # Apply confidence threshold
    show_signal = confidence >= config.CONFIDENCE_THRESHOLD
    high_conf = confidence >= config.HIGH_CONF_THRESHOLD

    return {
        "direction": direction,
        "confidence": round(confidence * 100),
        "bull_prob": round(avg_bull_prob, 3),
        "show_signal": show_signal,
        "high_confidence": high_conf,
        "models": results,
    }


def predict_all_horizons(feature_dict):
    """Run predictions for all horizons."""
    results = {}
    for horizon_key in config.HORIZONS:
        pred = predict(feature_dict, horizon_key)
        if pred is not None:
            results[horizon_key] = pred
    return results

"""Flask application for the Pre-Market Briefing dashboard."""
import datetime as dt
import json
import os
import traceback

import pytz
from flask import Flask, render_template, jsonify

import config
import data
import features
import models
import options_gex
import scorecard

app = Flask(__name__)
ET = pytz.timezone("US/Eastern")


def get_market_context():
    """Gather market context data."""
    prev_close, last_close = data.get_previous_close()
    vix = data.fetch_vix()

    # Fetch pre-market data
    pm_data = data.fetch_current_premarket()

    context = {
        "overnight_gap": None,
        "pm_move": None,
        "yesterday_rth": None,
        "yesterday_ibs": None,
        "vix": None,
        "vix_label": None,
    }

    if not pm_data.empty and prev_close:
        pm_open = float(pm_data["Open"].iloc[0])
        pm_close = float(pm_data["Close"].iloc[-1])
        context["overnight_gap"] = round((pm_open - prev_close) / prev_close * 100, 2)
        context["pm_move"] = round((pm_close - pm_open) / pm_open * 100, 2)

    if prev_close and last_close:
        context["yesterday_rth"] = round((last_close - prev_close) / prev_close * 100, 2)

    # Yesterday IBS from daily data
    try:
        daily = data.fetch_historical_daily(years=1)
        if not daily.empty and len(daily) >= 2:
            yesterday = daily.iloc[-1]
            y_range = float(yesterday["High"]) - float(yesterday["Low"])
            if y_range > 0:
                ibs = (float(yesterday["Close"]) - float(yesterday["Low"])) / y_range
                context["yesterday_ibs"] = round(ibs, 2)
                if ibs > 0.8:
                    context["ibs_label"] = "overbought"
                elif ibs < 0.2:
                    context["ibs_label"] = "oversold"
                else:
                    context["ibs_label"] = "neutral"
    except Exception:
        pass

    if vix is not None:
        context["vix"] = round(vix, 1)
        if vix >= 25:
            context["vix_label"] = "HIGH"
        elif vix >= 18:
            context["vix_label"] = "ELEVATED"
        else:
            context["vix_label"] = "LOW"

    return context


def run_predictions():
    """Run the full prediction pipeline."""
    now = dt.datetime.now(ET)
    today_str = now.strftime("%Y-%m-%d")

    # Ensure models are trained
    try:
        models.train_models()
    except Exception as e:
        print(f"Training error: {e}")
        return None

    # Get pre-market data
    pm_data = data.fetch_current_premarket()
    prev_close, _ = data.get_previous_close()

    # Get daily data for context features
    try:
        daily = data.fetch_historical_daily(years=1)
    except Exception:
        daily = None

    if pm_data.empty:
        # If no pre-market data, use daily-based features
        if daily is not None and not daily.empty and prev_close:
            feat = _build_features_from_daily(daily, prev_close)
        else:
            return None
    else:
        feat = features.compute_premarket_features(pm_data, daily, prev_close)

    if feat is None:
        return None

    # Add VIX to features if we have it
    vix = data.fetch_vix()

    # Run predictions
    predictions = models.predict_all_horizons(feat)

    result = {
        "date": today_str,
        "computed_at": now.strftime("%Y-%m-%d %H:%M:%S ET"),
        "predictions": predictions,
    }

    # Save predictions
    data.save_predictions(result)

    return result


def _build_features_from_daily(daily, prev_close):
    """Build proxy features when pre-market data isn't available."""
    import numpy as np
    feat = {}
    last = daily.iloc[-1]
    today_open = float(last["Close"])  # Use last close as proxy

    feat["overnight_gap_pct"] = 0.0
    feat["pm_return_pct"] = 0.0
    feat["pm_total_move_pct"] = 0.0
    feat["pm_range_pct"] = 0.0
    feat["pm_close_position"] = 0.5
    feat["pm_total_volume"] = float(last["Volume"]) * 0.15
    feat["pm_volume_acceleration"] = 1.0
    feat["pm_volume_ramp"] = 1.0
    feat["pm_vwap_deviation_pct"] = 0.0
    feat["pm_momentum"] = 0.0
    feat["pm_trend_slope"] = 0.0
    feat["pm_trend_r2"] = 0.0
    feat["pm_rsi"] = 50.0
    feat["pm_volatility"] = 0.0
    feat["pm_max_drawdown"] = 0.0
    feat["pm_max_rally"] = 0.0
    feat["pm_body_ratio"] = 0.0
    feat["pm_upper_wick_ratio"] = 0.0
    feat["pm_lower_wick_ratio"] = 0.0

    # IBS
    y_range = float(last["High"]) - float(last["Low"])
    if y_range > 0:
        feat["yesterday_ibs"] = (float(last["Close"]) - float(last["Low"])) / y_range
    else:
        feat["yesterday_ibs"] = 0.5

    if len(daily) >= 2:
        prev = daily.iloc[-2]
        if float(prev["Close"]) > 0:
            feat["yesterday_return_pct"] = (float(last["Close"]) - float(prev["Close"])) / float(prev["Close"]) * 100
        else:
            feat["yesterday_return_pct"] = 0

    if len(daily) >= 6:
        five_ago = float(daily.iloc[-6]["Close"])
        if five_ago > 0:
            feat["five_day_return_pct"] = (float(last["Close"]) - five_ago) / five_ago * 100
            rc = daily["Close"].iloc[-6:].astype(float).values
            dr = np.diff(rc) / rc[:-1]
            feat["five_day_vol"] = np.std(dr) * 100
        else:
            feat["five_day_return_pct"] = 0
            feat["five_day_vol"] = 0
    else:
        feat["five_day_return_pct"] = 0
        feat["five_day_vol"] = 0

    import numpy as np
    dow = dt.datetime.now(ET).weekday()
    feat["day_sin"] = np.sin(2 * np.pi * dow / 5)
    feat["day_cos"] = np.cos(2 * np.pi * dow / 5)

    return feat


@app.route("/")
def dashboard():
    """Render the main dashboard."""
    return render_template("dashboard.html")


@app.route("/guide")
def guide():
    """Render the interactive dashboard guide."""
    return render_template("guide.html")


@app.route("/api/briefing")
def api_briefing():
    """API: Full pre-market briefing data."""
    try:
        now = dt.datetime.now(ET)
        today_str = now.strftime("%Y-%m-%d")
        day_name = now.strftime("%A")

        # Check for macro event
        macro_event = config.MACRO_EVENTS.get(today_str)

        # Get market context
        context = get_market_context()

        # Run predictions
        pred_result = run_predictions()

        # Get previous predictions for display
        all_preds = data.load_predictions()
        prev_pred = all_preds.get(today_str, {})

        # Update actuals for past days
        try:
            scorecard.update_actuals()
        except Exception as e:
            print(f"Actuals update error: {e}")

        # Build scorecard
        try:
            sc = scorecard.build_scorecard()
        except Exception as e:
            print(f"Scorecard error: {e}")
            sc = {"rows": [], "summary": {}, "trained_date": "error", "n_days": 0}

        # Options/GEX analysis
        try:
            options = options_gex.get_full_options_analysis()
        except Exception as e:
            print(f"Options error: {e}")
            traceback.print_exc()
            options = None

        return jsonify({
            "date": today_str,
            "day_name": day_name,
            "computed_at": now.strftime("%Y-%m-%d %H:%M:%S ET"),
            "macro_event": macro_event,
            "context": context,
            "predictions": pred_result.get("predictions", {}) if pred_result else {},
            "previous_predictions": prev_pred,
            "scorecard": sc,
            "options": options,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/train", methods=["POST"])
def api_train():
    """API: Force retrain models."""
    try:
        meta = models.train_models(force_retrain=True)
        return jsonify({"status": "ok", "meta": meta})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5050, host="127.0.0.1")

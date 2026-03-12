"""Scorecard: track predictions vs actual outcomes."""
import datetime as dt
import json
import os

import numpy as np
import pytz

import config
import data

ET = pytz.timezone("US/Eastern")


def update_actuals():
    """
    Update actual outcomes for past predictions.
    Fetches RTH data and computes actual returns for each horizon.
    """
    predictions = data.load_predictions()
    actuals = data.load_actuals()

    if not predictions:
        return

    # Get recent intraday data
    intraday = data.fetch_intraday_history(duration="30 D", bar_size="5 mins")
    if intraday.empty:
        return

    for date_str, pred in predictions.items():
        if date_str in actuals and actuals[date_str].get("complete"):
            continue

        pred_date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
        today = dt.datetime.now(ET).date()

        # Skip today and future dates
        if pred_date >= today:
            continue

        # Get RTH data for this date
        rth_start = ET.localize(dt.datetime.combine(pred_date, dt.time(9, 30)))
        rth_end = ET.localize(dt.datetime.combine(pred_date, dt.time(16, 0)))

        day_data = intraday[(intraday.index >= rth_start) & (intraday.index <= rth_end)]
        if day_data.empty:
            continue

        rth_open = float(day_data["Open"].iloc[0])
        if rth_open == 0:
            continue

        actual = {"date": date_str, "rth_open": rth_open}

        # 30-minute return
        t_30m = rth_start + dt.timedelta(minutes=30)
        data_30m = day_data[day_data.index <= t_30m]
        if not data_30m.empty:
            close_30m = float(data_30m["Close"].iloc[-1])
            ret_30m = (close_30m - rth_open) / rth_open * 100
            actual["30m"] = {
                "return_pct": round(ret_30m, 2),
                "direction": classify_return(ret_30m),
            }

        # Morning return (9:30-11:00)
        t_morn = rth_start + dt.timedelta(minutes=90)
        data_morn = day_data[day_data.index <= t_morn]
        if not data_morn.empty:
            close_morn = float(data_morn["Close"].iloc[-1])
            ret_morn = (close_morn - rth_open) / rth_open * 100
            actual["morning"] = {
                "return_pct": round(ret_morn, 2),
                "direction": classify_return(ret_morn),
            }

        # Full day return
        close_day = float(day_data["Close"].iloc[-1])
        ret_day = (close_day - rth_open) / rth_open * 100
        actual["day"] = {
            "return_pct": round(ret_day, 2),
            "direction": classify_return(ret_day),
        }

        actual["complete"] = True
        data.save_actuals(actual)


def classify_return(ret_pct):
    """Classify a return as BULL, BEAR, or FLAT."""
    if ret_pct > config.FLAT_THRESHOLD:
        return "BULL"
    elif ret_pct < -config.FLAT_THRESHOLD:
        return "BEAR"
    return "FLAT"


def build_scorecard(n_days=config.SCORECARD_DAYS):
    """
    Build the scorecard comparing predictions to actuals.
    Returns list of daily results and summary stats.
    """
    predictions = data.load_predictions()
    actuals = data.load_actuals()

    # Sort by date descending
    all_dates = sorted(set(predictions.keys()) | set(actuals.keys()), reverse=True)
    all_dates = all_dates[:n_days]

    rows = []
    horizon_stats = {h: {"correct": 0, "total": 0, "high_conf_correct": 0, "high_conf_total": 0}
                     for h in config.HORIZONS}

    for date_str in all_dates:
        pred = predictions.get(date_str, {})
        act = actuals.get(date_str, {})
        row = {"date": date_str}

        for horizon_key in config.HORIZONS:
            h_pred = pred.get("predictions", {}).get(horizon_key, {})
            h_act = act.get(horizon_key, {})

            if h_pred and h_act:
                pred_dir = h_pred.get("direction", "")
                pred_conf = h_pred.get("confidence", 0)
                show_signal = h_pred.get("show_signal", False)
                act_dir = h_act.get("direction", "")
                act_ret = h_act.get("return_pct", 0)

                # Determine if prediction was correct
                if act_dir == "FLAT":
                    match = "--"  # Flat days are inconclusive
                elif pred_dir == act_dir:
                    match = "Y"
                else:
                    match = "X"

                row[horizon_key] = {
                    "pred_direction": pred_dir,
                    "pred_confidence": pred_conf,
                    "show_signal": show_signal,
                    "actual_direction": act_dir,
                    "actual_return": act_ret,
                    "match": match,
                }

                # Update stats
                if act_dir != "FLAT":
                    horizon_stats[horizon_key]["total"] += 1
                    if pred_dir == act_dir:
                        horizon_stats[horizon_key]["correct"] += 1
                    if pred_conf >= config.HIGH_CONF_THRESHOLD * 100:
                        horizon_stats[horizon_key]["high_conf_total"] += 1
                        if pred_dir == act_dir:
                            horizon_stats[horizon_key]["high_conf_correct"] += 1
            else:
                row[horizon_key] = {
                    "pred_direction": h_pred.get("direction", "--"),
                    "pred_confidence": h_pred.get("confidence", 0),
                    "show_signal": h_pred.get("show_signal", False),
                    "actual_direction": h_act.get("direction", "--") if h_act else "--",
                    "actual_return": h_act.get("return_pct", 0) if h_act else None,
                    "match": "--",
                }

        rows.append(row)

    # Compute summary
    summary = {}
    for h in config.HORIZONS:
        stats = horizon_stats[h]
        total = stats["total"]
        correct = stats["correct"]
        hc_total = stats["high_conf_total"]
        hc_correct = stats["high_conf_correct"]

        summary[h] = {
            "accuracy": round(correct / total * 100, 1) if total > 0 else 0,
            "record": f"{correct}/{total} all",
            "high_conf_accuracy": round(hc_correct / hc_total * 100, 1) if hc_total > 0 else 0,
            "high_conf_record": f"{hc_correct}/{hc_total} high-conf",
        }

    return {
        "rows": rows,
        "summary": summary,
        "trained_date": _get_trained_date(),
        "n_days": len(rows),
    }


def _get_trained_date():
    """Get the model training date."""
    meta_path = os.path.join(config.MODELS_DIR, "training_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("trained_date", "unknown")
    return "not trained"

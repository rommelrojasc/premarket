# Pre-Market Briefing Dashboard

A machine learning-powered pre-market analysis tool for SPY that combines predictive models, options flow analysis, and gamma exposure data into a single dark-themed dashboard. Designed to be read before the opening bell to form a directional bias for the trading day.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## What It Does

1. **Pulls pre-market data** from the 4:00-9:30 AM ET window via IBKR Gateway
2. **Engineers 25 features** from pre-market price action — volume ramp, VWAP deviation, momentum, volatility, candle patterns, and context (IBS, cyclical day encoding)
3. **Runs 3 ML classifiers** (Random Forest, Gradient Boosting, Logistic Regression) across 3 time horizons (30 min, morning, full day)
4. **Hides weak signals** — predictions below 60% confidence show "NO EDGE" to prevent anchoring bias
5. **Analyzes options flow** — put/call walls, max pain, GEX flip level, and gamma zone (+/- gamma)
6. **Generates trading scenarios** — concrete if/then table based on the current gamma regime
7. **Tracks accuracy** — 15-day scorecard comparing predictions to actual outcomes

## Requirements

- **IBKR Gateway or TWS** running on your machine (paper or live)
- **Python 3.10+** (tested on 3.14)
- **Market data subscription** for SPY and SPY options via IBKR

## Installation

```bash
git clone https://github.com/rommelrojasc/premarket.git
cd premarket
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

All settings are in `config.py`. Key values:

| Setting | Default | Description |
|---------|---------|-------------|
| `IB_HOST` | `127.0.0.1` | IBKR Gateway host |
| `IB_PORT` | `4002` | Gateway port (4001=live, 4002=paper) |
| `IB_CLIENT_ID` | `10` | Client ID (must be unique across your IBKR apps) |
| `CONFIDENCE_THRESHOLD` | `0.60` | Below this, prediction shows "NO EDGE" |
| `HIGH_CONF_THRESHOLD` | `0.75` | Above this, prediction gets "HIGH CONF" badge |
| `TRAINING_YEARS` | `5` | Years of daily data used for model training |
| `OPTIONS_EXPIRY_DAYS` | `7` | How many days of option expirations to include |
| `SCORECARD_DAYS` | `15` | Days shown in the prediction scorecard |

You can also set `IB_HOST`, `IB_PORT`, and `IB_CLIENT_ID` via environment variables.

### Macro Events Calendar

Edit the `MACRO_EVENTS` dict in `config.py` to flag days with scheduled releases (CPI, PPI, FOMC, NFP, etc.). The dashboard shows an orange warning banner on these days since pre-market signals may be unreliable.

```python
MACRO_EVENTS = {
    "2026-03-11": "CPI",
    "2026-03-12": "PPI",
    "2026-03-19": "FOMC",
}
```

## Usage

### Start the dashboard

```bash
python run.py
```

Then open http://127.0.0.1:5050 in your browser.

The first load takes ~60-90 seconds (connects to IBKR, fetches 5 years of daily data, trains models). Subsequent loads are fast since models are cached daily.

### CLI options

```bash
python run.py --train              # Force retrain models
python run.py --port 8080          # Custom dashboard port
python run.py --host 0.0.0.0      # Listen on all interfaces
python run.py --ib-port 4001      # Connect to live gateway
python run.py --ib-client-id 20   # Override client ID
```

### Force retrain via API

```bash
curl -X POST http://127.0.0.1:5050/api/train
```

## Dashboard Sections

### Prediction Cards

Three cards for each time horizon:
- **First 30 Min** (9:30-10:00) — direction of SPY in the first half hour
- **Morning** (9:30-11:00) — direction through the morning session
- **Full Day** (9:30-16:00) — close vs open direction

Each shows BULL/BEAR with confidence %, or **NO EDGE** when the ensemble doesn't have a strong enough signal. Confidence above 75% gets a **HIGH CONF** badge.

### Market Context

- **Overnight Gap** — % change from yesterday's close to pre-market open
- **Pre-market Move** — % change within the pre-market session
- **Yesterday RTH** — yesterday's regular session return
- **Yesterday IBS** — Internal Bar Strength: where yesterday closed within its range (< 0.2 = oversold/bullish, > 0.8 = overbought/bearish)
- **VIX** — volatility index with LOW/ELEVATED/HIGH labels

### Options & GEX

- **Put Wall** — highest put OI strike below price (support)
- **Call Wall** — highest call OI strike above price (resistance)
- **Max Pain** — strike where most options expire worthless (price magnet near expiry)
- **GEX Flip** — price level where gamma exposure flips sign
- **Gamma Zone** — positive gamma (moves dampened, mean-revert) vs negative gamma (moves amplified, trend)
- **Trading Scenarios** — if/then table with suggested trades based on gamma regime

### Scorecard

15-day history comparing predictions to actual outcomes:
- **Y** = correct direction
- **X** = wrong direction
- **--** = flat day (< 0.10% move) or no data

Tracks both overall accuracy and high-confidence accuracy separately.

### Interactive Guide

Visit http://127.0.0.1:5050/guide for a detailed walkthrough of every section, term, and the recommended daily sequence with visual examples.

## Architecture

```
premarket/
├── run.py              # Entry point with CLI args
├── app.py              # Flask routes: /, /guide, /api/briefing, /api/train
├── config.py           # All configuration in one place
├── ib_compat.py        # Python 3.14+ event loop fix for ib_insync
├── ib_client.py        # Thread-safe singleton IBKR connection manager
├── data.py             # Market data fetching via IBKR (daily, intraday, VIX)
├── features.py         # 25 pre-market feature engineering functions
├── models.py           # ML pipeline: RF, GB, LR with StandardScaler
├── validation.py       # Combinatorial Purged Cross-Validation (CPCV)
├── options_gex.py      # Options chain analysis, walls, max pain, GEX
├── scorecard.py        # Prediction tracking and accuracy computation
├── templates/
│   ├── dashboard.html  # Main dashboard (single-page with JS fetch)
│   └── guide.html      # Interactive guide with explanations
├── static/
│   └── style.css       # Dark theme stylesheet
├── data/               # (gitignored) Prediction/actuals JSON files
├── models_store/       # (gitignored) Cached trained models
└── requirements.txt
```

## ML Details

### Features (25 total)

| Category | Features |
|----------|----------|
| **Price** | Overnight gap, PM return, PM range, PM close position |
| **Volume** | Total PM volume, volume acceleration, volume ramp |
| **VWAP** | VWAP deviation % |
| **Momentum** | Slope, R-squared, RSI |
| **Volatility** | Std deviation, max drawdown, max rally |
| **Candle** | Body ratio, upper wick ratio, lower wick ratio |
| **Context** | IBS, yesterday return, 5-day return, 5-day volatility, cyclical day (sin/cos) |

### Models

- **Random Forest** — 300 trees, max depth 8
- **Gradient Boosting** — 200 trees, max depth 4, learning rate 0.05
- **Logistic Regression** — C=0.1 with L2 regularization

All wrapped in `StandardScaler` pipelines. Final prediction averages probabilities across the three models.

### Validation

Uses **Combinatorial Purged Cross-Validation (CPCV)** instead of standard k-fold to prevent look-ahead bias in financial time series:
- 10 splits, 2 test groups
- 5-day purge gap between train/test
- 2-day embargo after each test fold

### Training Data

Uses 5 years of daily bars from IBKR with proxy pre-market features derived from daily OHLCV (since IBKR only provides ~60 days of intraday history). Models retrain automatically once per day on first load.

## Options & GEX Details

### Gamma Fallback

IBKR's `modelGreeks.gamma` is not always available. The system uses a 3-tier fallback:
1. IBKR `modelGreeks.gamma` (preferred)
2. Black-Scholes gamma computed from IBKR's implied volatility
3. Black-Scholes gamma with default 20% volatility (last resort)

### Strike Filtering

Options chains are filtered to strikes within ±30 points of the current price with integer strikes only, limited to 20 strikes per expiration across the 3 nearest expirations. This keeps IBKR request volume manageable.

## Daily Workflow

| Time (ET) | Action |
|-----------|--------|
| **7:00-8:00 AM** | Primary read — refresh dashboard, form directional bias |
| **8:30 AM** | Macro event drop (CPI/PPI/NFP days) — wait, then refresh |
| **9:00-9:25 AM** | Final read — check predictions, options/GEX, pick scenario |
| **9:30 AM** | Market open — execute based on your read |
| **2:00-4:00 PM** | Max pain gravity strongest — check "Into close" scenario |
| **After close** | Next load auto-updates scorecard with actual outcomes |

## Troubleshooting

### "Options data unavailable"
IBKR can't provide option chain data outside market hours. This section populates during pre-market (4 AM+) and RTH.

### All predictions show "NO EDGE"
Normal when loading before pre-market starts or when models are using proxy features. Load between 7-9 AM for predictions based on real pre-market data.

### Connection errors
Ensure IBKR Gateway/TWS is running and the port matches your config. Paper trading uses port 4002, live uses 4001. Check that no other app is using the same `IB_CLIENT_ID`.

### Slow first load
The first load of the day fetches 5 years of daily data and trains 9 models (3 per horizon). This takes ~60-90 seconds. Models are cached in `models_store/` and reused until the next day.

### Python 3.14 compatibility
The `ib_compat.py` module handles the event loop changes in Python 3.14+. It must be imported before `ib_insync` in any file that uses it. This is already set up in all project files.

## License

MIT

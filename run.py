#!/usr/bin/env python3
"""
Pre-Market Analysis System (IBKR)
==================================
Run this script to start the dashboard.
Requires IB Gateway or TWS running on the configured port.

Usage:
    python run.py                    # Start dashboard (trains models if needed)
    python run.py --train            # Force retrain models
    python run.py --port 8080        # Custom dashboard port
    python run.py --ib-port 4001     # Connect to live gateway
"""
import asyncio
import argparse
import sys
import os

# Python 3.14+ event loop fix for ib_insync
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Pre-Market Analysis Dashboard (IBKR)")
    parser.add_argument("--train", action="store_true", help="Force retrain models")
    parser.add_argument("--port", type=int, default=5050, help="Dashboard port")
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard host")
    parser.add_argument("--ib-port", type=int, default=None, help="Override IB Gateway port")
    parser.add_argument("--ib-client-id", type=int, default=None, help="Override IB client ID")
    args = parser.parse_args()

    import config
    if args.ib_port:
        config.IB_PORT = args.ib_port
    if args.ib_client_id:
        config.IB_CLIENT_ID = args.ib_client_id

    if args.train:
        import models
        print("Force retraining models...")
        meta = models.train_models(force_retrain=True)
        print(f"Training complete: {meta['trained_date']}")
        for h, info in meta.get("horizons", {}).items():
            print(f"  {h}: {info['n_samples']} samples, balance={info['class_balance']:.2f}")
            for m, cv in info.get("cpcv", {}).items():
                if "mean_accuracy" in cv:
                    print(f"    {m} CPCV: {cv['mean_accuracy']:.3f} +/- {cv['std_accuracy']:.3f}")
        print()

    from app import app
    print(f"Starting Pre-Market Briefing at http://{args.host}:{args.port}")
    print(f"IBKR Gateway: {config.IB_HOST}:{config.IB_PORT} (clientId={config.IB_CLIENT_ID})")
    print("First load will connect to IBKR and train models (~60s).")
    app.run(debug=False, port=args.port, host=args.host)


if __name__ == "__main__":
    main()
